"""Ray trainer integration for Tool-OPD."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.tool_opd_utils import (
    ExperienceReplay,
    combine_reliability,
    compute_outcome_asymmetric_credit,
    compute_teacher_confidence,
    experience_weight,
    select_confidence_candidates,
)
from verl.utils.model import compute_position_id_with_mask


HINDSIGHT_PROMPT = """You are an expert tool-use teacher.
Trajectory:
{trajectory}

Outcome: {outcome}

Induce exactly one short, reusable tool-use strategy.
For success, capture the effective workflow or pivotal decision.
For failure, capture the root error and how to avoid it. Use this exact format:
<skill>Rule Name: imperative guidance. Apply when the relevant condition holds.</skill>
Use at most 50 words. Do not include analysis, scores, or a replacement trajectory.
"""

_ACTION_RE = re.compile(r"<(action|search|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.IGNORECASE | re.DOTALL)
_CANDIDATE_BLOCK_RE = re.compile(
    r"admissible actions.*?:\s*\[(.*?)\]\s*\.", re.IGNORECASE | re.DOTALL
)
_CURRENT_OBS_RE = re.compile(
    r"current observation is:\s*(.*?)\s*Your admissible actions", re.IGNORECASE | re.DOTALL
)
_TASK_RE = re.compile(r"Your task is to:\s*(.*?)\s*(?:Prior to this step|You are now)", re.IGNORECASE | re.DOTALL)


def _valid_prompt_text(batch: DataProto, row: int, tokenizer) -> str:
    response_length = batch.batch["responses"].shape[-1]
    prompt_ids = batch.batch["input_ids"][row, :-response_length]
    prompt_mask = batch.batch["attention_mask"][row, :-response_length].bool()
    return tokenizer.decode(prompt_ids[prompt_mask], skip_special_tokens=False)


def _valid_response_text(batch: DataProto, row: int, tokenizer) -> str:
    response_ids = batch.batch["responses"][row]
    response_mask = batch.batch["attention_mask"][row, -response_ids.shape[-1] :].bool()
    return tokenizer.decode(response_ids[response_mask], skip_special_tokens=True)


def _extract_action(response_text: str) -> str:
    match = _ACTION_RE.search(response_text)
    return match.group(2).strip() if match else ""


def _extract_candidates(prompt_text: str) -> list[str]:
    match = _CANDIDATE_BLOCK_RE.search(prompt_text)
    if match is None:
        return []
    return [candidate.strip() for candidate in re.findall(r"'([^']+)'", match.group(1)) if candidate.strip()]


def _inject_skill(prompt_text: str, skill: str) -> str:
    """Inject the hindsight skill into the user message when using ChatML."""
    skill_context = f"[Hindsight Skill]\n{skill}\n\n"
    user_marker = "<|im_start|>user\n"
    marker_index = prompt_text.find(user_marker)
    if marker_index < 0:
        return skill_context + prompt_text
    insertion_point = marker_index + len(user_marker)
    return prompt_text[:insertion_point] + skill_context + prompt_text[insertion_point:]


def _state_signature(prompt_text: str) -> str:
    observation = _CURRENT_OBS_RE.search(prompt_text)
    task = _TASK_RE.search(prompt_text)
    normalized = " | ".join(
        re.sub(r"\s+", " ", value.group(1)).strip().lower()
        for value in (task, observation)
        if value is not None
    )
    if not normalized:
        normalized = re.sub(r"\s+", " ", prompt_text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def build_action_token_mask(batch: DataProto, tokenizer) -> torch.Tensor:
    """Mark response tokens inside an action, search, or final-answer tag."""
    responses = batch.batch["responses"]
    response_mask = batch.batch["attention_mask"][:, -responses.shape[-1] :].bool()
    result = torch.zeros_like(responses, dtype=torch.float32)
    for row in range(responses.shape[0]):
        valid_positions = response_mask[row].nonzero(as_tuple=False).flatten().tolist()
        ids = responses[row, valid_positions].tolist()
        if not ids:
            continue
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        match = _ACTION_RE.search(text)
        if match is None:
            continue
        content_start, content_end = match.span(2)
        previous_end = 0
        for local_index in range(len(ids)):
            current_end = len(
                tokenizer.decode(
                    ids[: local_index + 1],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            if current_end > content_start and previous_end < content_end:
                result[row, valid_positions[local_index]] = 1
            previous_end = current_end
    return result


def _pad_prompt_batch(texts: list[str], tokenizer, max_prompt_length: int) -> DataProto:
    input_ids = []
    attention_masks = []
    raw_prompt_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[-max_prompt_length:]
        raw_prompt_ids.append(ids)
        padding = max_prompt_length - len(ids)
        input_ids.append([tokenizer.pad_token_id] * padding + ids)
        attention_masks.append([0] * padding + [1] * len(ids))
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_tensor = torch.tensor(attention_masks, dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "input_ids": input_tensor,
            "attention_mask": attention_tensor,
            "position_ids": compute_position_id_with_mask(attention_tensor),
        },
        non_tensors={"raw_prompt_ids": np.array(raw_prompt_ids, dtype=object)},
        meta_info={"do_sample": False},
        auto_padding=True,
    )


def _teacher_forcing_batch(
    prompt_texts: list[str],
    response_id_rows: list[list[int]],
    tokenizer,
    max_prompt_length: int,
    response_length: int,
) -> DataProto:
    prompt_batch = _pad_prompt_batch(prompt_texts, tokenizer, max_prompt_length)
    padded_responses = []
    response_masks = []
    for ids in response_id_rows:
        ids = ids[:response_length]
        padding = response_length - len(ids)
        padded_responses.append(ids + [tokenizer.pad_token_id] * padding)
        response_masks.append([1] * len(ids) + [0] * padding)
    responses = torch.tensor(padded_responses, dtype=torch.long)
    response_mask = torch.tensor(response_masks, dtype=torch.long)
    full_ids = torch.cat([prompt_batch.batch["input_ids"], responses], dim=-1)
    full_mask = torch.cat([prompt_batch.batch["attention_mask"], response_mask], dim=-1)
    return DataProto.from_dict(
        tensors={
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "position_ids": compute_position_id_with_mask(full_mask),
            "responses": responses,
        },
        auto_padding=True,
    )


class ToolOPDRayTrainer(RayPPOTrainer):
    """Pure on-policy distillation trainer with outcome-aware turn weights."""

    def __init__(self, *args, teacher_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_tokenizer is None:
            raise ValueError("Tool-OPD requires a teacher tokenizer")
        if teacher_tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError("Tool-OPD currently requires Teacher and Student to share an identical tokenizer")
        self.teacher_tokenizer = teacher_tokenizer
        cfg = self.config.algorithm.tool_opd
        self.beta_h = cfg.get("beta_h", 5.0)
        self.ablation_teacher_target = cfg.get("ablation_teacher_target", "skill")
        self.ablation_weight_mode = cfg.get("ablation_weight_mode", "h")
        if self.ablation_teacher_target not in {"base", "skill"}:
            raise ValueError("ablation_teacher_target must be either 'base' or 'skill'")
        if self.ablation_weight_mode not in {"uniform", "h"}:
            raise ValueError("ablation_weight_mode must be either 'uniform' or 'h'")
        if self.ablation_teacher_target == "base" and self.ablation_weight_mode != "uniform":
            raise ValueError("The base-teacher ablation only supports uniform weights")
        self.replay = ExperienceReplay(alpha=cfg.get("replay_alpha", 1.0))
        self.lambda_e_start = cfg.get("lambda_e_start", 0.2)
        self.lambda_e_end = cfg.get("lambda_e_end", 0.5)
        self.teacher_max_prompt_length = cfg.get("teacher_max_prompt_length", self.config.data.max_prompt_length)
        self.hindsight_max_prompt_length = cfg.get(
            "hindsight_max_prompt_length", self.teacher_max_prompt_length
        )
        self.max_confidence_candidates = int(cfg.get("max_confidence_candidates", 4))
        self.confidence_score_batch_size = int(cfg.get("confidence_score_batch_size", 2048))
        if self.max_confidence_candidates <= 0:
            raise ValueError("max_confidence_candidates must be positive")
        if self.confidence_score_batch_size <= 0:
            raise ValueError("confidence_score_batch_size must be positive")

    def _generate_hindsight_skills(self, batch: DataProto) -> list[str]:
        grouped_rows = defaultdict(list)
        for row, trajectory_id in enumerate(batch.non_tensor_batch["traj_uid"]):
            grouped_rows[str(trajectory_id)].append(row)

        trajectory_ids = list(grouped_rows)
        prompts = []
        for trajectory_id in trajectory_ids:
            rows = sorted(grouped_rows[trajectory_id], key=lambda row: int(batch.non_tensor_batch["turn_step"][row]))
            turns = []
            for row in rows:
                turns.append(
                    f"[Turn {int(batch.non_tensor_batch['turn_step'][row])}]\n"
                    f"State:\n{_valid_prompt_text(batch, row, self.tokenizer)}\n"
                    f"Student response:\n{_valid_response_text(batch, row, self.tokenizer)}"
                )
            outcome = int(float(batch.non_tensor_batch["episode_rewards"][rows[0]]) > 0)
            prompts.append(HINDSIGHT_PROMPT.format(outcome=outcome, trajectory="\n\n".join(turns)))

        generation_batch = _pad_prompt_batch(
            prompts, self.teacher_tokenizer, self.hindsight_max_prompt_length
        )
        generated = self.teacher_wg.generate_sequences(generation_batch)
        generated_responses = generated.batch["responses"]
        skills_by_trajectory = {}
        for index, trajectory_id in enumerate(trajectory_ids):
            text = self.teacher_tokenizer.decode(generated_responses[index], skip_special_tokens=True)
            match = _SKILL_RE.search(text)
            skills_by_trajectory[trajectory_id] = (match.group(1) if match else text).strip()
        return [skills_by_trajectory[str(trajectory_id)] for trajectory_id in batch.non_tensor_batch["traj_uid"]]

    def _skill_teacher_batch(self, batch: DataProto, skills: list[str]) -> DataProto:
        response_mask = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1] :].bool()
        prompts = [
            _inject_skill(_valid_prompt_text(batch, row, self.tokenizer), skills[row])
            for row in range(len(skills))
        ]
        responses = [batch.batch["responses"][row, response_mask[row]].tolist() for row in range(len(skills))]
        return _teacher_forcing_batch(
            prompts,
            responses,
            self.teacher_tokenizer,
            self.teacher_max_prompt_length,
            batch.batch["responses"].shape[-1],
        )

    def _teacher_confidence(self, batch: DataProto, skills: list[str]) -> torch.Tensor:
        flat_prompts = []
        flat_responses = []
        owners = []
        candidate_lists = []
        for row in range(len(skills)):
            prompt = _valid_prompt_text(batch, row, self.tokenizer)
            response = _valid_response_text(batch, row, self.tokenizer)
            candidates = _extract_candidates(prompt)
            candidates = select_confidence_candidates(
                candidates, _extract_action(response), self.max_confidence_candidates
            )
            candidate_lists.append(candidates)
            action_tag_end = response.lower().find("<action>")
            prefix = response[: action_tag_end + len("<action>")] if action_tag_end >= 0 else response
            conditioned_prompt = _inject_skill(prompt, skills[row]) + prefix
            for candidate in candidates:
                flat_prompts.append(conditioned_prompt)
                flat_responses.append(self.teacher_tokenizer.encode(candidate, add_special_tokens=False))
                owners.append(row)

        if not flat_prompts:
            return torch.ones(len(skills), dtype=torch.float32)
        max_candidate_length = max(len(ids) for ids in flat_responses)
        score_parts = []
        for start in range(0, len(flat_prompts), self.confidence_score_batch_size):
            stop = start + self.confidence_score_batch_size
            score_batch = _teacher_forcing_batch(
                flat_prompts[start:stop],
                flat_responses[start:stop],
                self.teacher_tokenizer,
                self.teacher_max_prompt_length,
                max_candidate_length,
            )
            mask = score_batch.batch["attention_mask"][:, -max_candidate_length:]
            output = self.teacher_wg.compute_log_prob(score_batch)
            log_probs = output.batch["old_log_probs"]
            score_parts.append((log_probs * mask.to(log_probs)).sum(dim=-1))
        sequence_scores = torch.cat(score_parts)

        max_candidates = max(max(len(candidates), 1) for candidates in candidate_lists)
        score_matrix = torch.full((len(skills), max_candidates), float("-inf"))
        candidate_mask = torch.zeros((len(skills), max_candidates), dtype=torch.bool)
        offsets = defaultdict(int)
        for score, owner in zip(sequence_scores, owners):
            column = offsets[owner]
            score_matrix[owner, column] = score
            candidate_mask[owner, column] = True
            offsets[owner] += 1
        return compute_teacher_confidence(score_matrix, candidate_mask)

    def prepare_actor_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Attach Tool-OPD supervision immediately before the actor update."""
        action_mask = build_action_token_mask(batch, self.tokenizer)
        base_log_probs = None
        if self.ablation_teacher_target == "base" or self.ablation_weight_mode == "h":
            base_output = self.teacher_wg.compute_log_prob(batch)
            base_log_probs = base_output.batch["old_log_probs"]

        if self.ablation_teacher_target == "skill":
            skills = self._generate_hindsight_skills(batch)
            skill_batch = self._skill_teacher_batch(batch, skills)
            skill_output = self.teacher_wg.compute_log_prob(skill_batch)
            teacher_log_probs = skill_output.batch["old_log_probs"]
        else:
            teacher_log_probs = base_log_probs

        metrics = {}
        if self.ablation_weight_mode == "h":
            outcomes = torch.tensor(
                [int(float(reward) > 0) for reward in batch.non_tensor_batch["episode_rewards"]],
                dtype=teacher_log_probs.dtype,
            )
            _, turn_weights = compute_outcome_asymmetric_credit(
                base_log_probs, teacher_log_probs, outcomes, action_mask, self.beta_h
            )
            metrics["tool_opd/outcome_relevance_mean"] = turn_weights.mean().item()
        else:
            turn_weights = torch.ones(
                teacher_log_probs.shape[0],
                dtype=teacher_log_probs.dtype,
                device=teacher_log_probs.device,
            )

        batch.batch["teacher_skill_log_probs"] = teacher_log_probs
        batch.batch["tool_opd_action_mask"] = action_mask
        batch.batch["tool_opd_turn_weights"] = turn_weights
        valid_actions = action_mask.sum().clamp_min(1)
        metrics.update(
            {
                "tool_opd/turn_weight_mean": turn_weights.mean().item(),
                "tool_opd/action_tokens": valid_actions.item(),
            }
        )
        return batch, metrics
