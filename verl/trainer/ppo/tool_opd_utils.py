"""Core utilities for Tool-OPD.

The functions in this module are model- and environment-agnostic. Model
forward passes and trajectory plumbing live in ``tool_opd_ray_trainer.py``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Iterable

import torch


def select_confidence_candidates(
    candidates: list[str], chosen_action: str, limit: int
) -> list[str]:
    """Bound confidence scoring while retaining the Student's chosen action."""
    if limit <= 0 or len(candidates) <= limit:
        return candidates

    normalized_action = chosen_action.strip().lower()
    chosen = next(
        (candidate for candidate in candidates if candidate.strip().lower() == normalized_action),
        None,
    )
    if chosen is None:
        return candidates[:limit]
    return [chosen, *(candidate for candidate in candidates if candidate != chosen)][:limit]


def compute_outcome_asymmetric_credit(
    teacher_base_log_probs: torch.Tensor,
    teacher_skill_log_probs: torch.Tensor,
    outcomes: torch.Tensor,
    action_mask: torch.Tensor,
    beta_h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token credit ``c`` and its action-token turn mean ``H``.

    ``outcomes`` contains one binary episode outcome per batch row. All other
    tensors have shape ``(batch, response_length)``.
    """
    if teacher_base_log_probs.shape != teacher_skill_log_probs.shape:
        raise ValueError("teacher log-prob tensors must have the same shape")
    if action_mask.shape != teacher_skill_log_probs.shape:
        raise ValueError("action_mask must match teacher log-prob shape")
    if outcomes.ndim != 1 or outcomes.shape[0] != action_mask.shape[0]:
        raise ValueError("outcomes must have shape (batch,)")

    mask = action_mask.to(dtype=teacher_skill_log_probs.dtype)
    direction = outcomes.to(dtype=teacher_skill_log_probs.dtype).mul(2).sub(1).unsqueeze(-1)
    delta = teacher_skill_log_probs - teacher_base_log_probs
    token_credit = (2 * torch.sigmoid(beta_h * direction * delta) - 1).clamp_min(0) * mask
    action_count = mask.sum(dim=-1)
    turn_credit = token_credit.sum(dim=-1) / action_count.clamp_min(1)
    turn_credit = torch.where(action_count > 0, turn_credit, torch.zeros_like(turn_credit))
    return token_credit, turn_credit


def compute_teacher_confidence(
    candidate_log_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return normalized confidence from a distribution over admissible actions.

    ``candidate_log_scores`` and ``candidate_mask`` have shape
    ``(batch, max_candidates)``. Scores are sequence log-likelihoods. Rows with
    zero or one valid candidate have confidence one because there is no action
    uncertainty.
    """
    if candidate_log_scores.shape != candidate_mask.shape:
        raise ValueError("candidate scores and mask must have the same shape")

    valid = candidate_mask.bool()
    counts = valid.sum(dim=-1)
    masked_scores = candidate_log_scores.masked_fill(~valid, float("-inf"))
    safe_scores = torch.where(counts.unsqueeze(-1) > 0, masked_scores, torch.zeros_like(masked_scores))
    probabilities = torch.softmax(safe_scores, dim=-1) * valid
    log_probabilities = torch.where(
        probabilities > 0,
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
        torch.zeros_like(probabilities),
    )
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    max_entropy = counts.clamp_min(2).to(candidate_log_scores.dtype).log()
    confidence = 1 - entropy / max_entropy
    return torch.where(counts <= 1, torch.ones_like(confidence), confidence).clamp(0, 1)


def experience_weight(step: int, total_steps: int, start: float = 0.2, end: float = 0.5) -> float:
    """Linearly schedule the experience contribution from ``start`` to ``end``."""
    if total_steps <= 1:
        return end
    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + progress * (end - start)


def combine_reliability(
    experience_reliability: torch.Tensor,
    teacher_confidence: torch.Tensor,
    lambda_e: float,
) -> torch.Tensor:
    """Compute ``R = lambda_E E + (1-lambda_E) R_T``."""
    if not 0 <= lambda_e <= 1:
        raise ValueError("lambda_e must be in [0, 1]")
    if experience_reliability.shape != teacher_confidence.shape:
        raise ValueError("reliability tensors must have the same shape")
    return lambda_e * experience_reliability + (1 - lambda_e) * teacher_confidence


def compute_tool_opd_loss(
    student_log_probs: torch.Tensor,
    teacher_skill_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    turn_weights: torch.Tensor,
    max_log_ratio: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute sampled Teacher-to-Student KL OPD with outcome weights.

    The transfer gate from Section 2.4.2 is intentionally absent. Gradients
    flow only through ``student_log_probs``. For ``r = pi_teacher / pi_student``,
    ``r * log(r) - r + 1`` estimates ``KL(pi_teacher || pi_student)`` on
    Student-sampled action tokens.
    """
    if student_log_probs.shape != teacher_skill_log_probs.shape:
        raise ValueError("student and teacher log-prob tensors must have the same shape")
    if action_mask.shape != student_log_probs.shape:
        raise ValueError("action_mask must match log-prob shape")
    if turn_weights.ndim != 1 or turn_weights.shape[0] != student_log_probs.shape[0]:
        raise ValueError("turn_weights must have shape (batch,)")

    teacher = teacher_skill_log_probs.detach()
    weights = turn_weights.detach().to(student_log_probs.dtype).unsqueeze(-1)
    mask = action_mask.to(student_log_probs.dtype)
    log_ratio = (teacher - student_log_probs).clamp(min=-max_log_ratio, max=max_log_ratio)
    ratio = torch.exp(log_ratio)
    per_token_loss = ratio * log_ratio - ratio + 1
    denominator = mask.sum().clamp_min(1)
    loss = (per_token_loss * weights * mask).sum() / denominator

    with torch.no_grad():
        active_weight = weights * mask
        active_denominator = mask.sum().clamp_min(1)
        metrics = {
            "tool_opd/loss": loss.detach(),
            "tool_opd/weight_mean": active_weight.sum() / active_denominator,
            "tool_opd/log_ratio_mean": (log_ratio * mask).sum() / active_denominator,
            "tool_opd/action_token_fraction": mask.sum() / torch.tensor(mask.numel(), device=mask.device),
        }
    return loss, metrics


@dataclass
class ExperienceReplay:
    """In-memory Beta-Bernoulli statistics for ``p_succ(state, action)``."""

    alpha: float = 1.0
    _successes: dict[tuple[Hashable, str], int] = field(default_factory=lambda: defaultdict(int))
    _totals: dict[tuple[Hashable, str], int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def success_probability(self, state_signature: Hashable, action: str) -> float:
        key = (state_signature, action)
        return (self._successes[key] + self.alpha) / (self._totals[key] + 2 * self.alpha)

    def outcome_reliability(self, state_signature: Hashable, action: str, outcome: int) -> float:
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        probability = self.success_probability(state_signature, action)
        return probability if outcome == 1 else 1 - probability

    def update(self, records: Iterable[tuple[Hashable, str, int]]) -> None:
        for state_signature, action, outcome in records:
            if outcome not in (0, 1):
                raise ValueError("outcome must be binary")
            key = (state_signature, action)
            self._successes[key] += outcome
            self._totals[key] += 1
