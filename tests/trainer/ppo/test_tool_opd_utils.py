import pytest
import torch

from verl.trainer.ppo.tool_opd_utils import (
    ExperienceReplay,
    combine_reliability,
    compute_outcome_asymmetric_credit,
    compute_teacher_confidence,
    compute_tool_opd_loss,
    experience_weight,
    select_confidence_candidates,
)


def test_select_confidence_candidates_keeps_student_action():
    candidates = ["go to table", "open drawer", "take apple", "look"]

    assert select_confidence_candidates(candidates, "take apple", 2) == [
        "take apple",
        "go to table",
    ]
    assert select_confidence_candidates(candidates, "missing", 2) == candidates[:2]
    assert select_confidence_candidates(candidates, "take apple", 8) == candidates




def test_outcome_asymmetric_credit_selects_supported_success_and_rejected_failure():
    base = torch.zeros(2, 3)
    skill = torch.tensor([[1.0, -1.0, 4.0], [1.0, -1.0, 4.0]])
    outcomes = torch.tensor([1, 0])
    action_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    token_credit, turn_credit = compute_outcome_asymmetric_credit(
        base, skill, outcomes, action_mask, beta_h=2.0
    )

    assert token_credit[0, 0] > 0
    assert token_credit[0, 1] == 0
    assert token_credit[1, 0] == 0
    assert token_credit[1, 1] > 0
    assert torch.allclose(turn_credit[0], turn_credit[1])
    assert torch.all(token_credit[:, 2] == 0)


def test_teacher_confidence_is_normalized_and_handles_single_candidate():
    scores = torch.tensor([[0.0, 0.0], [10.0, -10.0], [3.0, 0.0]])
    mask = torch.tensor([[1, 1], [1, 1], [1, 0]])

    confidence = compute_teacher_confidence(scores, mask)

    assert confidence[0].item() == pytest.approx(0.0, abs=1e-6)
    assert confidence[1].item() == pytest.approx(1.0, abs=1e-6)
    assert confidence[2].item() == pytest.approx(1.0, abs=1e-6)


def test_experience_replay_uses_prior_then_observed_outcomes():
    replay = ExperienceReplay(alpha=1.0)
    assert replay.success_probability("state", "open door") == pytest.approx(0.5)

    replay.update([("state", "open door", 1), ("state", "open door", 1)])

    assert replay.success_probability("state", "open door") == pytest.approx(0.75)
    assert replay.outcome_reliability("state", "open door", 1) == pytest.approx(0.75)
    assert replay.outcome_reliability("state", "open door", 0) == pytest.approx(0.25)


def test_reliability_schedule_and_combination():
    assert experience_weight(0, 11) == pytest.approx(0.2)
    assert experience_weight(10, 11) == pytest.approx(0.5)
    combined = combine_reliability(torch.tensor([0.8]), torch.tensor([0.2]), lambda_e=0.25)
    assert combined.item() == pytest.approx(0.35)


def test_tool_opd_loss_masks_think_tokens_and_has_student_gradient():
    student = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[8.0, 1.0, -1.0]])
    action_mask = torch.tensor([[0, 1, 1]])
    weights = torch.tensor([0.5])

    loss, metrics = compute_tool_opd_loss(student, teacher, action_mask, weights)
    loss.backward()

    assert loss.item() >= 0
    assert student.grad[0, 0] == 0
    assert student.grad[0, 1] != 0
    assert student.grad[0, 2] != 0
    assert metrics["tool_opd/action_token_fraction"].item() == pytest.approx(2 / 3)


def test_tool_opd_loss_is_zero_when_teacher_matches_student():
    student = torch.tensor([[-0.2, -1.0]], requires_grad=True)
    teacher = student.detach().clone()
    loss, _ = compute_tool_opd_loss(student, teacher, torch.ones_like(student), torch.ones(1))
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_tool_opd_loss_matches_sampled_teacher_to_student_kl():
    student = torch.zeros(1, 1, requires_grad=True)
    teacher = torch.log(torch.tensor([[2.0]]))
    loss, _ = compute_tool_opd_loss(
        student, teacher, torch.ones_like(student), torch.tensor([0.5])
    )

    # 0.5 * (2 * log(2) - 2 + 1)
