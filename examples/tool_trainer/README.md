# Tool-OPD training

The Python entry point is
`verl.trainer.main_tool_opd`. Teacher and Student must currently share the same
tokenizer so sampled action-token probabilities align exactly.

```bash
export STUDENT_MODEL_PATH=/path/to/student
export TEACHER_MODEL_PATH=/path/to/rl-teacher
bash examples/tool_trainer/run_alfworld_qwen3_0_6B.sh vllm
```

| Environment | Qwen3-0.6B | Qwen3-1.7B |
| --- | --- | --- |
| ALFWorld | `run_alfworld_qwen3_0_6B.sh` | `run_alfworld_qwen3_1_7B.sh` |
| SearchQA | `run_searchqa_qwen3_0_6B.sh` | `run_searchqa_qwen3_1_7B.sh` |
| WebShop | `run_webshop_qwen3_0_6B.sh` | `run_webshop_qwen3_1_7B.sh` |

Tool-OPD uses only tokens inside `<action>...</action>`. Its turn weight is
`H * R`, where `R` combines replay-calibrated outcome consistency with Teacher
confidence over the admissible-action set. Section 2.4.2's Student transfer
gate `M` is intentionally not implemented.

The unified objective is Student-on-policy Teacher-to-Student KL:
`E[H * R * KL(pi_teacher^skill || pi_student)]`. With
`r = exp(teacher_skill_log_prob - student_log_prob)`, the implementation uses
the sampled estimator `r * log(r) - r + 1` on action tokens. It does not use a
PPO ratio, PPO clipping, GRPO/task-reward advantages, or an additional policy
loss.
