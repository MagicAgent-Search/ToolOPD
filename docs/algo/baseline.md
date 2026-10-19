# Algorithm Baselines

## Datasets

Assuming GSM8k/math dataset is preprocessed via:

```bash
python3 examples/data_preprocess/*.py
```

Refer to the table below to reproduce RL training from different pre-trained checkpoints. Below is the performance on the GSM8k dataset if not specified otherwise.


| Hardware    | Model                            | Method            | Test score   | Details |
|-------------|----------------------------------|-------------------|--------------|---------|
| NVIDIA GPU  | google/gemma-2-2b-it             | hf checkpoint     | 23.9         | [Huggingface](https://huggingface.co/google/gemma-2-2b-it#benchmark-results) |
| NVIDIA GPU  | google/gemma-2-2b-it             | SFT               | 52.06        |  |
| NVIDIA GPU  | google/gemma-2-2b-it             | SFT + PPO         | 64.02        |  |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | hf checkpoint     | 36.4         | [Qwen blog](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | PPO               | 56.7         |  |
| NVIDIA GPU  | Qwen/Qwen2.5-0.5B-Instruct       | PRIME             | 58.7         | [script](https://github.com/volcengine/verl/blob/main/recipe/prime/run_prime_qwen.sh) |
| NVIDIA GPU  | deepseek-ai/deepseek-llm-7b-chat | PPO (Megatron)    | 69.5 [1]     |  |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GRPO              | 89           | [script](https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/examples/grpo_trainer/run_qwen2-7b_seq_balance.sh) |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GRPO (FSDP2)      | 89.8         |  |
| NVIDIA GPU  | Qwen/Qwen2-7B-Instruct           | GRPO (Megatron)   | 89.6         |  |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | ReMax             | 97           | [script](https://github.com/eric-haibin-lin/verl/blob/main/examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh) |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | SPPO              | 65.6 (MATH)  | [SPPO script](https://github.com/volcengine/verl/tree/main/recipe/sppo/README.md) |
| NVIDIA GPU  | Mixtral-8x22B-Instruct-v0.1      | Instruct model    | 83.7         | [Qwen Blog](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| NVIDIA GPU  | Mixtral-8x22B-Instruct-v0.1      | RLOO (Megatron)   | 92.3         | |
| NVIDIA GPU  | Qwen/Qwen2.5-7B-Instruct         | SPIN              | 92           | [script](https://github.com/volcengine/verl/tree/main/recipe/spin/README.md) |
| AMD MI300   | deepseek-ai/deepseek-llm-7b-chat | PPO               | 70.5 [1]     |  |
| AMD MI300   | deepseek-ai/deepseek-llm-7b-chat | GRPO              | 71.4 [1]     |  |

### Notes

[1] During evaluation, we have only extracted answers following the format `"####"`. A more flexible answer extraction, longer response length, and better prompt engineering may lead to a higher score.

[2] The default value of `actor_rollout_ref.actor.entropy_coeff` is set to `0.0` since verl 0.3.x on 2025-05-30, which is different from previous versions.