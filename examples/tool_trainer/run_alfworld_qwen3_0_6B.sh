#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-$HOME/models/Qwen3-0.6B}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$HOME/checkpoints/verl_agent_alfworld/grpo_qwen3_4b/global_step_150/huggingface}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$PROJECT_ROOT/data/alfworld/train.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$PROJECT_ROOT/data/alfworld/test.parquet}

train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}

beta_h=${BETA_H:-5.0}
replay_alpha=${REPLAY_ALPHA:-1.0}
lambda_e_start=${LAMBDA_E_START:-0.2}
lambda_e_end=${LAMBDA_E_END:-0.5}
experiment_name=${EXPERIMENT_NAME:-grpo_opd_qwen3_0_6b_teacher4b_val128}
checkpoint_dir=${CHECKPOINT_DIR:-$REPO_ROOT/outputs/$experiment_name}
opd_loss_coef=${OPD_LOSS_COEF:-0.01}
max_log_ratio=${MAX_LOG_RATIO:-5.0}
save_freq=${SAVE_FREQ:-5}
max_confidence_candidates=${MAX_CONFIDENCE_CANDIDATES:-4}
confidence_score_batch_size=${CONFIDENCE_SCORE_BATCH_SIZE:-2048}
hindsight_response_length=${HINDSIGHT_RESPONSE_LENGTH:-96}

export ALFWORLD_DATA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO_ROOT/outputs/tensorboard/$experiment_name}

python3 -m verl.trainer.main_tool_opd \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +algorithm.tool_opd.teacher_model_path="$TEACHER_MODEL_PATH" \
    +algorithm.tool_opd.beta_h="$beta_h" \
    +algorithm.tool_opd.replay_alpha="$replay_alpha" \
    +algorithm.tool_opd.lambda_e_start="$lambda_e_start" \
    +algorithm.tool_opd.lambda_e_end="$lambda_e_end" \
    +algorithm.tool_opd.loss_coef="$opd_loss_coef" \
    +algorithm.tool_opd.max_log_ratio="$max_log_ratio" \
    +algorithm.tool_opd.max_confidence_candidates="$max_confidence_candidates" \
    +algorithm.tool_opd.confidence_score_batch_size="$confidence_score_batch_size" \
    +algorithm.tool_opd.hindsight_response_length="$hindsight_response_length" \
    +algorithm.tool_opd.teacher_max_prompt_length=4096 \
    +algorithm.tool_opd.hindsight_max_prompt_length=8192 \
    +algorithm.tool_opd.separate_teacher_pool=True \
    +algorithm.tool_opd.teacher_n_gpus_per_node=4 \
    +algorithm.tool_opd.teacher_nnodes=1 \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    trainer.critic_warmup=0 \
    trainer.logger='[console, tensorboard]' \
    trainer.project_name=tool_opd_alfworld \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq="$save_freq" \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$checkpoint_dir" \
    "${@:2}"
