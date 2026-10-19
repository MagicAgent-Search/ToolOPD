#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}

STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-$HOME/decision/docker-env/agent_data/open_model/Qwen_Qwen3-1.7B}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$HOME/checkpoints/search_grpo_qwen3_4b/actor_hf}
SEARCH_DATA_ROOT=${SEARCH_DATA_ROOT:-$HOME/data/search-data}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$SEARCH_DATA_ROOT/searchR1_processed_direct/train.parquet}
VAL_DATA_PATH=${VAL_DATA_PATH:-$SEARCH_DATA_ROOT/searchR1_processed_direct/test.parquet}
SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:8000/retrieve}

train_data_size=${TRAIN_DATA_SIZE:-128}
val_data_size=${VAL_DATA_SIZE:-512}
group_size=${GROUP_SIZE:-8}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}

beta_h=${BETA_H:-5.0}
replay_alpha=${REPLAY_ALPHA:-1.0}
lambda_e_start=${LAMBDA_E_START:-0.2}
lambda_e_end=${LAMBDA_E_END:-0.5}
opd_loss_coef=${OPD_LOSS_COEF:-0.01}
max_log_ratio=${MAX_LOG_RATIO:-5.0}
max_confidence_candidates=${MAX_CONFIDENCE_CANDIDATES:-4}
confidence_score_batch_size=${CONFIDENCE_SCORE_BATCH_SIZE:-2048}
hindsight_response_length=${HINDSIGHT_RESPONSE_LENGTH:-96}

experiment_name=${EXPERIMENT_NAME:-grpo_opd_searchqa_qwen3_1_7b_teacher4b}
checkpoint_dir=${CHECKPOINT_DIR:-$REPO_ROOT/outputs/$experiment_name}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-150}
total_training_steps=${TOTAL_TRAINING_STEPS:-150}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO_ROOT/outputs/tensorboard/$experiment_name}
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

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
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n="$group_size" \
    env.history_length=4 \
    env.resources_per_worker.num_cpus="$num_cpus_per_env_worker" \
    env.search.search_url="$SEARCH_URL" \
    trainer.critic_warmup=0 \
    trainer.logger='[console, tensorboard]' \
    trainer.project_name=tool_opd_searchqa \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq="$save_freq" \
    trainer.test_freq="$test_freq" \
    trainer.total_training_steps="$total_training_steps" \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$checkpoint_dir" \
    "${@:2}"
