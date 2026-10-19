#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-Qwen/Qwen3-1.7B}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$HOME/checkpoints/verl_agent_alfworld/grpo_qwen3_4b/global_step_150/huggingface}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$PROJECT_ROOT/data/alfworld/train.parquet}
export VAL_DATA_PATH=${VAL_DATA_PATH:-$PROJECT_ROOT/data/alfworld/test.parquet}

export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
export GROUP_SIZE=${GROUP_SIZE:-8}
export OPD_LOSS_COEF=${OPD_LOSS_COEF:-0.01}
export MAX_LOG_RATIO=${MAX_LOG_RATIO:-5.0}
export SAVE_FREQ=${SAVE_FREQ:-5}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-tool_opd_alfworld_qwen3_1_7b_teacher4b_opd001}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJECT_ROOT/outputs/$EXPERIMENT_NAME}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$PROJECT_ROOT/outputs/tensorboard/$EXPERIMENT_NAME}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

exec bash "$SCRIPT_DIR/run_alfworld_qwen3_0_6B.sh" "${@:-vllm}"
