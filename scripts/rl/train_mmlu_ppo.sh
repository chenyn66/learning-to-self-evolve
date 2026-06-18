#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

runname="${1:-mmlu_lse_ppo}"
if [ "$#" -gt 0 ]; then
  shift
fi
extra_args=("$@")

VENV_DIR="${VERL_VENV_DIR:-${REPO_ROOT}/.venvs/verl}"
if [ ! -f "${VENV_DIR}/.install_ok" ]; then
  bash "${SCRIPT_DIR}/install_verl.sh"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

export LSE_RL_TRAIN_FILES="${LSE_RL_TRAIN_FILES:-${REPO_ROOT}/logs/gpqa-subfield-data/mmlu}"
export LSE_RL_VAL_FILES="${LSE_RL_VAL_FILES:-${REPO_ROOT}/logs/vl4b-test/mmlu}"

if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPU="$(nvidia-smi --query-gpu=count --format=csv,noheader | awk 'NR==1 {gsub(/ /, "", $0); print; exit}')"
else
  N_GPU="${N_GPU:-1}"
fi

TRAIN_N_GPU="${TRAIN_N_GPU:-${N_GPU}}"
REWARD_N_GPU="${REWARD_N_GPU:-${N_GPU}}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_DP="${ROLLOUT_DP:-${TRAIN_N_GPU}}"
REWARD_TP="${REWARD_TP:-1}"
REWARD_DP="${REWARD_DP:-${REWARD_N_GPU}}"
ROLLOUT_NUM_WORKERS="${ROLLOUT_NUM_WORKERS:-${N_GPU}}"

ACTOR_ENGINE="${ACTOR_ENGINE:-vllm}"
REWARD_ENGINE="${REWARD_ENGINE:-vllm}"
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
REWARD_MODEL="${REWARD_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"

TRAINER_LOGGER='["console"]'
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-offline}" != "disabled" ]; then
  wandb login "${WANDB_API_KEY}"
  TRAINER_LOGGER='["console","wandb"]'
fi

logdir="${LSE_LOG_DIR}/rl/${runname}"
mkdir -p "${logdir}"

python -m verl.trainer.main_lse_ppo \
  data=lse_data \
  data.train_files="${LSE_RL_TRAIN_FILES}" \
  data.val_files="${LSE_RL_VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-100000}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-16384}" \
  data.return_raw_chat=True \
  algorithm.adv_estimator="${ADV_ESTIMATOR:-delta}" \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.model.path="${ACTOR_MODEL}" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-1}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS:-1}" \
  actor_rollout_ref.rollout.name="${ACTOR_ENGINE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ACTOR_GPU_UTIL:-0.8}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_SAMPLES:-8}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  critic.enable=false \
  reward_model.enable="${REWARD_ENABLE:-true}" \
  reward_model.use_reward_loop="${REWARD_USE_LOOP:-true}" \
  reward_model.enable_resource_pool="${REWARD_ENABLE_RESOURCE_POOL:-false}" \
  reward_model.n_gpus_per_node="${REWARD_N_GPU}" \
  reward_model.nnodes=1 \
  reward_model.model.path="${REWARD_MODEL}" \
  reward_model.rollout.name="${REWARD_ENGINE}" \
  reward_model.rollout.tensor_model_parallel_size="${REWARD_TP}" \
  reward_model.rollout.data_parallel_size="${REWARD_DP}" \
  reward_model.rollout.gpu_memory_utilization="${REWARD_GPU_UTIL:-0.8}" \
  reward_model.rollout.prompt_length="${REWARD_PROMPT_LENGTH:-16384}" \
  reward_model.rollout.response_length="${REWARD_RESPONSE_LENGTH:-16384}" \
  reward_model.launch_reward_fn_async=false \
  reward_model.reward_manager="${REWARD_MANAGER_NAME:-naive}" \
  reward_manager.name="${REWARD_MANAGER_NAME:-naive}" \
  custom_reward_function.path="${CUSTOM_REWARD_PATH:-pkg://verl.utils.lse_reward_fn}" \
  trainer.project_name="${RL_PROJECT_NAME:-mmlu_lse_rl}" \
  trainer.experiment_name="${runname}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.n_gpus_per_node="${TRAIN_N_GPU}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.val_before_train=false \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  "${extra_args[@]}" 2>&1 | tee "${logdir}/${runname}.log"
