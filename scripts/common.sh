#!/usr/bin/env bash
set -euo pipefail

COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_SH_DIR}/.." && pwd)"
export REPO_ROOT

if [ -f "${REPO_ROOT}/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.local"
  set +a
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export LSE_LOG_DIR="${LSE_LOG_DIR:-${REPO_ROOT}/logs}"
export LSE_RL_OUTPUT_DIR="${LSE_RL_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/rl}"

export LSE_MMLU_DATA_DIR="${LSE_MMLU_DATA_DIR:-${REPO_ROOT}/data}"
export LSE_BIRD_DATA_ROOT="${LSE_BIRD_DATA_ROOT:-${REPO_ROOT}/data/bird}"

mkdir -p \
  "${HF_HOME}" \
  "${LSE_LOG_DIR}" \
  "${LSE_RL_OUTPUT_DIR}"

maybe_activate_core_venv() {
  local venv_dir="${LSE_VENV_DIR:-${REPO_ROOT}/.venv}"
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${venv_dir}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
  fi
}
