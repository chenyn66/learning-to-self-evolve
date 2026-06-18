#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

VENV_DIR="${VERL_VENV_DIR:-${REPO_ROOT}/.venvs/verl}"
PYTHON_VERSION="${VERL_PYTHON_VERSION:-3.12.12}"
LOCKFILE="${SCRIPT_DIR}/requirements.lock.txt"

uv venv "${VENV_DIR}" --python="${PYTHON_VERSION}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_ROOT}"
unset UV_TORCH_BACKEND || true
uv pip sync "${LOCKFILE}"
uv pip install --no-deps -e "${REPO_ROOT}/verl"
uv pip install --no-deps -e "${REPO_ROOT}"

python - <<'PY'
try:
    from opencv_fixer import AutoFix
    AutoFix()
except Exception as exc:
    print(f"[install_verl] opencv-fixer skipped: {exc}")
PY

touch "${VENV_DIR}/.install_ok"
echo "VERL environment is ready at ${VENV_DIR}"
