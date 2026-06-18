#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

VENV_DIR="${LSE_VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_VERSION="${LSE_PYTHON_VERSION:-3.10}"

uv venv "${VENV_DIR}" --python="${PYTHON_VERSION}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${REPO_ROOT}"
uv pip install -r requirements.txt
uv pip install -e .

if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
  uv pip install flash-attn --no-build-isolation
fi

echo "Core environment is ready at ${VENV_DIR}"
