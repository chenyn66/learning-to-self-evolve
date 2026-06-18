#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

usage() {
  echo "Usage: bash scripts/setup_bird_data.sh [dev|train|all]"
}

download_and_extract() {
  local split="$1"
  local url=""
  local archive_name=""
  local inner_dir=""
  local dest_dir=""

  case "${split}" in
    dev)
      url="${BIRD_MINIDEV_URL:-https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip}"
      archive_name="minidev.zip"
      inner_dir="dev_databases"
      dest_dir="${LSE_BIRD_DATA_ROOT}/dev_data/dev_databases"
      ;;
    train)
      url="${BIRD_TRAIN_URL:-https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip}"
      archive_name="train.zip"
      inner_dir="train_databases"
      dest_dir="${LSE_BIRD_DATA_ROOT}/train_data/train_databases"
      ;;
    *)
      usage
      exit 1
      ;;
  esac

  local download_dir="${LSE_BIRD_DATA_ROOT}/.downloads"
  local archive_path="${download_dir}/${archive_name}"

  mkdir -p "${download_dir}"

  if [ ! -f "${archive_path}" ] || [ "${FORCE_DOWNLOAD:-0}" = "1" ]; then
    curl -L "${url}" -o "${archive_path}"
  else
    echo "Using existing archive: ${archive_path}"
  fi

  mkdir -p "${dest_dir}"

python3 - "${archive_path}" "${inner_dir}" "${dest_dir}" <<'PY'
from pathlib import Path, PurePosixPath
import io
import sys
import zipfile

archive_path = Path(sys.argv[1])
inner_dir = sys.argv[2]
dest_dir = Path(sys.argv[3])

def should_skip(path: PurePosixPath) -> bool:
    parts = path.parts
    return (
        not parts
        or parts[0] == "__MACOSX"
        or parts[-1] == ".DS_Store"
        or path.name.startswith("._")
    )

def extract_from_zip(zf: zipfile.ZipFile) -> int:
    extracted = 0
    for info in zf.infolist():
        path = PurePosixPath(info.filename)
        if should_skip(path):
            continue
        if inner_dir not in path.parts:
            continue
        idx = path.parts.index(inner_dir)
        rel_parts = path.parts[idx + 1 :]
        if not rel_parts:
            continue
        target = dest_dir.joinpath(*rel_parts)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            dst.write(src.read())
        extracted += 1
    return extracted

extracted = 0
with zipfile.ZipFile(archive_path) as zf:
    extracted = extract_from_zip(zf)
    if extracted == 0:
        nested_zip_name = None
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if should_skip(path):
                continue
            if path.name == f"{inner_dir}.zip":
                nested_zip_name = info.filename
                break
        if nested_zip_name is not None:
            with zipfile.ZipFile(io.BytesIO(zf.read(nested_zip_name))) as nested_zf:
                extracted = extract_from_zip(nested_zf)

if extracted == 0:
    raise SystemExit(
        f"Did not find '{inner_dir}/' entries inside {archive_path} "
        f"or any nested {inner_dir}.zip archive."
    )

print(f"Extracted {extracted} files into {dest_dir}")
PY
}

target="${1:-dev}"

case "${target}" in
  dev|train)
    download_and_extract "${target}"
    ;;
  all)
    download_and_extract dev
    download_and_extract train
    ;;
  *)
    usage
    exit 1
    ;;
esac
