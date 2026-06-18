from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return REPO_ROOT


def path_from_env(env_name: str, default_relative_path: str) -> Path:
    raw = (os.environ.get(env_name) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (REPO_ROOT / default_relative_path).resolve()


def bird_data_root() -> Path:
    return path_from_env("LSE_BIRD_DATA_ROOT", "data/bird")


def bird_split_dir(split: str) -> Path:
    return bird_data_root() / f"{split}_data"


def bird_prompt_path(split: str) -> Path:
    return bird_split_dir(split) / f"{split}_prompt.json"


def bird_schema_path(split: str) -> Path:
    return bird_split_dir(split) / "db2schema.json"


def bird_ground_truth_cache_path() -> Path:
    return bird_data_root() / "ground_truth_cache.pkl"


def bird_json_path(split: str) -> Path:
    return bird_split_dir(split) / f"{split}.json"


def bird_db_root(split: str) -> Path:
    env_name = "BIRD_DB_ROOT_DEV" if split == "dev" else "BIRD_DB_ROOT_TRAIN"
    raw = (os.environ.get(env_name) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return bird_split_dir(split) / f"{split}_databases"


def mmlu_data_root() -> Path:
    return path_from_env("LSE_MMLU_DATA_DIR", "data")


def mmlu_dataset_path(dataset_name: str) -> Path:
    return mmlu_data_root() / dataset_name
