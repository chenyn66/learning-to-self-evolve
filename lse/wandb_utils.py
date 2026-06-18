from __future__ import annotations

import os
from typing import Any

import wandb


def wandb_mode(debug: bool) -> str:
    if debug:
        return "disabled"
    return os.environ.get("WANDB_MODE", "offline")


def login_if_configured(wandb_key: Any) -> None:
    key = ("" if wandb_key is None else str(wandb_key)).strip()
    env_key = (os.environ.get("WANDB_API_KEY") or "").strip()

    if key:
        wandb.login(key=key)
        return

    if env_key:
        wandb.login(key=env_key)
