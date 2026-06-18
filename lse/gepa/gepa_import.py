"""Vendored GEPA import helper.

This repo vendors the GEPA codebase under `./gepa/`. The actual Python package
sources live at `gepa/src/gepa/`. Since we don't require a `pip install` of the
vendored package, we add that `src` directory to `sys.path` at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_gepa_on_path() -> None:
    """Ensure `import gepa` resolves to the vendored package.

    Safe to call multiple times.
    """
    # Allow external override.
    env_src = os.environ.get("GEPA_VENDOR_DIR")
    if env_src:
        gepa_src = Path(env_src).expanduser()
    else:
        # Search upwards from this file for repo root containing ./gepa/src.
        here = Path(__file__).resolve()
        gepa_src = None
        for p in [here] + list(here.parents):
            cand = p / "gepa" / "src"
            if cand.exists():
                gepa_src = cand
                break
        if gepa_src is None:
            raise FileNotFoundError(
                "Vendored GEPA not found. Expected repo layout with ./gepa/src/gepa/. "
                f"Starting from {here}."
            )

    if not gepa_src.exists():
        raise FileNotFoundError(
            f"Vendored GEPA not found at {gepa_src}. Expected repo layout with ./gepa/src/gepa/."
        )

    gepa_src_str = str(gepa_src)
    if gepa_src_str not in sys.path:
        # Prepend to prefer vendored GEPA over any installed version.
        sys.path.insert(0, gepa_src_str)

    os.environ.setdefault("GEPA_VENDOR_DIR", str(gepa_src))


__all__ = ["ensure_gepa_on_path"]

