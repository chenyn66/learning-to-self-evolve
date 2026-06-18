"""Stop conditions for running GEPA inside LSE.

GEPA (vendored under ./gepa/) primarily supports stopping by *metric call budget*
(#examples evaluated). For our baselines, we often want to stop after a fixed
number of GEPA *iterations* to match LSE's `n_round`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaxIterationsStopper:
    """Stop after `max_iters` GEPA iterations.

    GEPA initializes `state.i = -1` and increments it at the start of each
    iteration (see `gepa/src/gepa/core/engine.py`). After the first iteration
    begins, `state.i == 0`. Therefore, the number of iterations executed so far
    is `state.i + 1`.
    """

    max_iters: int

    def __call__(self, gepa_state) -> bool:  # type: ignore[override]
        # Defensive: treat non-positive max_iters as "stop immediately".
        if self.max_iters <= 0:
            return True
        # Stop once we've *completed* max_iters iterations.
        return (int(getattr(gepa_state, "i", -1)) + 1) >= self.max_iters


__all__ = ["MaxIterationsStopper"]

