from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEnv(ABC):
    """
    Abstract task environment interface.

    Concrete tasks may be sequential (step-based) or single-shot. This base
    interface focuses on reset and summary reporting for compatibility with
    self-evolving agents and simulators.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment for a new run/episode/batch."""
        raise NotImplementedError

    @abstractmethod
    def get_summary(self) -> List[Dict[str, Any]]:
        """
        Return a list of per-episode summaries as dictionaries. Each summary
        should capture the information needed for an agent to self-evolve
        (e.g., actions/rewards for bandits, predictions/labels for datasets).
        """
        raise NotImplementedError


__all__ = ["BaseEnv"]


