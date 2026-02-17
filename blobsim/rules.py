"""Rules engine ABC for blob decision-making.

Spec reference: tech_spec_mvp.md §7 (Rules Engine).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from blobsim.types import Action, Observation


class RulesEngine(ABC):
    """Abstract base for blob decision logic.

    Instances persist across steps and are owned by a single blob.
    Implementations MAY maintain internal state (observation history,
    learned parameters, etc.).
    """

    @abstractmethod
    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        """Given what the blob sees, return one action.

        Contract:
        - Must return action from observation.self_attributes.allowed_actions
        - MOVE direction must be from observation.self_attributes.allowed_directions
        - REPRODUCE only if empty neighbor exists in allowed_directions
        - Violations crash the engine with RulesEngineError
        """
        ...
