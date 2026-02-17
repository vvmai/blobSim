"""RandomWalkerRules: random movement with opportunistic reproduction.

Decision priority (tech_spec_mvp.md §7):
1. If energy >= reproduction_threshold AND empty neighbor exists
   in allowed_directions -> REPRODUCE
2. Otherwise, pick uniformly at random from allowed_directions
   offsets (as MOVE) + IDLE.
   Total options = len(allowed_directions) + 1, equally weighted.
"""
from __future__ import annotations

import numpy as np

from blobsim.rules import RulesEngine
from blobsim.types import Action, ActionType, Observation


class RandomWalkerRules(RulesEngine):
    """Stateless random walker with opportunistic reproduction."""

    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        """Return REPRODUCE, random MOVE, or IDLE."""
        attrs = observation.self_attributes
        status = observation.self_status

        # Priority 1: Reproduce if above threshold and empty neighbor
        # in allowed_directions exists
        if status.energy >= attrs.reproduction_threshold:
            has_empty = any(
                not cell.occupied and cell.offset in attrs.allowed_directions
                for cell in observation.neighborhood
            )
            if has_empty:
                return Action(type=ActionType.REPRODUCE)

        # Priority 2: Uniform random choice from allowed_directions (MOVE) + IDLE
        # Sort directions for deterministic ordering given same RNG seed
        directions = sorted(attrs.allowed_directions)
        choice = rng.integers(len(directions) + 1)
        if choice == len(directions):
            return Action(type=ActionType.IDLE)
        return Action(type=ActionType.MOVE, direction=directions[choice])
