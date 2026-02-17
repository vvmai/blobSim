"""GrazerRules: move toward highest-energy cell, reproduce when able.

Decision priority (tech_spec_mvp.md §7):
1. If energy >= reproduction_threshold AND empty neighbor exists
   in allowed_directions -> REPRODUCE
2. Find unoccupied cell in neighborhood with highest cell.energy.
   If tie, break uniformly at random using rng.
3. If best cell has energy > 0 AND offset is in allowed_directions
   -> MOVE toward it
4. Otherwise -> IDLE
"""
from __future__ import annotations

import numpy as np

from blobsim.rules import RulesEngine
from blobsim.types import Action, ActionType, Observation


class GrazerRules(RulesEngine):
    """Energy-seeking grazer with opportunistic reproduction."""

    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        """Return REPRODUCE, MOVE toward richest cell, or IDLE."""
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

        # Priority 2: Find best unoccupied cell in allowed_directions
        # with positive energy
        candidates = [
            cell for cell in observation.neighborhood
            if not cell.occupied
            and cell.offset in attrs.allowed_directions
            and cell.energy > 0
        ]

        if not candidates:
            return Action(type=ActionType.IDLE)

        # Find max energy among candidates
        max_energy = max(c.energy for c in candidates)
        best = [c for c in candidates if c.energy == max_energy]

        # Tiebreak: uniform random
        if len(best) == 1:
            chosen = best[0]
        else:
            chosen = best[rng.integers(len(best))]

        return Action(type=ActionType.MOVE, direction=chosen.offset)
