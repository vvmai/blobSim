"""OmnivoreRules: a grazer with an opportunistic predation rule.

An omnivore forages grass to sustain itself and preys opportunistically on
adjacent enemies — but only when it is stronger than its target, so it never
wastes attack_cost on a losing fight.

Decision priority each step:
  1. ATTACK the weakest beatable adjacent enemy in preys_on
     (only when my_energy > occupant_energy).
  2. REPRODUCE if energy >= threshold and an empty Moore neighbor exists.
  3. MOVE toward the richest adjacent unoccupied grass cell (graze).
  4. IDLE.
"""
from __future__ import annotations

import numpy as np

from blobsim.rules import RulesEngine
from blobsim.types import Action, ActionType, CellView, Observation


def _is_moore(cell: CellView) -> bool:
    """Return True if cell is within the Moore-1 neighbourhood (|dr|<=1, |dc|<=1)."""
    return abs(cell.offset[0]) <= 1 and abs(cell.offset[1]) <= 1


class OmnivoreRules(RulesEngine):
    """Omnivore blob decision logic with ATTACK > REPRODUCE > GRAZE > IDLE."""

    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        """Return the highest-priority available action for this step."""
        attrs = observation.self_attributes
        my_energy = observation.self_status.energy
        adj = attrs.allowed_directions

        # Priority 1: opportunistic ATTACK — only if I am strictly stronger.
        beatable = [
            c for c in observation.neighborhood
            if c.occupied
            and c.offset in adj
            and _is_moore(c)
            and c.occupant_species in attrs.preys_on
            and c.occupant_energy is not None
            and my_energy > c.occupant_energy
        ]
        if beatable:
            target = min(beatable, key=lambda c: c.occupant_energy or 0.0)
            return Action(type=ActionType.ATTACK, direction=target.offset)

        # Priority 2: REPRODUCE if fat and an empty neighbour is available.
        if (
            ActionType.REPRODUCE in attrs.allowed_actions
            and my_energy >= attrs.reproduction_threshold
        ):
            has_empty = any(
                not c.occupied and c.offset in adj and _is_moore(c)
                for c in observation.neighborhood
            )
            if has_empty:
                return Action(type=ActionType.REPRODUCE)

        # Priority 3: GRAZE — move to the richest adjacent unoccupied grass cell.
        graze_candidates = [
            c for c in observation.neighborhood
            if not c.occupied
            and c.offset in adj
            and _is_moore(c)
            and c.energy > 0
        ]
        if graze_candidates:
            best = max(graze_candidates, key=lambda c: c.energy)
            return Action(type=ActionType.MOVE, direction=best.offset)

        return Action(type=ActionType.IDLE)
