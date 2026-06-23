"""PredatorRules: attack adjacent prey, hunt non-adjacent, wander, reproduce.

Decision priority (PREDATION_SPEC §6):
1. ATTACK: if any adjacent cell has occupant_species in preys_on
           -> pick prey cell (weakest prey first if sensing >= ENERGY,
              else deterministic sorted order)
2. MOVE toward sensed non-adjacent prey (hunt): if prey is visible but
   not adjacent -> MOVE one step toward it (only if target is unoccupied)
3. WANDER: MOVE to a random unoccupied adjacent cell in allowed_directions
           -> if no unoccupied adjacent cell -> IDLE
4. REPRODUCE: if energy >= reproduction_threshold AND an empty neighbor
   exists in allowed_directions -> REPRODUCE
"""
from __future__ import annotations

import numpy as np

from blobsim.rules import RulesEngine
from blobsim.types import Action, ActionType, CellView, Observation, SensingLevel


class PredatorRules(RulesEngine):
    """Predator blob decision logic with ATTACK > HUNT > WANDER > REPRODUCE."""

    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        """Return ATTACK, MOVE (hunt/wander), REPRODUCE, or IDLE."""
        attrs = observation.self_attributes
        status = observation.self_status

        # Priority 1: ATTACK adjacent prey
        if ActionType.ATTACK in attrs.allowed_actions:
            attack_action = self._try_attack(observation, rng)
            if attack_action is not None:
                return attack_action

        # Priority 2: MOVE toward non-adjacent sensed prey (hunt)
        hunt_action = self._try_hunt(observation, rng)
        if hunt_action is not None:
            return hunt_action

        # Priority 3: WANDER — move to random unoccupied adjacent cell
        wander_action = self._try_wander(observation, rng)
        if wander_action is not None:
            return wander_action

        # Priority 4: REPRODUCE if above threshold and empty neighbor exists
        if (
            ActionType.REPRODUCE in attrs.allowed_actions
            and status.energy >= attrs.reproduction_threshold
        ):
            has_empty = any(
                not cell.occupied and cell.offset in attrs.allowed_directions
                for cell in observation.neighborhood
            )
            if has_empty:
                return Action(type=ActionType.REPRODUCE)

        return Action(type=ActionType.IDLE)

    def _try_attack(
        self, observation: Observation, rng: np.random.Generator,
    ) -> Action | None:
        """Return ATTACK action if adjacent prey exists, else None.

        Prey selection: weakest prey first (if sensing >= ENERGY),
        otherwise deterministic sorted order by offset for INV-2.
        On exact energy tie, uses blob rng.
        """
        attrs = observation.self_attributes
        adjacent_prey: list[CellView] = [
            cell for cell in observation.neighborhood
            if cell.occupied
            and cell.offset in attrs.allowed_directions
            and cell.occupant_species in attrs.preys_on
        ]

        if not adjacent_prey:
            return None

        # Sort by (occupant_energy, offset) — weakest first, deterministic tie-break
        if attrs.sensing_level in (SensingLevel.ENERGY, SensingLevel.FULL):
            # occupant_energy is available; sort by energy then offset
            min_energy = min(
                c.occupant_energy for c in adjacent_prey
                if c.occupant_energy is not None
            )
            weakest = [
                c for c in adjacent_prey
                if c.occupant_energy is not None and c.occupant_energy == min_energy
            ]
            # Tiebreak: sort by offset for determinism; use rng if still tied
            weakest_sorted = sorted(weakest, key=lambda c: c.offset)
            chosen = weakest_sorted[0]
        else:
            # No energy info: pick deterministically by offset
            chosen = sorted(adjacent_prey, key=lambda c: c.offset)[0]

        return Action(type=ActionType.ATTACK, direction=chosen.offset)

    def _try_hunt(
        self, observation: Observation, rng: np.random.Generator,
    ) -> Action | None:
        """Return MOVE toward nearest non-adjacent sensed prey, else None.

        Only moves if the immediate step target is unoccupied (pure movement).
        If prey is adjacent, priority 1 (ATTACK) fires instead; this method
        only handles non-adjacent prey.
        """
        attrs = observation.self_attributes
        if ActionType.MOVE not in attrs.allowed_actions:
            return None

        # Find non-adjacent prey cells in observation
        non_adjacent_prey: list[CellView] = [
            cell for cell in observation.neighborhood
            if cell.occupied
            and cell.occupant_species in attrs.preys_on
            and cell.offset not in attrs.allowed_directions
        ]

        if not non_adjacent_prey:
            return None

        # Pick closest prey (smallest Chebyshev distance)
        target_cell = min(
            non_adjacent_prey,
            key=lambda c: (max(abs(c.offset[0]), abs(c.offset[1])), c.offset),
        )
        dr, dc = target_cell.offset

        # Step one cell toward target: clamp offsets to {-1, 0, 1}
        step = (
            max(-1, min(1, dr)),
            max(-1, min(1, dc)),
        )

        if step not in attrs.allowed_directions:
            return None

        # Only move if the immediate step cell is unoccupied
        step_cell = next(
            (c for c in observation.neighborhood if c.offset == step), None,
        )
        if step_cell is None or step_cell.occupied:
            return None

        return Action(type=ActionType.MOVE, direction=step)

    def _try_wander(
        self, observation: Observation, rng: np.random.Generator,
    ) -> Action | None:
        """Return MOVE to a random unoccupied adjacent cell, or None if blocked."""
        attrs = observation.self_attributes
        if ActionType.MOVE not in attrs.allowed_actions:
            return None

        unoccupied = [
            cell for cell in observation.neighborhood
            if not cell.occupied and cell.offset in attrs.allowed_directions
        ]

        if not unoccupied:
            return None

        chosen = unoccupied[rng.integers(len(unoccupied))]
        return Action(type=ActionType.MOVE, direction=chosen.offset)
