"""OmnivoreRules unit tests.

Tests call decide() directly with synthetic Observations covering the four
priority branches: ATTACK > REPRODUCE > GRAZE > IDLE.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from blobsim.blob import BlobAttributes, BlobStatus
from blobsim.species.omnivore import OmnivoreRules
from blobsim.types import (
    ActionType,
    CellView,
    MOORE,
    Observation,
    SensingLevel,
)

from conftest import make_cell_view


# ---------------------------------------------------------------------------
# Observation factory
# ---------------------------------------------------------------------------

def make_omnivore_observation(
    *,
    energy: float,
    preys_on: frozenset,
    neighborhood: tuple[CellView, ...],
    reproduction_threshold: float = float("inf"),
    allowed_actions: frozenset[ActionType] | None = None,
    allowed_directions: frozenset[tuple[int, int]] | None = None,
) -> Observation:
    """Build an Observation for an omnivore with the given parameters."""
    if allowed_actions is None:
        allowed_actions = frozenset({
            ActionType.IDLE, ActionType.MOVE,
            ActionType.ATTACK, ActionType.REPRODUCE,
        })
    if allowed_directions is None:
        allowed_directions = MOORE

    attrs = BlobAttributes.create(
        species="omnivore",
        max_energy=20.0,
        offspring_energy=2.0,
        reproduction_threshold=reproduction_threshold,
        base_metabolic_cost=0.2,
        move_cost=0.1,
        reproduce_cost=0.0,
        attack_cost=0.1,
        observation_radius=1,
        sensing_level=SensingLevel.ENERGY,
        allowed_actions=allowed_actions,
        allowed_directions=allowed_directions,
        preys_on=preys_on,
    )
    status = BlobStatus(energy=energy, age=0, position=(5, 5))
    current_cell = CellView(
        offset=(0, 0),
        layers=MappingProxyType({"energy": 0.0}),
        occupied=True,
    )
    return Observation(
        self_status=status,
        self_attributes=attrs,
        neighborhood=neighborhood,
        current_cell=current_cell,
        step=0,
    )


# ---------------------------------------------------------------------------
# O1: Attacks the weakest adjacent prey when stronger
# ---------------------------------------------------------------------------

def test_omnivore_attacks_weakest_beatable_prey() -> None:
    """Two beatable adjacent prey (E=3 and E=7); omnivore (E=10) attacks E=3."""
    weak_prey = CellView(
        offset=(0, 1),
        layers=MappingProxyType({"energy": 0.0}),
        occupied=True,
        occupant_species="prey",
        occupant_energy=3.0,
    )
    strong_prey = CellView(
        offset=(1, 0),
        layers=MappingProxyType({"energy": 0.0}),
        occupied=True,
        occupant_species="prey",
        occupant_energy=7.0,
    )
    obs = make_omnivore_observation(
        energy=10.0,
        preys_on=frozenset({"prey"}),
        neighborhood=(weak_prey, strong_prey),
    )

    rules = OmnivoreRules()
    rng = np.random.default_rng(42)
    action = rules.decide(obs, rng)

    assert action.type == ActionType.ATTACK
    assert action.direction == (0, 1)  # weakest prey at offset (0,1)


# ---------------------------------------------------------------------------
# O2: Does NOT attack when weaker-or-equal (falls through to graze)
# ---------------------------------------------------------------------------

def test_omnivore_does_not_attack_when_weaker_or_equal() -> None:
    """Omnivore (E=5) vs adjacent prey (E=5 and E=8) — none beatable — grazes."""
    equal_prey = CellView(
        offset=(0, 1),
        layers=MappingProxyType({"energy": 0.0}),
        occupied=True,
        occupant_species="prey",
        occupant_energy=5.0,  # equal energy — not beatable
    )
    stronger_prey = CellView(
        offset=(1, 0),
        layers=MappingProxyType({"energy": 0.0}),
        occupied=True,
        occupant_species="prey",
        occupant_energy=8.0,
    )
    grass_cell = CellView(
        offset=(-1, 0),
        layers=MappingProxyType({"energy": 3.0}),
        occupied=False,
    )
    obs = make_omnivore_observation(
        energy=5.0,
        preys_on=frozenset({"prey"}),
        neighborhood=(equal_prey, stronger_prey, grass_cell),
    )

    rules = OmnivoreRules()
    rng = np.random.default_rng(42)
    action = rules.decide(obs, rng)

    assert action.type != ActionType.ATTACK
    # Falls through to graze since grass is available
    assert action.type == ActionType.MOVE
    assert action.direction == (-1, 0)


# ---------------------------------------------------------------------------
# O3: Grazes toward the richest grass when no beatable prey
# ---------------------------------------------------------------------------

def test_omnivore_grazes_toward_richest_grass() -> None:
    """No beatable prey; two grass cells (E=1 and E=4) — moves to E=4."""
    poor_grass = make_cell_view(offset=(0, 1), energy=1.0, occupied=False)
    rich_grass = make_cell_view(offset=(1, 0), energy=4.0, occupied=False)
    obs = make_omnivore_observation(
        energy=5.0,
        preys_on=frozenset({"prey"}),
        neighborhood=(poor_grass, rich_grass),
    )

    rules = OmnivoreRules()
    rng = np.random.default_rng(42)
    action = rules.decide(obs, rng)

    assert action.type == ActionType.MOVE
    assert action.direction == (1, 0)  # richest grass at offset (1,0)


# ---------------------------------------------------------------------------
# O4: Reproduces when fat with an empty neighbour and no adjacent beatable prey
# ---------------------------------------------------------------------------

def test_omnivore_reproduces_when_fat_no_beatable_prey() -> None:
    """Energy=15 >= threshold=10; one empty neighbour; no beatable prey -> REPRODUCE."""
    # Adjacent prey is stronger — not beatable
    strong_enemy = CellView(
        offset=(0, 1),
        layers=MappingProxyType({"energy": 0.0}),
        occupied=True,
        occupant_species="prey",
        occupant_energy=20.0,
    )
    # Empty neighbour with no grass energy — triggers REPRODUCE, not GRAZE
    empty_cell = make_cell_view(offset=(1, 0), energy=0.0, occupied=False)
    obs = make_omnivore_observation(
        energy=15.0,
        preys_on=frozenset({"prey"}),
        neighborhood=(strong_enemy, empty_cell),
        reproduction_threshold=10.0,
    )

    rules = OmnivoreRules()
    rng = np.random.default_rng(42)
    action = rules.decide(obs, rng)

    assert action.type == ActionType.REPRODUCE


# ---------------------------------------------------------------------------
# O5: Idles when boxed in with no grass
# ---------------------------------------------------------------------------

def test_omnivore_idles_when_boxed_in_no_grass() -> None:
    """All neighbours occupied and no grass energy — omnivore idles."""
    all_occupied = tuple(
        CellView(
            offset=d,
            layers=MappingProxyType({"energy": 0.0}),
            occupied=True,
            occupant_species="other",  # not in preys_on
        )
        for d in sorted(MOORE)
    )
    obs = make_omnivore_observation(
        energy=5.0,
        preys_on=frozenset({"prey"}),
        neighborhood=all_occupied,
    )

    rules = OmnivoreRules()
    rng = np.random.default_rng(42)
    action = rules.decide(obs, rng)

    assert action.type == ActionType.IDLE
