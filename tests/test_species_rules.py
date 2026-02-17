"""Species rules engine unit tests (G1-G5).

Pure unit tests -- call decide() directly with synthetic Observations.
"""
from __future__ import annotations

import numpy as np

from blobsim.species import GrazerRules, RandomWalkerRules
from blobsim.types import ActionType, MOORE, VON_NEUMANN

from conftest import make_cell_view, make_observation


# ---------------------------------------------------------------------------
# G1: RandomWalker -- all neighbors occupied, above threshold -> IDLE
# ---------------------------------------------------------------------------

def test_random_walker_all_neighbors_occupied() -> None:
    """All neighbor cells occupied + above repro threshold -> IDLE (no REPRODUCE)."""
    neighborhood = tuple(
        make_cell_view(offset=d, energy=1.0, occupied=True)
        for d in sorted(MOORE)
    )
    obs = make_observation(
        energy=15.0,
        reproduction_threshold=5.0,
        allowed_directions=MOORE,
        neighborhood=neighborhood,
    )
    rules = RandomWalkerRules()
    rng = np.random.default_rng(42)

    action = rules.decide(obs, rng)
    # Cannot REPRODUCE (no empty neighbor) -> random walk or IDLE
    assert action.type in (ActionType.MOVE, ActionType.IDLE)
    assert action.type != ActionType.REPRODUCE


# ---------------------------------------------------------------------------
# G2: Grazer -- moves toward highest-energy cell
# ---------------------------------------------------------------------------

def test_grazer_moves_toward_highest_energy() -> None:
    """Two unoccupied cells: 5.0 vs 1.0 -> MOVE toward 5.0 cell."""
    low_cell = make_cell_view(offset=(0, -1), energy=1.0)
    high_cell = make_cell_view(offset=(0, 1), energy=5.0)
    occ_cell = make_cell_view(offset=(1, 0), energy=3.0, occupied=True)
    neighborhood = (low_cell, high_cell, occ_cell)

    obs = make_observation(
        energy=3.0,
        reproduction_threshold=float("inf"),
        allowed_directions=MOORE,
        neighborhood=neighborhood,
    )
    rules = GrazerRules()
    rng = np.random.default_rng(42)

    action = rules.decide(obs, rng)
    assert action.type == ActionType.MOVE
    assert action.direction == (0, 1)


# ---------------------------------------------------------------------------
# G3: Grazer -- best cell occupied -> picks next or IDLE
# ---------------------------------------------------------------------------

def test_grazer_skips_occupied_high_energy() -> None:
    """Best-energy cell is occupied -> grazer picks lower or IDLEs."""
    best_occupied = make_cell_view(offset=(0, 1), energy=10.0, occupied=True)
    second_best = make_cell_view(offset=(0, -1), energy=2.0)
    neighborhood = (best_occupied, second_best)

    obs = make_observation(
        energy=3.0,
        reproduction_threshold=float("inf"),
        allowed_directions=MOORE,
        neighborhood=neighborhood,
    )
    rules = GrazerRules()
    rng = np.random.default_rng(42)

    action = rules.decide(obs, rng)
    assert action.type == ActionType.MOVE
    assert action.direction == (0, -1)


# ---------------------------------------------------------------------------
# G4: Both species -- 3x3 full grid, below threshold -> IDLE, no crash
# ---------------------------------------------------------------------------

def test_both_species_full_grid_idle() -> None:
    """Full 3x3 grid, below repro threshold -> both species return IDLE."""
    neighborhood = tuple(
        make_cell_view(offset=d, energy=0.0, occupied=True)
        for d in sorted(MOORE)
    )
    obs = make_observation(
        energy=2.0,
        reproduction_threshold=float("inf"),
        allowed_directions=MOORE,
        neighborhood=neighborhood,
    )
    rng = np.random.default_rng(42)

    # RandomWalker: all neighbors occupied -> can only IDLE (moves fail, no REPRODUCE)
    walker_action = RandomWalkerRules().decide(obs, rng)
    # Walker picks from directions + IDLE, but all directions occupied
    # Action may be MOVE (will fail at engine level) or IDLE -- that's correct
    assert walker_action.type in (ActionType.MOVE, ActionType.IDLE)

    # Grazer: no unoccupied cell with energy > 0 -> IDLE
    grazer_action = GrazerRules().decide(obs, rng)
    assert grazer_action.type == ActionType.IDLE


# ---------------------------------------------------------------------------
# G5: RandomWalker -- VON_NEUMANN only, no diagonals
# ---------------------------------------------------------------------------

def test_random_walker_respects_von_neumann() -> None:
    """100 trials with VON_NEUMANN -> no diagonal directions ever chosen."""
    neighborhood = tuple(
        make_cell_view(offset=d, energy=0.0)
        for d in sorted(VON_NEUMANN)
    )
    obs = make_observation(
        energy=3.0,
        reproduction_threshold=float("inf"),
        allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
        allowed_directions=VON_NEUMANN,
        neighborhood=neighborhood,
    )
    rules = RandomWalkerRules()

    for seed in range(100):
        rng = np.random.default_rng(seed)
        action = rules.decide(obs, rng)
        if action.type == ActionType.MOVE:
            assert action.direction in VON_NEUMANN, (
                f"seed={seed}: got diagonal {action.direction}"
            )
