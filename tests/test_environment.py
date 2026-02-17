"""Environment unit tests (B4-B6 spec constraints).

Pure unit tests -- Grid + RegeneratingEnvironment only, no engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from blobsim.environment import RegeneratingEnvironment
from blobsim.grid import Grid


def _make_grid(
    size: int,
    cell_energy: float = 0.0,
    occupied_positions: list[tuple[int, int]] | None = None,
) -> Grid:
    layers = {"energy": np.full((size, size), cell_energy)}
    grid = Grid(size, layers)
    if occupied_positions:
        for i, pos in enumerate(occupied_positions):
            grid.place_blob(i, pos)
    return grid


# ---------------------------------------------------------------------------
# B4: Cell at/above capacity -> no addition
# ---------------------------------------------------------------------------

def test_regen_cell_above_capacity_unchanged() -> None:
    """Cell energy > capacity receives 0 addition."""
    env = RegeneratingEnvironment(rate=0.5, capacity=10.0, initial_cell_energy=0.0)
    grid = _make_grid(3, cell_energy=11.0)
    rng = np.random.default_rng(42)

    before = grid.energy.copy()
    injected = env.replenish(grid, step=0, rng=rng)

    np.testing.assert_array_equal(grid.energy, before)
    assert injected == pytest.approx(0.0, abs=1e-15)


def test_regen_cell_exactly_at_capacity() -> None:
    """Cell at exactly capacity -> 0 addition."""
    env = RegeneratingEnvironment(rate=0.5, capacity=10.0, initial_cell_energy=0.0)
    grid = _make_grid(3, cell_energy=10.0)
    rng = np.random.default_rng(42)

    injected = env.replenish(grid, step=0, rng=rng)
    assert injected == pytest.approx(0.0, abs=1e-15)
    assert grid.energy[0, 0] == pytest.approx(10.0, abs=1e-15)


# ---------------------------------------------------------------------------
# B5: Occupied cell -> no addition
# ---------------------------------------------------------------------------

def test_regen_occupied_cell_untouched() -> None:
    """Occupied cell gets no energy addition even if below capacity."""
    env = RegeneratingEnvironment(rate=0.5, capacity=10.0, initial_cell_energy=0.0)
    grid = _make_grid(3, cell_energy=5.0, occupied_positions=[(1, 1)])
    rng = np.random.default_rng(42)

    before_occ = float(grid.energy[1, 1])
    env.replenish(grid, step=0, rng=rng)

    assert grid.energy[1, 1] == pytest.approx(before_occ, abs=1e-15)


# ---------------------------------------------------------------------------
# B4: Rate-limited addition
# ---------------------------------------------------------------------------

def test_regen_rate_limited() -> None:
    """Empty cell with space >> rate -> addition clamped to rate."""
    env = RegeneratingEnvironment(rate=0.3, capacity=10.0, initial_cell_energy=0.0)
    grid = _make_grid(1, cell_energy=0.0)
    rng = np.random.default_rng(42)

    env.replenish(grid, step=0, rng=rng)
    assert grid.energy[0, 0] == pytest.approx(0.3, abs=1e-15)
