"""Integration-level conservation and reproducibility tests.

Full Simulation(config).run() -- public API only.
"""
from __future__ import annotations

import numpy as np
import pytest

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.environment import DecayEnvironment, RegeneratingEnvironment
from blobsim.grid import Grid
from blobsim.ledger import ConservationError, EnergyLedger, check_conservation
from blobsim.simulation import Simulation
from blobsim.species import RandomWalkerRules
from blobsim.types import ActionType, MOORE


def _walker_attrs(
    *,
    base_metabolic_cost: float = 0.1,
    offspring_energy: float = 1.0,
    reproduction_threshold: float = 5.0,
    max_energy: float = 20.0,
) -> BlobAttributes:
    return BlobAttributes.create(
        species="walker",
        max_energy=max_energy,
        offspring_energy=offspring_energy,
        reproduction_threshold=reproduction_threshold,
        base_metabolic_cost=base_metabolic_cost,
        move_cost=0.05,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
        allowed_directions=MOORE,
    )


# ---------------------------------------------------------------------------
# Test 1: 500-step conservation with RegeneratingEnvironment
# ---------------------------------------------------------------------------

def test_multi_step_conservation_500_steps() -> None:
    """500 steps, 30x30, 50 walkers, RegeneratingEnv. residuals < 1e-10."""
    attrs = _walker_attrs()
    config = SimulationConfig(
        grid_size=30,
        environment=RegeneratingEnvironment(
            rate=0.1, capacity=5.0, initial_cell_energy=1.0,
        ),
        species=[SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=attrs,
            count=50,
            initial_energy=5.0,
        )],
        seed=12345,
    )

    result = Simulation(config).run(steps=500)
    residuals = result.recorder.conservation_residuals()
    max_residual = float(np.abs(residuals).max())
    max_scale = max(
        max(
            1.0,
            abs(ls.e_dissipated) + abs(ls.e_injected) + abs(ls.e_blobs) + abs(ls.e_grid),
        )
        for ls in result.recorder.ledger_records
    )
    max_tol = 1e-10 + 1e-12 * max_scale
    assert max_residual < max_tol, (
        f"max residual = {max_residual}, tol = {max_tol}"
    )


# ---------------------------------------------------------------------------
# Test 2: Reproducibility -- identical runs yield identical ledger
# ---------------------------------------------------------------------------

def test_reproducibility_identical_runs() -> None:
    """Same config+seed twice -> identical LedgerSnapshot at every step."""
    attrs = _walker_attrs()
    config = SimulationConfig(
        grid_size=15,
        environment=RegeneratingEnvironment(
            rate=0.1, capacity=5.0, initial_cell_energy=1.0,
        ),
        species=[SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=attrs,
            count=20,
            initial_energy=5.0,
        )],
        seed=99999,
    )

    result_a = Simulation(config).run(steps=100)
    result_b = Simulation(config).run(steps=100)

    for i, (la, lb) in enumerate(zip(
        result_a.recorder.ledger_records,
        result_b.recorder.ledger_records,
        strict=True,
    )):
        assert la.e_blobs == lb.e_blobs, f"step {i}: e_blobs mismatch"
        assert la.e_grid == lb.e_grid, f"step {i}: e_grid mismatch"
        assert la.e_dissipated == lb.e_dissipated, f"step {i}: e_dissipated mismatch"
        assert la.e_injected == lb.e_injected, f"step {i}: e_injected mismatch"
        assert la.n_alive == lb.n_alive, f"step {i}: n_alive mismatch"
        assert la.n_births == lb.n_births, f"step {i}: n_births mismatch"
        assert la.n_deaths == lb.n_deaths, f"step {i}: n_deaths mismatch"


# ---------------------------------------------------------------------------
# Test 4: Eventual decay -- DecayEnv, population reaches 0
# ---------------------------------------------------------------------------

def test_eventual_decay_population_zero() -> None:
    """DecayEnv, 500 steps -> population = 0."""
    attrs = _walker_attrs(
        base_metabolic_cost=0.2,
        reproduction_threshold=float("inf"),  # no reproduction
    )
    config = SimulationConfig(
        grid_size=10,
        environment=DecayEnvironment(initial_cell_energy=0.0),
        species=[SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=attrs,
            count=10,
            initial_energy=5.0,
        )],
        seed=42,
    )

    result = Simulation(config).run(steps=500)
    final = result.recorder.ledger_records[-1]
    assert final.n_alive == 0, f"Expected 0 alive, got {final.n_alive}"


# ---------------------------------------------------------------------------
# Test 5: Genuine conservation violation raises ConservationError
# ---------------------------------------------------------------------------

def test_genuine_conservation_violation_raises() -> None:
    """check_conservation raises ConservationError when whole joules are missing.

    Simulates a genuine energy leak by recording a false initial_total that
    is 100 J higher than the true system energy. The resulting delta (~100 J)
    is orders of magnitude above the scale-aware tolerance, so ConservationError
    must be raised. This verifies the new relative tolerance still catches
    real bugs.
    """
    grid = Grid(4, {"energy": np.zeros((4, 4), dtype=np.float64)})

    # initial_total is 100 J higher than the actual system energy (0 blobs, 0 grid)
    ledger = EnergyLedger(initial_total=100.0)

    with pytest.raises(ConservationError):
        check_conservation(blobs=[], grid=grid, ledger=ledger)
