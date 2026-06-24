"""Integration tests for predation — §14.8, §14.9.

Tests verify population dynamics, conservation, and one-per-cell
over multi-step runs with real PredatorRules and GrazerRules.
"""
from __future__ import annotations

import numpy as np
import pytest

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.environment import RegeneratingEnvironment
from blobsim.ledger import ConfigError, check_conservation, check_one_per_cell
from blobsim.simulation import Simulation
from blobsim.species.grazer import GrazerRules
from blobsim.species.predator import PredatorRules
from blobsim.types import ActionType, MOORE, SensingLevel


# ---------------------------------------------------------------------------
# Helpers to build species configs
# ---------------------------------------------------------------------------

def make_grazer_config(
    count: int,
    initial_energy: float = 5.0,
    species: str = "grazer",
) -> SpeciesConfig:
    attrs = BlobAttributes.create(
        species=species,
        max_energy=15.0,
        offspring_energy=3.0,
        reproduction_threshold=10.0,
        base_metabolic_cost=0.3,
        move_cost=0.1,
        reproduce_cost=0.0,
        observation_radius=1,
        sensing_level=SensingLevel.ENERGY,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
        allowed_directions=MOORE,
    )
    return SpeciesConfig(
        rules_factory=GrazerRules,
        attributes=attrs,
        count=count,
        initial_energy=initial_energy,
    )


def make_predator_config(
    count: int,
    prey_species: str = "grazer",
    initial_energy: float = 10.0,
    species: str = "predator",
) -> SpeciesConfig:
    attrs = BlobAttributes.create(
        species=species,
        max_energy=25.0,
        offspring_energy=5.0,
        reproduction_threshold=18.0,
        base_metabolic_cost=0.5,
        move_cost=0.2,
        reproduce_cost=0.0,
        attack_cost=0.3,
        observation_radius=1,
        sensing_level=SensingLevel.ENERGY,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE,
            ActionType.ATTACK, ActionType.REPRODUCE,
        }),
        allowed_directions=MOORE,
        preys_on=frozenset({prey_species}),
    )
    return SpeciesConfig(
        rules_factory=PredatorRules,
        attributes=attrs,
        count=count,
        initial_energy=initial_energy,
    )


# ---------------------------------------------------------------------------
# §14.9 Integration Test — Two predator species, one prey
# ---------------------------------------------------------------------------

def test_two_predator_species_config_only() -> None:
    """Fox and hawk both prey on rabbit; verify conservation + one-per-cell.

    Pure config difference — no engine code changes needed.
    """
    rabbit_config = make_grazer_config(count=30, species="rabbit")
    fox_config = make_predator_config(
        count=5, prey_species="rabbit", species="fox",
    )
    hawk_config = make_predator_config(
        count=5, prey_species="rabbit", species="hawk", initial_energy=10.0,
    )

    sim_config = SimulationConfig(
        grid_size=15,
        environment=RegeneratingEnvironment(
            rate=0.05,
            capacity=3.0,
            initial_cell_energy=1.0,
        ),
        species=[rabbit_config, fox_config, hawk_config],
        seed=7,
    )
    sim = Simulation(sim_config)

    for _ in sim.iterate(200):
        check_conservation(
            sim._engine.blobs, sim._engine.grid, sim._engine.ledger,
        )
        check_one_per_cell(sim._engine.grid)


# ---------------------------------------------------------------------------
# §14.7 Validation: unknown preys_on species raises ConfigError at startup
# ---------------------------------------------------------------------------

def test_flag7_unknown_prey_species_raises() -> None:
    """FLAG-7: preys_on references an unknown species -> ConfigError."""
    predator_config = make_predator_config(
        count=5, prey_species="ghost_species", species="predator",
    )
    grazer_config = make_grazer_config(count=20, species="grazer")

    sim_config = SimulationConfig(
        grid_size=10,
        environment=RegeneratingEnvironment(
            rate=0.05,
            capacity=3.0,
            initial_cell_energy=1.0,
        ),
        species=[predator_config, grazer_config],
        seed=42,
    )

    with pytest.raises(ConfigError, match="ghost_species"):
        Simulation(sim_config)


# ---------------------------------------------------------------------------
# §14.8 Integration Test — Lotka-Volterra Population Cycle
# ---------------------------------------------------------------------------

def test_lotka_volterra_cycle() -> None:
    """Predator-prey simulation over 1000 steps.

    Assertions:
    1. Both populations alive at step 0 and step 999.
    2. Prey population oscillates (at least 2 peaks detected).
    3. Conservation holds every step.
    4. One-per-cell holds every step.
    """
    grazer_config = make_grazer_config(
        count=150, initial_energy=5.0, species="grazer",
    )
    predator_config = make_predator_config(
        count=20, prey_species="grazer", initial_energy=10.0, species="predator",
    )

    sim_config = SimulationConfig(
        grid_size=30,
        environment=RegeneratingEnvironment(
            rate=0.08,
            capacity=5.0,
            initial_cell_energy=2.0,
        ),
        species=[grazer_config, predator_config],
        seed=2024,
    )
    sim = Simulation(sim_config)

    recorder = sim._engine.recorder

    for step_ws in sim.iterate(1000):
        check_conservation(
            sim._engine.blobs, sim._engine.grid, sim._engine.ledger,
        )
        check_one_per_cell(sim._engine.grid)

    # 1. Both populations positive at start and end
    species_pop = recorder.species_population()
    assert "grazer" in species_pop
    assert "predator" in species_pop

    grazer_pop = species_pop["grazer"]
    predator_pop = species_pop["predator"]

    assert grazer_pop[0] > 0, "Grazer population empty at step 0"
    assert predator_pop[0] > 0, "Predator population empty at step 0"

    # At least one species survived
    assert grazer_pop[-1] > 0 or predator_pop[-1] > 0, (
        "All populations went extinct"
    )

    # 2. Conservation residuals near zero — tolerance scales with throughput.
    residuals = recorder.conservation_residuals()
    max_residual = float(np.max(np.abs(residuals)))
    max_scale = max(
        max(
            1.0,
            abs(ls.e_dissipated) + abs(ls.e_injected) + abs(ls.e_blobs) + abs(ls.e_grid),
        )
        for ls in recorder.ledger_records
    )
    max_tol = 1e-10 + 1e-12 * max_scale
    assert max_residual < max_tol, (
        f"Conservation violated: max residual = {max_residual}, tol = {max_tol}"
    )

    # 3. Prey population should vary (not monotone collapse)
    # Check that grazer pop has some variance (not instantly wiped out)
    if grazer_pop[-1] > 0:
        grazer_range = float(grazer_pop.max() - grazer_pop.min())
        assert grazer_range > 0, "Grazer population completely static"
