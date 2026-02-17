"""Initial condition factories for benchmark scenarios.

Each function returns a SimulationConfig ready to pass to Simulation().
Seed is always a parameter -- never hardcoded.
"""
from __future__ import annotations

import math

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.environment import DecayEnvironment, RegeneratingEnvironment
from blobsim.species.grazer import GrazerRules
from blobsim.species.random_walker import RandomWalkerRules
from blobsim.types import ActionType, MOORE


# ---------------------------------------------------------------------------
# IC-1: Decay grid, isolated walker (calibration)
# ---------------------------------------------------------------------------

def ic1(
    seed: int,
    *,
    move_cost: float = 0.0,
    initial_energy: float = 100.0,
    max_energy: float = 200.0,
) -> SimulationConfig:
    """IC-1: Decay grid, isolated walker. No grid energy, no reproduction."""
    attrs = BlobAttributes.create(
        species="walker",
        max_energy=max_energy,
        offspring_energy=1.0,
        reproduction_threshold=math.inf,
        base_metabolic_cost=0.1,
        move_cost=move_cost,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
        allowed_directions=MOORE,
    )
    return SimulationConfig(
        grid_size=20,
        environment=DecayEnvironment(initial_cell_energy=0.0),
        species=[SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=attrs,
            count=1,
            initial_energy=initial_energy,
        )],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# IC-2: Decay grid, 2 species with reproduction (convergence)
# ---------------------------------------------------------------------------

_REPRO_ACTIONS = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE})


def ic_decay_multi(seed: int) -> SimulationConfig:
    """IC-2: Decay grid with cell energy, 2 species, reproduction enabled.

    20x20 grid, initial_cell_energy=2.0, 5 walkers + 5 grazers.
    No energy injection -> guaranteed extinction.
    """
    walker_attrs = BlobAttributes.create(
        species="walker",
        max_energy=5.0,
        offspring_energy=0.5,
        reproduction_threshold=1.0,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=_REPRO_ACTIONS,
        allowed_directions=MOORE,
    )
    grazer_attrs = BlobAttributes.create(
        species="grazer",
        max_energy=5.0,
        offspring_energy=0.5,
        reproduction_threshold=1.0,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=_REPRO_ACTIONS,
        allowed_directions=MOORE,
    )

    return SimulationConfig(
        grid_size=20,
        environment=DecayEnvironment(initial_cell_energy=2.0),
        species=[
            SpeciesConfig(
                rules_factory=RandomWalkerRules,
                attributes=walker_attrs,
                count=5,
                initial_energy=1.0,
            ),
            SpeciesConfig(
                rules_factory=GrazerRules,
                attributes=grazer_attrs,
                count=5,
                initial_energy=1.0,
            ),
        ],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# IC-3: Rich decay grid, grazers with low reproduction threshold (convergence)
# ---------------------------------------------------------------------------

def ic_fertile_decay(seed: int) -> SimulationConfig:
    """IC-3: Rich decay grid, grazers with low reproduction threshold.

    30x30 grid, initial_cell_energy=10.0, 5 grazers.
    Boom-bust then guaranteed extinction.
    """
    attrs = BlobAttributes.create(
        species="grazer",
        max_energy=20.0,
        offspring_energy=0.5,
        reproduction_threshold=1.0,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=_REPRO_ACTIONS,
        allowed_directions=MOORE,
    )
    return SimulationConfig(
        grid_size=30,
        environment=DecayEnvironment(initial_cell_energy=10.0),
        species=[SpeciesConfig(
            rules_factory=GrazerRules,
            attributes=attrs,
            count=5,
            initial_energy=1.0,
        )],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# IC-4: Regenerating grid, random walkers at specified density (convergence)
# ---------------------------------------------------------------------------

def ic_carrying_capacity(seed: int, *, count: int) -> SimulationConfig:
    """IC-4: Regenerating grid, random walkers at variable density.

    20x20 grid, RegeneratingEnvironment(rate=0.1, capacity=5.0).
    Used for sparse (count=10) vs dense (count=300) convergence runs.
    """
    attrs = BlobAttributes.create(
        species="walker",
        max_energy=20.0,
        offspring_energy=0.5,
        reproduction_threshold=1.0,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=_REPRO_ACTIONS,
        allowed_directions=MOORE,
    )
    return SimulationConfig(
        grid_size=20,
        environment=RegeneratingEnvironment(
            rate=0.1, capacity=5.0, initial_cell_energy=5.0,
        ),
        species=[SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=attrs,
            count=count,
            initial_energy=1.0,
        )],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# IC-5: Large regenerating grid, single walker, no reproduction (convergence)
# ---------------------------------------------------------------------------

def ic_single_regen(seed: int) -> SimulationConfig:
    """IC-5: Large regenerating grid, single walker, no reproduction.

    50x50 grid, RegeneratingEnvironment(rate=1.0, capacity=5.0).
    Blob energy should converge to max_energy.
    """
    attrs = BlobAttributes.create(
        species="walker",
        max_energy=20.0,
        offspring_energy=0.5,
        reproduction_threshold=math.inf,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
        allowed_directions=MOORE,
    )
    return SimulationConfig(
        grid_size=50,
        environment=RegeneratingEnvironment(
            rate=1.0, capacity=5.0, initial_cell_energy=5.0,
        ),
        species=[SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=attrs,
            count=1,
            initial_energy=1.0,
        )],
        seed=seed,
    )
