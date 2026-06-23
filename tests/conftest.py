"""Shared test helpers and fixtures.

Extracted from test_integration_mechanical.py + new helpers for
species rules engine testing.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from blobsim.blob import Blob, BlobAttributes, BlobStatus
from blobsim.config import SpeciesConfig
from blobsim.conflict import RandomResolver
from blobsim.engine import SimulationEngine
from blobsim.environment import DecayEnvironment, EnvironmentFn
from blobsim.grid import Grid
from blobsim.ledger import EnergyLedger
from blobsim.observability import StateRecorder
from blobsim.rng import create_rng_hierarchy, make_blob_rng
from blobsim.rules import RulesEngine
from blobsim.types import (
    Action,
    ActionType,
    CellView,
    MOORE,
    Observation,
    SensingLevel,
)


# ---------------------------------------------------------------------------
# Deterministic rules engine
# ---------------------------------------------------------------------------

class FixedActionRules(RulesEngine):
    """Returns the same action every step."""

    def __init__(self, action: Action) -> None:
        self._action = action

    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        return self._action


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def make_engine(
    *,
    grid_size: int,
    blobs: list[Blob],
    initial_cell_energy: float = 0.0,
    cell_overrides: dict[tuple[int, int], float] | None = None,
    seed: int = 42,
    environment: EnvironmentFn | None = None,
) -> SimulationEngine:
    """Build a SimulationEngine with exact blob placement.

    Args:
        grid_size: Side length of the toroidal grid.
        blobs: Pre-constructed blobs (positions already set in BlobStatus).
        initial_cell_energy: Uniform initial energy for all cells.
        cell_overrides: Per-cell energy overrides applied after grid init.
        seed: Root RNG seed.
        environment: Optional environment; defaults to DecayEnvironment.

    Returns:
        A ready-to-step SimulationEngine.
    """
    if environment is None:
        environment = DecayEnvironment(initial_cell_energy=initial_cell_energy)

    layers = environment.initial_grid(grid_size)

    if cell_overrides:
        for (r, c), energy in cell_overrides.items():
            layers["energy"][r, c] = energy

    grid = Grid(grid_size, layers)

    for blob in blobs:
        grid.place_blob(blob.id, blob.status.position)

    e_blobs = math.fsum(b.status.energy for b in blobs)
    e_grid = float(grid.energy.sum())
    ledger = EnergyLedger(e_blobs + e_grid)

    engine_rng, env_rng, blob_pool_entropy = create_rng_hierarchy(seed)

    species_configs: dict[str, SpeciesConfig] = {}
    for blob in blobs:
        sp = blob.attributes.species
        if sp not in species_configs:
            rules = blob.rules
            assert isinstance(rules, FixedActionRules)
            fixed_action = rules._action
            species_configs[sp] = SpeciesConfig(
                rules_factory=lambda a=fixed_action: FixedActionRules(a),
                attributes=blob.attributes,
                count=1,
                initial_energy=blob.status.energy,
            )

    return SimulationEngine(
        grid=grid,
        blobs=list(blobs),
        environment=environment,
        conflict_resolver=RandomResolver(),
        ledger=ledger,
        engine_rng=engine_rng,
        env_rng=env_rng,
        blob_pool_entropy=blob_pool_entropy,
        species_configs=species_configs,
        recorder=StateRecorder(),
    )


# ---------------------------------------------------------------------------
# Blob factory
# ---------------------------------------------------------------------------

def make_blob(
    *,
    blob_id: int,
    position: tuple[int, int],
    energy: float,
    action: Action,
    max_energy: float = 20.0,
    base_metabolic_cost: float = 0.5,
    move_cost: float = 0.2,
    reproduce_cost: float = 0.0,
    offspring_energy: float = 1.0,
    reproduction_threshold: float = float("inf"),
    allowed_actions: frozenset[ActionType] | None = None,
    allowed_directions: frozenset[tuple[int, int]] | None = None,
    blob_pool_entropy: int = 0,
    species: str = "test",
) -> Blob:
    """Build a Blob with FixedActionRules for deterministic testing."""
    if allowed_actions is None:
        allowed_actions = frozenset({ActionType.IDLE, ActionType.MOVE})
    if allowed_directions is None:
        allowed_directions = MOORE

    attrs = BlobAttributes.create(
        species=species,
        max_energy=max_energy,
        offspring_energy=offspring_energy,
        reproduction_threshold=reproduction_threshold,
        base_metabolic_cost=base_metabolic_cost,
        move_cost=move_cost,
        reproduce_cost=reproduce_cost,
        observation_radius=1,
        allowed_actions=allowed_actions,
        allowed_directions=allowed_directions,
    )

    return Blob(
        blob_id=blob_id,
        attributes=attrs,
        status=BlobStatus(energy=energy, age=0, position=position),
        rules=FixedActionRules(action),
        rng=make_blob_rng(blob_pool_entropy, blob_id),
    )


# ---------------------------------------------------------------------------
# Observation factory for direct rules engine testing
# ---------------------------------------------------------------------------

def make_observation(
    *,
    energy: float,
    age: int = 0,
    position: tuple[int, int] = (5, 5),
    step: int = 0,
    species: str = "test",
    max_energy: float = 20.0,
    base_metabolic_cost: float = 0.5,
    offspring_energy: float = 1.0,
    reproduction_threshold: float = float("inf"),
    allowed_actions: frozenset[ActionType] | None = None,
    allowed_directions: frozenset[tuple[int, int]] | None = None,
    neighborhood: tuple[CellView, ...] = (),
    current_cell_energy: float = 0.0,
) -> Observation:
    """Build a synthetic Observation for direct rules engine testing."""
    if allowed_actions is None:
        allowed_actions = frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        })
    if allowed_directions is None:
        allowed_directions = MOORE

    attrs = BlobAttributes.create(
        species=species,
        max_energy=max_energy,
        offspring_energy=offspring_energy,
        reproduction_threshold=reproduction_threshold,
        base_metabolic_cost=base_metabolic_cost,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        sensing_level=SensingLevel.BASIC,
        allowed_actions=allowed_actions,
        allowed_directions=allowed_directions,
    )

    status = BlobStatus(energy=energy, age=age, position=position)

    current_cell = CellView(
        offset=(0, 0),
        layers=MappingProxyType({"energy": current_cell_energy}),
        occupied=True,
    )

    return Observation(
        self_status=status,
        self_attributes=attrs,
        neighborhood=neighborhood,
        current_cell=current_cell,
        step=step,
    )


def make_cell_view(
    offset: tuple[int, int],
    energy: float = 0.0,
    occupied: bool = False,
    occupant_species: str | None = None,
) -> CellView:
    """Build a CellView for use in synthetic neighborhoods."""
    return CellView(
        offset=offset,
        layers=MappingProxyType({"energy": energy}),
        occupied=occupied,
        occupant_species=occupant_species,
    )
