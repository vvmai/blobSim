"""Config validation unit tests (C1, C2, B-App constraints).

Pure unit tests -- no engine needed. Tests validate_config() and
BlobAttributes.create() raise ConfigError for invalid inputs.
"""
from __future__ import annotations

import pytest

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig, validate_config
from blobsim.environment import DecayEnvironment
from blobsim.ledger import ConfigError
from blobsim.species import RandomWalkerRules
from blobsim.types import ActionType, MOORE


def _attrs(
    *,
    offspring_energy: float = 1.0,
    base_metabolic_cost: float = 0.5,
    max_energy: float = 20.0,
    reproduction_threshold: float = 5.0,
    allowed_actions: frozenset[ActionType] | None = None,
    allowed_directions: frozenset[tuple[int, int]] | None = None,
    idle_cost: float = 0.0,
    species: str = "walker",
) -> BlobAttributes:
    if allowed_actions is None:
        allowed_actions = frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        })
    if allowed_directions is None:
        allowed_directions = MOORE
    return BlobAttributes.create(
        species=species,
        max_energy=max_energy,
        offspring_energy=offspring_energy,
        reproduction_threshold=reproduction_threshold,
        base_metabolic_cost=base_metabolic_cost,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=allowed_actions,
        allowed_directions=allowed_directions,
        idle_cost=idle_cost,
    )


def _config(
    species_list: list[SpeciesConfig],
    grid_size: int = 10,
) -> SimulationConfig:
    return SimulationConfig(
        grid_size=grid_size,
        environment=DecayEnvironment(initial_cell_energy=0.0),
        species=species_list,
        seed=42,
    )


# ---------------------------------------------------------------------------
# C1: offspring_energy <= base_metabolic_cost
# ---------------------------------------------------------------------------

def test_create_offspring_below_metabolic_raises() -> None:
    with pytest.raises(ConfigError, match="offspring_energy"):
        BlobAttributes.create(
            species="bad",
            max_energy=20.0,
            offspring_energy=0.05,
            reproduction_threshold=5.0,
            base_metabolic_cost=0.1,
            move_cost=0.2,
            reproduce_cost=0.0,
            observation_radius=1,
            allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
            allowed_directions=MOORE,
        )


# ---------------------------------------------------------------------------
# C2: IDLE not in allowed_actions
# ---------------------------------------------------------------------------

def test_create_no_idle_raises() -> None:
    with pytest.raises(ConfigError, match="IDLE"):
        BlobAttributes.create(
            species="bad",
            max_energy=20.0,
            offspring_energy=1.0,
            reproduction_threshold=5.0,
            base_metabolic_cost=0.5,
            move_cost=0.2,
            reproduce_cost=0.0,
            observation_radius=1,
            allowed_actions=frozenset({ActionType.MOVE}),
            allowed_directions=MOORE,
        )


# ---------------------------------------------------------------------------
# B-App: too many blobs
# ---------------------------------------------------------------------------

def test_validate_too_many_blobs_raises() -> None:
    attrs = _attrs()
    sc = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=101,
        initial_energy=5.0,
    )
    with pytest.raises(ConfigError, match="exceeds"):
        validate_config(_config([sc], grid_size=10))


# ---------------------------------------------------------------------------
# B-App: initial_energy <= base_metabolic_cost
# ---------------------------------------------------------------------------

def test_validate_initial_energy_below_metabolic_raises() -> None:
    attrs = _attrs(base_metabolic_cost=0.5)
    sc = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=1,
        initial_energy=0.5,  # == metabolic, must exceed
    )
    with pytest.raises(ConfigError, match="initial_energy"):
        validate_config(_config([sc]))


# ---------------------------------------------------------------------------
# B-App: repro threshold too low
# ---------------------------------------------------------------------------

def test_validate_repro_threshold_too_low_raises() -> None:
    attrs = _attrs(
        offspring_energy=3.0,
        base_metabolic_cost=0.5,
        reproduction_threshold=2.0,  # < 3.0 + 0.5 + 0.0
    )
    sc = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=1,
        initial_energy=5.0,
    )
    with pytest.raises(ConfigError, match="reproduction_threshold"):
        validate_config(_config([sc]))


# ---------------------------------------------------------------------------
# B-App: duplicate species names
# ---------------------------------------------------------------------------

def test_validate_duplicate_species_raises() -> None:
    attrs = _attrs(species="dup")
    sc1 = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=1,
        initial_energy=5.0,
    )
    sc2 = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=1,
        initial_energy=5.0,
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        validate_config(_config([sc1, sc2]))


# ---------------------------------------------------------------------------
# B-App: idle cost nonzero
# ---------------------------------------------------------------------------

def test_validate_idle_cost_nonzero_raises() -> None:
    """action_costs[IDLE] != 0 should fail B10."""
    attrs = _attrs(idle_cost=0.5)
    sc = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=1,
        initial_energy=5.0,
    )
    with pytest.raises(ConfigError, match="IDLE.*must be 0"):
        validate_config(_config([sc]))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_validate_happy_path() -> None:
    attrs = _attrs()
    sc = SpeciesConfig(
        rules_factory=RandomWalkerRules,
        attributes=attrs,
        count=5,
        initial_energy=5.0,
    )
    validate_config(_config([sc]))  # no exception
