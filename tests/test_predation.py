"""Predation unit tests — §14.1 through §14.6.

Tests cover:
- Single predator / single prey energy transfer and kill
- N predators competing on one prey (conflict resolution)
- Prey acting in same step as being killed (ATTACK-first ordering)
- Predator at low energy edge cases
- Mutual predation (A↔B)
"""
from __future__ import annotations

import math

import pytest

from blobsim.blob import Blob, BlobAttributes, BlobStatus
from blobsim.config import SpeciesConfig
from blobsim.conflict import RandomResolver
from blobsim.engine import SimulationEngine
from blobsim.environment import DecayEnvironment
from blobsim.grid import Grid
from blobsim.ledger import (
    ConfigError,
    EnergyLedger,
    RulesEngineError,
    check_conservation,
    check_one_per_cell,
)
from blobsim.observability import StateRecorder
from blobsim.rng import create_rng_hierarchy, make_blob_rng
from blobsim.types import Action, ActionType, MOORE

from conftest import FixedActionRules, make_blob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred_attrs(
    species: str = "predator",
    prey_species: str = "prey",
    max_energy: float = 20.0,
    base_metabolic_cost: float = 0.5,
    attack_cost: float = 0.2,
    move_cost: float = 0.2,
    offspring_energy: float = 1.0,
    reproduction_threshold: float = float("inf"),
) -> BlobAttributes:
    """Build predator BlobAttributes."""
    return BlobAttributes.create(
        species=species,
        max_energy=max_energy,
        offspring_energy=offspring_energy,
        reproduction_threshold=reproduction_threshold,
        base_metabolic_cost=base_metabolic_cost,
        move_cost=move_cost,
        reproduce_cost=0.0,
        attack_cost=attack_cost,
        observation_radius=1,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
        }),
        allowed_directions=MOORE,
        preys_on=frozenset({prey_species}),
    )


def _prey_attrs(
    species: str = "prey",
    max_energy: float = 20.0,
    base_metabolic_cost: float = 0.3,
) -> BlobAttributes:
    """Build prey BlobAttributes."""
    return BlobAttributes.create(
        species=species,
        max_energy=max_energy,
        offspring_energy=1.0,
        reproduction_threshold=float("inf"),
        base_metabolic_cost=base_metabolic_cost,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
        allowed_directions=MOORE,
    )


def make_predator_blob(
    *,
    blob_id: int,
    position: tuple[int, int],
    energy: float,
    attack_direction: tuple[int, int],
    prey_species: str = "prey",
    max_energy: float = 20.0,
    base_metabolic_cost: float = 0.5,
    attack_cost: float = 0.2,
    move_cost: float = 0.2,
    offspring_energy: float = 1.0,
    reproduction_threshold: float = float("inf"),
    species: str = "predator",
) -> Blob:
    """Build a predator blob with FixedActionRules returning ATTACK."""
    attrs = _pred_attrs(
        species=species,
        prey_species=prey_species,
        max_energy=max_energy,
        base_metabolic_cost=base_metabolic_cost,
        attack_cost=attack_cost,
        move_cost=move_cost,
        offspring_energy=offspring_energy,
        reproduction_threshold=reproduction_threshold,
    )
    return Blob(
        blob_id=blob_id,
        attributes=attrs,
        status=BlobStatus(energy=energy, age=0, position=position),
        rules=FixedActionRules(Action(ActionType.ATTACK, direction=attack_direction)),
        rng=make_blob_rng(0, blob_id),
    )


def make_prey_blob(
    *,
    blob_id: int,
    position: tuple[int, int],
    energy: float,
    action: Action,
    max_energy: float = 20.0,
    base_metabolic_cost: float = 0.3,
    species: str = "prey",
) -> Blob:
    """Build a prey blob with FixedActionRules."""
    return make_blob(
        blob_id=blob_id,
        position=position,
        energy=energy,
        action=action,
        max_energy=max_energy,
        base_metabolic_cost=base_metabolic_cost,
        species=species,
        allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
        allowed_directions=MOORE,
    )


def _build_engine(
    blobs: list[Blob],
    species_configs: dict[str, SpeciesConfig],
    grid_size: int = 10,
    seed: int = 42,
) -> SimulationEngine:
    """Build a SimulationEngine from a list of blobs and species configs."""
    env = DecayEnvironment(initial_cell_energy=0.0)
    layers = env.initial_grid(grid_size)
    grid = Grid(grid_size, layers)
    for blob in blobs:
        grid.place_blob(blob.id, blob.status.position)

    e_blobs = math.fsum(b.status.energy for b in blobs)
    e_grid = float(grid.energy.sum())
    ledger = EnergyLedger(e_blobs + e_grid)
    engine_rng, env_rng, blob_pool_entropy = create_rng_hierarchy(seed)

    return SimulationEngine(
        grid=grid,
        blobs=blobs,
        environment=env,
        conflict_resolver=RandomResolver(),
        ledger=ledger,
        engine_rng=engine_rng,
        env_rng=env_rng,
        blob_pool_entropy=blob_pool_entropy,
        species_configs=species_configs,
        recorder=StateRecorder(),
    )


def make_two_species_engine(
    predator: Blob,
    prey: Blob,
    grid_size: int = 10,
    seed: int = 42,
) -> SimulationEngine:
    """Build engine with one predator and one prey blob."""
    all_blobs = [predator, prey]
    species_configs: dict[str, SpeciesConfig] = {}
    for blob in all_blobs:
        sp = blob.attributes.species
        if sp not in species_configs:
            fixed_action = blob.rules._action
            species_configs[sp] = SpeciesConfig(
                rules_factory=lambda a=fixed_action: FixedActionRules(a),
                attributes=blob.attributes,
                count=1,
                initial_energy=blob.status.energy,
            )
    return _build_engine(all_blobs, species_configs, grid_size=grid_size, seed=seed)


# ---------------------------------------------------------------------------
# §14.1 Unit Tests — Data Model (B-P1..B-P4, action_costs, preys_on)
# ---------------------------------------------------------------------------

def test_bp1_attack_without_preys_on_raises() -> None:
    """B-P1: ATTACK in allowed_actions but preys_on empty raises ConfigError."""
    with pytest.raises(ConfigError, match="requires non-empty preys_on"):
        BlobAttributes.create(
            species="wolf",
            max_energy=20.0,
            offspring_energy=1.0,
            reproduction_threshold=10.0,
            base_metabolic_cost=0.5,
            move_cost=0.2,
            reproduce_cost=0.0,
            observation_radius=1,
            allowed_actions=frozenset({
                ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
            }),
            allowed_directions=MOORE,
            preys_on=frozenset(),
        )


def test_bp2_preys_on_without_attack_raises() -> None:
    """B-P2: preys_on non-empty but ATTACK not in allowed_actions raises ConfigError."""
    with pytest.raises(ConfigError, match="ATTACK not in allowed_actions"):
        BlobAttributes.create(
            species="wolf",
            max_energy=20.0,
            offspring_energy=1.0,
            reproduction_threshold=10.0,
            base_metabolic_cost=0.5,
            move_cost=0.2,
            reproduce_cost=0.0,
            observation_radius=1,
            allowed_actions=frozenset({ActionType.IDLE, ActionType.MOVE}),
            allowed_directions=MOORE,
            preys_on=frozenset({"sheep"}),
        )


def test_bp3_self_predation_raises() -> None:
    """B-P3: species cannot prey on itself."""
    with pytest.raises(ConfigError, match="cannot prey on itself"):
        BlobAttributes.create(
            species="wolf",
            max_energy=20.0,
            offspring_energy=1.0,
            reproduction_threshold=10.0,
            base_metabolic_cost=0.5,
            move_cost=0.2,
            reproduce_cost=0.0,
            observation_radius=1,
            allowed_actions=frozenset({
                ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
            }),
            allowed_directions=MOORE,
            preys_on=frozenset({"wolf"}),
        )


def test_bp4_negative_attack_cost_raises() -> None:
    """B-P4: attack_cost < 0 raises ConfigError."""
    with pytest.raises(ConfigError, match="attack_cost must be >= 0"):
        BlobAttributes.create(
            species="wolf",
            max_energy=20.0,
            offspring_energy=1.0,
            reproduction_threshold=10.0,
            base_metabolic_cost=0.5,
            move_cost=0.2,
            reproduce_cost=0.0,
            observation_radius=1,
            allowed_actions=frozenset({
                ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
            }),
            allowed_directions=MOORE,
            preys_on=frozenset({"sheep"}),
            attack_cost=-0.1,
        )


def test_attack_cost_in_action_costs() -> None:
    """attack_cost appears in action_costs dict after create."""
    attrs = BlobAttributes.create(
        species="wolf",
        max_energy=20.0,
        offspring_energy=1.0,
        reproduction_threshold=10.0,
        base_metabolic_cost=0.5,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
        }),
        allowed_directions=MOORE,
        preys_on=frozenset({"sheep"}),
        attack_cost=0.7,
    )
    assert attrs.action_costs[ActionType.ATTACK] == pytest.approx(0.7)


def test_preys_on_immutable() -> None:
    """preys_on is a frozenset (immutable)."""
    attrs = BlobAttributes.create(
        species="wolf",
        max_energy=20.0,
        offspring_energy=1.0,
        reproduction_threshold=10.0,
        base_metabolic_cost=0.5,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
        }),
        allowed_directions=MOORE,
        preys_on=frozenset({"sheep"}),
    )
    assert isinstance(attrs.preys_on, frozenset)
    with pytest.raises((TypeError, AttributeError)):
        attrs.preys_on.add("goat")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# §14.2 Unit Tests — Single Predator, Single Prey
# ---------------------------------------------------------------------------

def test_attack_kills_prey() -> None:
    """After one step: prey not in engine.blobs; predator gained energy."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)

    engine.step()

    prey_ids = {b.id for b in engine.blobs}
    assert 1 not in prey_ids, "Prey should be dead"
    assert 0 in prey_ids, "Predator should be alive"


def test_attack_energy_conservation_full_transfer() -> None:
    """Headroom > prey.energy: predator gains all prey energy, dissipated += 0."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1), max_energy=20.0, attack_cost=0.0,
        base_metabolic_cost=0.0,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey)

    pre_dissipated = engine.ledger.dissipated
    engine.step()

    # Predator gains min(5.0, headroom=10.0) = 5.0; no metabolic cost
    assert engine.blobs[0].status.energy == pytest.approx(15.0, abs=1e-10)
    delta_dissipated = engine.ledger.dissipated - pre_dissipated
    assert delta_dissipated == pytest.approx(0.0, abs=1e-10)


def test_attack_energy_conservation_partial_transfer() -> None:
    """Predator at max-2, prey at 5E: gains 2E, remainder 3E dissipated."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=18.0,
        attack_direction=(0, 1), max_energy=20.0, attack_cost=0.0,
        base_metabolic_cost=0.0,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey)

    pre_dissipated = engine.ledger.dissipated
    engine.step()

    # headroom = 20 - 18 = 2; transfer = min(5, 2) = 2; dissipate = 3
    assert engine.blobs[0].status.energy == pytest.approx(20.0, abs=1e-10)
    delta_dissipated = engine.ledger.dissipated - pre_dissipated
    assert delta_dissipated == pytest.approx(3.0, abs=1e-10)


def test_attack_energy_conservation_zero_headroom() -> None:
    """Predator at max_energy: transfer=0, all prey energy dissipated."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=20.0,
        attack_direction=(0, 1), max_energy=20.0, attack_cost=0.0,
        base_metabolic_cost=0.0,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey)

    pre_dissipated = engine.ledger.dissipated
    engine.step()

    # headroom = 0; transfer = 0; dissipate = 5
    assert engine.blobs[0].status.energy == pytest.approx(20.0, abs=1e-10)
    delta_dissipated = engine.ledger.dissipated - pre_dissipated
    assert delta_dissipated == pytest.approx(5.0, abs=1e-10)
    check_conservation(engine.blobs, engine.grid, engine.ledger)


def test_predator_pays_attack_cost_on_success() -> None:
    """Predator energy after step = pre + transfer - metabolic - attack_cost."""
    attack_cost = 0.3
    metabolic = 0.5
    pre_energy = 10.0
    prey_energy = 5.0

    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=pre_energy,
        attack_direction=(0, 1), max_energy=20.0,
        attack_cost=attack_cost, base_metabolic_cost=metabolic,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=prey_energy,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    headroom = 20.0 - pre_energy
    transfer = min(prey_energy, headroom)
    expected = pre_energy + transfer - metabolic - attack_cost
    assert engine.blobs[0].status.energy == pytest.approx(expected, abs=1e-10)


def test_prey_not_in_blobs_after_attack() -> None:
    """prey.id not in engine._blob_index after kill."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    assert 1 not in engine._blob_index


def test_prey_removed_from_grid_after_attack() -> None:
    """grid.occupancy at prey's former position == -1 after kill."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    assert engine.grid.occupancy[5, 6] == -1


def test_inv1_preserved_single_attack() -> None:
    """check_conservation passes after single predator kills prey."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=12.0,
        attack_direction=(0, 1), attack_cost=0.2,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=4.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    check_conservation(engine.blobs, engine.grid, engine.ledger)


def test_inv3_preserved_single_attack() -> None:
    """check_one_per_cell passes after single predator kills prey."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    check_one_per_cell(engine.grid)


# ---------------------------------------------------------------------------
# §14.3 Unit Tests — N Predators, One Prey
# ---------------------------------------------------------------------------

def _make_n_predators_one_prey_engine(
    n: int, seed: int = 42,
) -> tuple[SimulationEngine, int]:
    """Build engine with n predators all targeting the same prey at (5, 6)."""
    prey_pos = (5, 6)
    pred_positions = [(5, 5), (4, 6), (6, 6)][:n]
    attack_dirs = [(0, 1), (1, 0), (-1, 0)][:n]

    pred_attrs = _pred_attrs()
    prey_attrs = _prey_attrs()

    blobs: list[Blob] = []
    for i, (pos, direction) in enumerate(zip(pred_positions, attack_dirs)):
        blobs.append(Blob(
            blob_id=i,
            attributes=pred_attrs,
            status=BlobStatus(energy=10.0, age=0, position=pos),
            rules=FixedActionRules(Action(ActionType.ATTACK, direction=direction)),
            rng=make_blob_rng(0, i),
        ))
    prey_id = n
    blobs.append(Blob(
        blob_id=prey_id,
        attributes=prey_attrs,
        status=BlobStatus(energy=5.0, age=0, position=prey_pos),
        rules=FixedActionRules(Action(ActionType.IDLE)),
        rng=make_blob_rng(0, prey_id),
    ))

    species_configs = {
        "predator": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=pred_attrs, count=n, initial_energy=10.0,
        ),
        "prey": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=prey_attrs, count=1, initial_energy=5.0,
        ),
    }
    engine = _build_engine(blobs, species_configs, seed=seed)
    return engine, prey_id


def test_n_predators_one_prey_one_wins() -> None:
    """3 predators compete for 1 prey: exactly 1 gains energy, prey dies."""
    engine, prey_id = _make_n_predators_one_prey_engine(3)
    engine.step()

    assert prey_id not in engine._blob_index
    winners = [
        b for b in engine.blobs
        if b.status.energy > 10.0 - 0.5 - 0.2  # above pure-cost baseline
    ]
    assert len(winners) == 1


def test_n_predators_one_prey_losing_pays_cost() -> None:
    """Losing predators pay base_metabolic + attack_cost."""
    engine, _ = _make_n_predators_one_prey_engine(3)
    engine.step()

    # Losers paid 0.5 + 0.2 = 0.7; started at 10.0 -> at most 9.3
    min_energy = min(b.status.energy for b in engine.blobs)
    assert min_energy == pytest.approx(9.3, abs=1e-10)


def test_n_predators_one_prey_conservation() -> None:
    """Conservation holds with 3 predators competing for 1 prey."""
    engine, _ = _make_n_predators_one_prey_engine(3)
    engine.step()
    check_conservation(engine.blobs, engine.grid, engine.ledger)


def test_attack_resolution_deterministic() -> None:
    """Same seed -> same winner across two independent runs."""
    engine1, _ = _make_n_predators_one_prey_engine(3, seed=123)
    engine2, _ = _make_n_predators_one_prey_engine(3, seed=123)

    engine1.step()
    engine2.step()

    winner1 = next(
        b.id for b in engine1.blobs
        if b.status.energy > 10.0 - 0.5 - 0.2
    )
    winner2 = next(
        b.id for b in engine2.blobs
        if b.status.energy > 10.0 - 0.5 - 0.2
    )
    assert winner1 == winner2


# ---------------------------------------------------------------------------
# §14.4 Unit Tests — Prey Acting in Same Step
# ---------------------------------------------------------------------------

def test_prey_move_cancelled_by_predation() -> None:
    """ATTACK-first: prey that chose MOVE is killed before it can move."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
    )
    # Prey at (5,6) wants to move to (5,7)
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.MOVE, direction=(0, 1)),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    # Prey dead: not at (5,7)
    assert engine.grid.occupancy[5, 7] == -1
    assert 1 not in engine._blob_index


def test_prey_reproduce_cancelled_by_predation() -> None:
    """ATTACK-first: prey that chose REPRODUCE is killed before child is born."""
    prey_attrs = BlobAttributes.create(
        species="prey",
        max_energy=20.0,
        offspring_energy=2.0,
        reproduction_threshold=3.0,
        base_metabolic_cost=0.3,
        move_cost=0.2,
        reproduce_cost=0.0,
        observation_radius=1,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
        allowed_directions=MOORE,
    )
    pred_attrs = _pred_attrs()

    predator = Blob(
        blob_id=0, attributes=pred_attrs,
        status=BlobStatus(energy=10.0, age=0, position=(5, 5)),
        rules=FixedActionRules(Action(ActionType.ATTACK, direction=(0, 1))),
        rng=make_blob_rng(0, 0),
    )
    prey = Blob(
        blob_id=1, attributes=prey_attrs,
        status=BlobStatus(energy=10.0, age=0, position=(5, 6)),
        rules=FixedActionRules(Action(ActionType.REPRODUCE)),
        rng=make_blob_rng(0, 1),
    )

    species_configs = {
        "predator": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=pred_attrs, count=1, initial_energy=10.0,
        ),
        "prey": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=prey_attrs, count=1, initial_energy=10.0,
        ),
    }
    engine = _build_engine([predator, prey], species_configs)

    n_births, _ = engine.step()

    # No child should be born (prey killed before REPRODUCE executed)
    assert n_births == 0
    assert 1 not in engine._blob_index
    assert len(engine.blobs) == 1  # only predator survives


def test_killed_prey_does_not_absorb() -> None:
    """Killed prey does not absorb cell energy; cell energy unchanged after step."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1), attack_cost=0.0, base_metabolic_cost=0.0,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey, grid_size=10)
    # Place energy under prey's cell and reinitialize ledger
    engine.grid.energy[5, 6] = 3.0
    new_initial = math.fsum(
        b.status.energy for b in engine.blobs
    ) + float(engine.grid.energy.sum())
    engine.ledger.__init__(new_initial)

    engine.step()

    # Cell (5,6) energy unchanged (prey never absorbed it)
    assert engine.grid.energy[5, 6] == pytest.approx(3.0, abs=1e-10)


def test_killed_prey_does_not_age() -> None:
    """Killed prey does not appear with age+1 in recorder."""
    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    all_records_flat = [
        rec for step_recs in engine.recorder.blob_records for rec in step_recs
    ]
    prey_records = [r for r in all_records_flat if r.blob_id == 1]
    ages = [r.age for r in prey_records]
    assert all(a == 0 for a in ages), f"Killed prey should have age=0, got {ages}"


def test_killed_prey_not_charged() -> None:
    """Killed prey does not pay metabolism; only predator costs are charged."""
    pred_metabolic = 0.5
    pred_attack_cost = 0.2
    prey_energy = 5.0

    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=10.0,
        attack_direction=(0, 1),
        attack_cost=pred_attack_cost, base_metabolic_cost=pred_metabolic,
        max_energy=20.0,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=prey_energy,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.3,
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    # dissipated = pred_metabolic + pred_attack_cost + dissipate_from_prey
    # headroom = 20 - 10 = 10; transfer = min(5, 10) = 5; dissipate = 0
    # prey.base_metabolic_cost NOT charged (dead before CHARGE)
    assert engine.ledger.dissipated == pytest.approx(0.7, abs=1e-10)


# ---------------------------------------------------------------------------
# §14.5 Unit Tests — Predator at Low Energy
# ---------------------------------------------------------------------------

def test_starving_predator_killed_after_attack_cost() -> None:
    """Predator at exact cost; headroom=0 so gains nothing; removed by DEATH_CHECK."""
    metabolic = 0.5
    attack_cost = 0.2
    pred_energy = metabolic + attack_cost  # 0.7; max_energy = 0.7 => headroom=0

    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=pred_energy,
        attack_direction=(0, 1), max_energy=0.7,
        attack_cost=attack_cost, base_metabolic_cost=metabolic,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey)

    _, n_deaths = engine.step()

    assert 0 not in engine._blob_index
    assert 1 not in engine._blob_index
    assert n_deaths >= 1


def test_starving_predator_saved_by_energy_gain() -> None:
    """Predator barely alive; prey energy saves it from starvation."""
    metabolic = 0.5
    attack_cost = 0.2
    pred_energy = metabolic + attack_cost  # 0.7
    max_energy = 10.0

    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=pred_energy,
        attack_direction=(0, 1), max_energy=max_energy,
        attack_cost=attack_cost, base_metabolic_cost=metabolic,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE), base_metabolic_cost=0.0,
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    # headroom = 10.0 - 0.7 = 9.3; transfer = min(5.0, 9.3) = 5.0
    # energy after = 0.7 + 5.0 - 0.5 - 0.2 = 5.0
    assert 0 in engine._blob_index
    assert engine.blobs[0].status.energy == pytest.approx(5.0, abs=1e-10)


def test_predator_cannot_afford_attack_downgrades_to_idle() -> None:
    """Predator energy < metabolic + attack_cost -> downgraded to IDLE, no kill."""
    metabolic = 0.5
    attack_cost = 0.2
    pred_energy = metabolic + attack_cost - 0.01  # just below threshold

    predator = make_predator_blob(
        blob_id=0, position=(5, 5), energy=pred_energy,
        attack_direction=(0, 1),
        attack_cost=attack_cost, base_metabolic_cost=metabolic,
    )
    prey = make_prey_blob(
        blob_id=1, position=(5, 6), energy=5.0,
        action=Action(ActionType.IDLE),
    )
    engine = make_two_species_engine(predator, prey)
    engine.step()

    # Prey survives (attack was downgraded to IDLE)
    assert 1 in engine._blob_index


# ---------------------------------------------------------------------------
# §14.6 Unit Tests — Mutual Predation (A↔B)
# ---------------------------------------------------------------------------

def _make_mutual_predation_engine(seed: int = 42) -> SimulationEngine:
    """A at (5,5) attacks B at (5,6); B at (5,6) attacks A at (5,5)."""
    def _mutual_attrs(species: str, preys_on_species: str) -> BlobAttributes:
        return BlobAttributes.create(
            species=species,
            max_energy=20.0,
            offspring_energy=1.0,
            reproduction_threshold=float("inf"),
            base_metabolic_cost=0.5,
            move_cost=0.2,
            reproduce_cost=0.0,
            attack_cost=0.2,
            observation_radius=1,
            allowed_actions=frozenset({
                ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
            }),
            allowed_directions=MOORE,
            preys_on=frozenset({preys_on_species}),
        )

    attrs_a = _mutual_attrs("A", "B")
    attrs_b = _mutual_attrs("B", "A")

    blob_a = Blob(
        blob_id=0, attributes=attrs_a,
        status=BlobStatus(energy=10.0, age=0, position=(5, 5)),
        rules=FixedActionRules(Action(ActionType.ATTACK, direction=(0, 1))),
        rng=make_blob_rng(0, 0),
    )
    blob_b = Blob(
        blob_id=1, attributes=attrs_b,
        status=BlobStatus(energy=10.0, age=0, position=(5, 6)),
        rules=FixedActionRules(Action(ActionType.ATTACK, direction=(0, -1))),
        rng=make_blob_rng(0, 1),
    )

    species_configs = {
        "A": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=attrs_a, count=1, initial_energy=10.0,
        ),
        "B": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=attrs_b, count=1, initial_energy=10.0,
        ),
    }
    return _build_engine([blob_a, blob_b], species_configs, seed=seed)


def test_mutual_attack_one_survives() -> None:
    """A↔B mutual attack: exactly one survives (sorted-cell order)."""
    engine = _make_mutual_predation_engine(seed=42)
    engine.step()

    assert len(engine.blobs) == 1
    check_conservation(engine.blobs, engine.grid, engine.ledger)


def test_mutual_attack_survivor_is_deterministic() -> None:
    """Same seed -> same survivor in both runs."""
    engine1 = _make_mutual_predation_engine(seed=99)
    engine2 = _make_mutual_predation_engine(seed=99)

    engine1.step()
    engine2.step()

    survivor1 = engine1.blobs[0].id if engine1.blobs else None
    survivor2 = engine2.blobs[0].id if engine2.blobs else None
    assert survivor1 == survivor2


# ---------------------------------------------------------------------------
# §4.4 Engine validation — ATTACK direction checks
# ---------------------------------------------------------------------------

def test_attack_direction_none_raises_validate() -> None:
    """ATTACK with direction=None raises RulesEngineError in VALIDATE."""
    pred_attrs = _pred_attrs()
    prey_attrs = _prey_attrs()

    predator = Blob(
        blob_id=0, attributes=pred_attrs,
        status=BlobStatus(energy=10.0, age=0, position=(5, 5)),
        rules=FixedActionRules(Action(ActionType.ATTACK, direction=None)),
        rng=make_blob_rng(0, 0),
    )
    prey = Blob(
        blob_id=1, attributes=prey_attrs,
        status=BlobStatus(energy=5.0, age=0, position=(5, 6)),
        rules=FixedActionRules(Action(ActionType.IDLE)),
        rng=make_blob_rng(0, 1),
    )

    species_configs = {
        "predator": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=pred_attrs, count=1, initial_energy=10.0,
        ),
        "prey": SpeciesConfig(
            rules_factory=lambda: FixedActionRules(Action(ActionType.IDLE)),
            attributes=prey_attrs, count=1, initial_energy=5.0,
        ),
    }
    engine = _build_engine([predator, prey], species_configs)

    with pytest.raises(RulesEngineError, match="ATTACK requires a direction"):
        engine.step()
