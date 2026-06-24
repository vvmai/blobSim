"""Chain-predation tests — §5.3 (coordinate-independent "death cancels action").

A kill chain A→B→C means A's attack on B and B's attack on C both qualify in
RESOLVE (each attacker strictly out-energies its prey). The outcome must be
determined by energy alone, independent of grid cell coordinates:

  - a blob killed this step does NOT execute its own attack;
  - cancellation does NOT cascade past a surviving (freed) blob.

So for A(15)→B(10)→C(5): A kills B; B's attack on C is cancelled; C survives.
For A→B→C→D: A kills B; C (freed) kills D; survivors {A, C}.
"""
from __future__ import annotations

import pytest

from blobsim.blob import Blob, BlobAttributes, BlobStatus
from blobsim.config import SpeciesConfig
from blobsim.engine import SimulationEngine
from blobsim.ledger import check_conservation
from blobsim.rng import make_blob_rng
from blobsim.types import Action, ActionType, MOORE

from conftest import FixedActionRules

from test_predation import _build_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_attrs(species: str, prey: str | None) -> BlobAttributes:
    """Attributes for a chain link. Zero costs / large max_energy so a killer's
    post-step energy is simply start + prey.energy.

    prey=None builds an inert tail (IDLE only, no preys_on).
    """
    if prey is None:
        allowed = frozenset({ActionType.IDLE})
        preys_on = frozenset()
    else:
        allowed = frozenset({ActionType.IDLE, ActionType.ATTACK})
        preys_on = frozenset({prey})
    return BlobAttributes.create(
        species=species,
        max_energy=1000.0,
        offspring_energy=1.0,
        reproduction_threshold=float("inf"),
        base_metabolic_cost=0.0,
        move_cost=0.0,
        reproduce_cost=0.0,
        attack_cost=0.0,
        observation_radius=1,
        allowed_actions=allowed,
        allowed_directions=MOORE,
        preys_on=preys_on,
    )


def _dir(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    return (dst[0] - src[0], dst[1] - src[1])


def _build_chain_engine(
    links: list[tuple[str, float, tuple[int, int]]],
) -> SimulationEngine:
    """Build an engine for a kill chain.

    links is ordered predator→prey: links[i] attacks links[i+1]; the last link
    is an inert tail. Blob ids equal list index, so links[0]=A=id 0, etc.
    """
    blobs: list[Blob] = []
    species_configs: dict[str, SpeciesConfig] = {}

    for i, (species, energy, pos) in enumerate(links):
        prey = links[i + 1][0] if i + 1 < len(links) else None
        attrs = _chain_attrs(species, prey)
        if prey is None:
            action = Action(ActionType.IDLE)
        else:
            action = Action(ActionType.ATTACK, direction=_dir(pos, links[i + 1][2]))
        blobs.append(Blob(
            blob_id=i,
            attributes=attrs,
            status=BlobStatus(energy=energy, age=0, position=pos),
            rules=FixedActionRules(action),
            rng=make_blob_rng(0, i),
        ))
        species_configs[species] = SpeciesConfig(
            rules_factory=lambda a=action: FixedActionRules(a),
            attributes=attrs,
            count=1,
            initial_energy=energy,
        )

    return _build_engine(blobs, species_configs)


def _survivors(engine: SimulationEngine) -> set[str]:
    return {b.attributes.species for b in engine.blobs}


# ---------------------------------------------------------------------------
# §5.3 — 3-chain A→B→C, coordinate independence
# ---------------------------------------------------------------------------

def test_chain_abc_survivors_coordinate_independent() -> None:
    """A(15)→B(10)→C(5): survivors {A, C} regardless of cell sort order."""
    # Forward: prey cells sort ascending B<C → A's prey sorts first.
    fwd = _build_chain_engine([
        ("A", 15.0, (5, 4)), ("B", 10.0, (5, 5)), ("C", 5.0, (5, 6)),
    ])
    # Reverse: prey cells sort ascending C<B → tail's prey sorts first.
    rev = _build_chain_engine([
        ("A", 15.0, (5, 7)), ("B", 10.0, (5, 6)), ("C", 5.0, (5, 5)),
    ])
    fwd.step()
    rev.step()

    assert _survivors(fwd) == {"A", "C"}
    assert _survivors(rev) == {"A", "C"}
    # C untouched in both (its attacker B was killed → B's attack cancelled).
    c_fwd = next(b for b in fwd.blobs if b.attributes.species == "C")
    c_rev = next(b for b in rev.blobs if b.attributes.species == "C")
    assert c_fwd.status.energy == pytest.approx(5.0, abs=1e-10)
    assert c_rev.status.energy == pytest.approx(5.0, abs=1e-10)
    # A absorbed B's full energy (zero costs, ample headroom).
    a_fwd = next(b for b in fwd.blobs if b.attributes.species == "A")
    assert a_fwd.status.energy == pytest.approx(25.0, abs=1e-10)


# ---------------------------------------------------------------------------
# §5.3 — 4-chain A→B→C→D
# ---------------------------------------------------------------------------

def test_chain_four_abcd() -> None:
    """A(15)→B(10)→C(5)→D(2): survivors {A, C}, dead {B, D}, both layouts.

    Parity: A executes (kills B); B killed → cancelled; C freed → executes
    (kills D); D killed. Cancellation does not cascade past freed C.
    """
    fwd = _build_chain_engine([
        ("A", 15.0, (5, 4)), ("B", 10.0, (5, 5)),
        ("C", 5.0, (5, 6)), ("D", 2.0, (5, 7)),
    ])
    rev = _build_chain_engine([
        ("A", 15.0, (5, 7)), ("B", 10.0, (5, 6)),
        ("C", 5.0, (5, 5)), ("D", 2.0, (5, 4)),
    ])
    fwd.step()
    rev.step()

    assert _survivors(fwd) == {"A", "C"}
    assert _survivors(rev) == {"A", "C"}
    # C survived AND ate D → energy 5 + 2 = 7.
    c_fwd = next(b for b in fwd.blobs if b.attributes.species == "C")
    assert c_fwd.status.energy == pytest.approx(7.0, abs=1e-10)


# ---------------------------------------------------------------------------
# §5.3 — conservation
# ---------------------------------------------------------------------------

def test_chain_conservation() -> None:
    """Energy conservation holds after a chain step."""
    engine = _build_chain_engine([
        ("A", 15.0, (5, 4)), ("B", 10.0, (5, 5)),
        ("C", 5.0, (5, 6)), ("D", 2.0, (5, 7)),
    ])
    engine.step()
    # Raises ConservationError if violated.
    check_conservation(engine.blobs, engine.grid, engine.ledger)


# ---------------------------------------------------------------------------
# §5.3 — cancelled attacker records failure
# ---------------------------------------------------------------------------

def test_cancelled_attacker_records_failure() -> None:
    """B (killed by A) has its ATTACK recorded as failed with zero gain."""
    engine = _build_chain_engine([
        ("A", 15.0, (5, 4)), ("B", 10.0, (5, 5)), ("C", 5.0, (5, 6)),
    ])
    engine.step()

    step0 = engine.recorder.blob_records[0]
    b_rec = next(r for r in step0 if r.species == "B")
    assert b_rec.action_taken == ActionType.ATTACK
    assert b_rec.action_succeeded is False
    assert b_rec.energy_gained_from_predation == pytest.approx(0.0, abs=1e-10)
    assert b_rec.alive is False


# ---------------------------------------------------------------------------
# §5.3 — mutual A↔B plus B→C tail: non-cascading cancellation
# ---------------------------------------------------------------------------

def _build_mutual_plus_tail(
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    pos_c: tuple[int, int],
) -> SimulationEngine:
    """A(15)↔B(10) mutual attack; B also attacks C(5). A eats B; B's attack on
    C is cancelled, so C survives. A preys on B, B preys on A and C, C inert."""
    attrs_a = BlobAttributes.create(
        species="A", max_energy=1000.0, offspring_energy=1.0,
        reproduction_threshold=float("inf"), base_metabolic_cost=0.0,
        move_cost=0.0, reproduce_cost=0.0, attack_cost=0.0, observation_radius=1,
        allowed_actions=frozenset({ActionType.IDLE, ActionType.ATTACK}),
        allowed_directions=MOORE, preys_on=frozenset({"B"}),
    )
    attrs_b = BlobAttributes.create(
        species="B", max_energy=1000.0, offspring_energy=1.0,
        reproduction_threshold=float("inf"), base_metabolic_cost=0.0,
        move_cost=0.0, reproduce_cost=0.0, attack_cost=0.0, observation_radius=1,
        allowed_actions=frozenset({ActionType.IDLE, ActionType.ATTACK}),
        allowed_directions=MOORE, preys_on=frozenset({"A", "C"}),
    )
    attrs_c = _chain_attrs("C", None)

    # A attacks B; B attacks C (B is stronger than C and also targeted by A).
    blob_a = Blob(0, attrs_a, BlobStatus(15.0, 0, pos_a),
                  FixedActionRules(Action(ActionType.ATTACK, _dir(pos_a, pos_b))),
                  make_blob_rng(0, 0))
    blob_b = Blob(1, attrs_b, BlobStatus(10.0, 0, pos_b),
                  FixedActionRules(Action(ActionType.ATTACK, _dir(pos_b, pos_c))),
                  make_blob_rng(0, 1))
    blob_c = Blob(2, attrs_c, BlobStatus(5.0, 0, pos_c),
                  FixedActionRules(Action(ActionType.IDLE)), make_blob_rng(0, 2))

    species_configs = {
        "A": SpeciesConfig(lambda: FixedActionRules(Action(ActionType.IDLE)),
                           attrs_a, 1, 15.0),
        "B": SpeciesConfig(lambda: FixedActionRules(Action(ActionType.IDLE)),
                           attrs_b, 1, 10.0),
        "C": SpeciesConfig(lambda: FixedActionRules(Action(ActionType.IDLE)),
                           attrs_c, 1, 5.0),
    }
    return _build_engine([blob_a, blob_b, blob_c], species_configs)


def test_two_cycle_plus_tail_c_survives() -> None:
    """A↔B mutual + B→C: A kills B, B's attack on C cancelled, C survives.

    Coordinate-independent across two layouts.
    """
    # Layout 1: A left, B middle, C right.
    e1 = _build_mutual_plus_tail((5, 4), (5, 5), (5, 6))
    # Layout 2: mirrored so prey-cell sort order differs.
    e2 = _build_mutual_plus_tail((5, 6), (5, 5), (5, 4))
    e1.step()
    e2.step()

    assert _survivors(e1) == {"A", "C"}
    assert _survivors(e2) == {"A", "C"}
    c1 = next(b for b in e1.blobs if b.attributes.species == "C")
    assert c1.status.energy == pytest.approx(5.0, abs=1e-10)
    # A ate B: 15 + 10 = 25.
    a1 = next(b for b in e1.blobs if b.attributes.species == "A")
    assert a1.status.energy == pytest.approx(25.0, abs=1e-10)
    # Conservation holds despite the cancelled B→C attack.
    check_conservation(e1.blobs, e1.grid, e1.ledger)
