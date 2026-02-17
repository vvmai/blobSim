"""Layer 0 mechanical unit tests (M1--M6).

Validates single-step, deterministic engine mechanics against hardcoded
predictions from experimental_validation.md section 4.

Every test constructs SimulationEngine directly and calls engine.step()
once, asserting against literal expected values -- never computed from
the code under test.
"""
from __future__ import annotations

import pytest
from scipy.stats import chisquare

from blobsim.types import Action, ActionType, MOORE

from conftest import make_blob as _make_blob, make_engine as _make_engine


# ---------------------------------------------------------------------------
# M1: Single-Step Energy Balance (4 sub-tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, action, cell_energy, expected_energy",
    [
        ("M1a", Action(ActionType.IDLE), 0.0, 9.5),
        ("M1b", Action(ActionType.IDLE), 3.0, 12.5),
        ("M1c", Action(ActionType.MOVE, direction=(1, 0)), 0.0, 9.3),
        ("M1d", Action(ActionType.MOVE, direction=(1, 0)), 3.0, 12.3),
    ],
    ids=["M1a_idle_no_absorb", "M1b_idle_absorb", "M1c_move_no_absorb", "M1d_move_absorb"],
)
def test_m1_energy_balance(
    label: str,
    action: Action,
    cell_energy: float,
    expected_energy: float,
) -> None:
    """M1: energy_after = E_0 - base_metabolic - action_cost + absorbed."""
    position = (5, 5)

    # For MOVE: destination is (6, 5) -- set cell_energy there, not at origin.
    # For IDLE: blob stays at (5, 5) -- set cell_energy at origin.
    overrides: dict[tuple[int, int], float] | None = None
    if cell_energy > 0:
        if action.type == ActionType.MOVE:
            assert action.direction is not None
            dest = (position[0] + action.direction[0], position[1] + action.direction[1])
            overrides = {dest: cell_energy}
        else:
            overrides = {position: cell_energy}

    blob = _make_blob(
        blob_id=0,
        position=position,
        energy=10.0,
        action=action,
        max_energy=20.0,
        base_metabolic_cost=0.5,
        move_cost=0.2,
    )

    engine = _make_engine(
        grid_size=10,
        blobs=[blob],
        initial_cell_energy=0.0,
        cell_overrides=overrides,
    )

    engine.step()

    assert blob.status.energy == pytest.approx(expected_energy, abs=1e-15)


# ---------------------------------------------------------------------------
# M2: Toroidal Movement Correctness (8 sub-tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "start, direction, expected_pos",
    [
        ((0, 0), (-1, -1), (9, 9)),
        ((0, 0), (-1, 0), (9, 0)),
        ((0, 0), (0, -1), (0, 9)),
        ((9, 9), (1, 1), (0, 0)),
        ((9, 9), (1, 0), (0, 9)),
        ((9, 9), (0, 1), (9, 0)),
        ((5, 5), (1, 0), (6, 5)),
        ((5, 5), (-1, 1), (4, 6)),
    ],
    ids=["M2a", "M2b", "M2c", "M2d", "M2e", "M2f", "M2g", "M2h"],
)
def test_m2_toroidal_movement(
    start: tuple[int, int],
    direction: tuple[int, int],
    expected_pos: tuple[int, int],
) -> None:
    """M2: blob at start + MOVE(direction) wraps correctly on 10x10 torus."""
    blob = _make_blob(
        blob_id=0,
        position=start,
        energy=10.0,
        action=Action(ActionType.MOVE, direction=direction),
    )

    engine = _make_engine(grid_size=10, blobs=[blob])
    engine.step()

    assert blob.status.position == expected_pos


# ---------------------------------------------------------------------------
# M3: Reproduction Energy Transfer
# ---------------------------------------------------------------------------

def test_m3_reproduction_energy_transfer() -> None:
    """M3: parent pays offspring_energy + metabolism + reproduce_cost.

    Child receives offspring_energy, pays metabolism only.
    Conservation: 6.2 + 2.5 + 1.3 == 10.0.
    """
    parent = _make_blob(
        blob_id=0,
        position=(10, 10),
        energy=10.0,
        action=Action(ActionType.REPRODUCE),
        max_energy=20.0,
        base_metabolic_cost=0.5,
        reproduce_cost=0.3,
        offspring_energy=3.0,
        reproduction_threshold=5.0,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
    )

    engine = _make_engine(grid_size=20, blobs=[parent])
    engine.step()

    # Parent: 10.0 - 3.0 (transfer) - 0.5 (metabolism) - 0.3 (action) = 6.2
    assert parent.status.energy == pytest.approx(6.2, abs=1e-15)

    # Exactly one child should have been created
    children = [b for b in engine.blobs if b.id != parent.id]
    assert len(children) == 1
    child = children[0]

    # Child: 3.0 (received) - 0.5 (metabolism) = 2.5
    assert child.status.energy == pytest.approx(2.5, abs=1e-15)
    assert child.status.alive is True

    # Child position is a MOORE offset from parent
    pr, pc = parent.status.position
    cr, cc = child.status.position
    dr = (cr - pr) % 20
    dc = (cc - pc) % 20
    # Offset must be in MOORE when mapped to [-1, 0, 1]
    if dr > 1:
        dr -= 20
    if dc > 1:
        dc -= 20
    assert (dr, dc) in MOORE

    # Conservation: parent + child + dissipated == 10.0
    dissipated = engine.ledger.dissipated
    assert dissipated == pytest.approx(1.3, abs=1e-15)
    assert parent.status.energy + child.status.energy + dissipated == pytest.approx(
        10.0, abs=1e-15,
    )


# ---------------------------------------------------------------------------
# M4: Failed Action Cost Accounting
# ---------------------------------------------------------------------------

def test_m4_failed_action_cost() -> None:
    """M4: failed MOVE pays move_cost (0.2), not idle_cost (0.0).

    Blob A at (5,4) moves toward occupied (5,5). Fails. Pays 0.5+0.2=0.7.
    Blob B at (5,5) idles. Pays 0.5+0.0=0.5.
    """
    blob_a = _make_blob(
        blob_id=0,
        position=(5, 4),
        energy=10.0,
        action=Action(ActionType.MOVE, direction=(0, 1)),
        base_metabolic_cost=0.5,
        move_cost=0.2,
    )
    blob_b = _make_blob(
        blob_id=1,
        position=(5, 5),
        energy=10.0,
        action=Action(ActionType.IDLE),
        base_metabolic_cost=0.5,
        move_cost=0.2,
    )

    engine = _make_engine(grid_size=10, blobs=[blob_a, blob_b])
    engine.step()

    # A: 10.0 - 0.5 - 0.2 = 9.3  (move_cost, NOT idle_cost=0.0)
    assert blob_a.status.energy == pytest.approx(9.3, abs=1e-15)
    assert blob_a.status.position == (5, 4)

    # B: 10.0 - 0.5 - 0.0 = 9.5
    assert blob_b.status.energy == pytest.approx(9.5, abs=1e-15)


# ---------------------------------------------------------------------------
# M5: Transient Negative Energy Recovery (2 cases)
# ---------------------------------------------------------------------------

def test_m5a_transient_negative_recovery() -> None:
    """M5 Case A: energy goes negative in CHARGE, recovers in ABSORB.

    CHARGE: 0.05 - 0.1 = -0.05 (not clamped)
    ABSORB: -0.05 + 1.0 = 0.95
    DEATH:  0.95 > 1e-10 -> survives
    """
    blob = _make_blob(
        blob_id=0,
        position=(5, 5),
        energy=0.05,
        action=Action(ActionType.IDLE),
        max_energy=10.0,
        base_metabolic_cost=0.1,
        move_cost=0.0,
    )

    engine = _make_engine(
        grid_size=10,
        blobs=[blob],
        cell_overrides={(5, 5): 1.0},
    )
    engine.step()

    assert blob.status.alive is True
    assert blob.status.energy == pytest.approx(0.95, abs=1e-15)


def test_m5b_transient_negative_death() -> None:
    """M5 Case B: energy goes negative and stays negative -> death.

    CHARGE: 0.05 - 0.1 = -0.05
    ABSORB: -0.05 + 0.0 = -0.05
    DEATH:  -0.05 <= 1e-10 -> dies
    """
    blob = _make_blob(
        blob_id=0,
        position=(5, 5),
        energy=0.05,
        action=Action(ActionType.IDLE),
        max_energy=10.0,
        base_metabolic_cost=0.1,
        move_cost=0.0,
    )

    engine = _make_engine(grid_size=10, blobs=[blob])
    engine.step()

    # Blob removed from engine
    assert len(engine.blobs) == 0

    # Death remainder (-0.05) was dissipated into ledger.
    # Total dissipated = metabolism (0.1) + death remainder (-0.05) = 0.05
    # Conservation: initial=0.05, dissipated=0.05, grid=0, blobs=0
    # 0 + 0 + 0.05 - 0 == 0.05  (passes conservation check)
    assert engine.ledger.dissipated == pytest.approx(0.05, abs=1e-15)


# ---------------------------------------------------------------------------
# M6: Conflict Resolution Fairness
# ---------------------------------------------------------------------------

def test_m6_conflict_resolution_fairness() -> None:
    """M6: 3 blobs targeting the same empty cell, 300 trials.

    Each blob should win ~100 times. Chi-squared p > 0.01, no blob
    wins < 70 or > 130.
    """
    n_trials = 300
    wins = {0: 0, 1: 0, 2: 0}

    for i in range(n_trials):
        seed = 42 + i

        blob_a = _make_blob(
            blob_id=0,
            position=(5, 4),
            energy=10.0,
            action=Action(ActionType.MOVE, direction=(0, 1)),
        )
        blob_b = _make_blob(
            blob_id=1,
            position=(5, 6),
            energy=10.0,
            action=Action(ActionType.MOVE, direction=(0, -1)),
        )
        blob_c = _make_blob(
            blob_id=2,
            position=(4, 5),
            energy=10.0,
            action=Action(ActionType.MOVE, direction=(1, 0)),
        )

        engine = _make_engine(
            grid_size=10,
            blobs=[blob_a, blob_b, blob_c],
            seed=seed,
        )
        engine.step()

        for blob in engine.blobs:
            if blob.status.position == (5, 5):
                wins[blob.id] += 1

    observed = [wins[0], wins[1], wins[2]]
    expected_freq = [n_trials / 3] * 3

    _, p_value = chisquare(observed, f_exp=expected_freq)

    assert p_value > 0.01, (
        f"Chi-squared p={p_value:.4f} <= 0.01; wins={observed}"
    )
    for blob_id in range(3):
        assert 70 <= wins[blob_id] <= 130, (
            f"Blob {blob_id} won {wins[blob_id]} times, outside [70, 130]"
        )
