"""Engine mechanics tests (E3, E5, D5).

Engine-level tests using make_engine + make_blob. Same pattern as M1-M6
but targeting gaps: failed reproduce cost, newborn handling, conservative
vacancy when a blob vacates a cell.
"""
from __future__ import annotations

import pytest

from blobsim.types import Action, ActionType, MOORE

from conftest import make_blob, make_engine


# ---------------------------------------------------------------------------
# E3: Failed REPRODUCE pays attempt cost, NOT offspring_energy
# ---------------------------------------------------------------------------

def test_e3_failed_reproduce_cost() -> None:
    """REPRODUCE fails in RESOLVE (collision). Parent pays base_metabolic +
    reproduce_cost, NOT offspring_energy transfer.

    Setup:
    - Blob A at (5,4): REPRODUCE, allowed_directions={(0,1)} -> targets (5,5)
    - Blob B at (4,5): MOVE(1,0) -> targets (5,5)
    - Both claim (5,5). Seed determines winner.
    - Loser pays attempt cost only.

    Expected (regardless of winner):
    - A's REPRODUCE cost = base_metabolic(0.5) + reproduce_cost(0.3) = 0.8
    - If A loses: energy = 10.0 - 0.8 = 9.2  (no offspring_energy deducted)
    - If A wins:  energy = 10.0 - 3.0(transfer) - 0.8 = 6.2
    """
    blob_a = make_blob(
        blob_id=0,
        position=(5, 4),
        energy=10.0,
        action=Action(ActionType.REPRODUCE),
        base_metabolic_cost=0.5,
        reproduce_cost=0.3,
        offspring_energy=3.0,
        reproduction_threshold=5.0,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
        allowed_directions=frozenset({(0, 1)}),
    )
    blob_b = make_blob(
        blob_id=1,
        position=(4, 5),
        energy=10.0,
        action=Action(ActionType.MOVE, direction=(1, 0)),
        base_metabolic_cost=0.5,
    )

    engine = make_engine(grid_size=10, blobs=[blob_a, blob_b])
    engine.step()

    # Determine who won
    a_reproduced = any(b.id != blob_a.id and b.id != blob_b.id for b in engine.blobs)

    if a_reproduced:
        # A succeeded: paid offspring_energy(3.0) + metabolic(0.5) + reproduce_cost(0.3)
        assert blob_a.status.energy == pytest.approx(6.2, abs=1e-15)
    else:
        # A failed: paid only metabolic(0.5) + reproduce_cost(0.3)
        assert blob_a.status.energy == pytest.approx(9.2, abs=1e-15)

    # B always pays metabolic(0.5) + move_cost(0.2) = 0.7
    assert blob_b.status.energy == pytest.approx(9.3, abs=1e-15)


# ---------------------------------------------------------------------------
# E5: Newborn charged metabolic only (no action cost)
# ---------------------------------------------------------------------------

def test_e5_newborn_charged_metabolic_only() -> None:
    """After REPRODUCE, child energy = offspring_energy - base_metabolic."""
    parent = make_blob(
        blob_id=0,
        position=(5, 5),
        energy=10.0,
        action=Action(ActionType.REPRODUCE),
        base_metabolic_cost=0.5,
        reproduce_cost=0.0,
        offspring_energy=3.0,
        reproduction_threshold=5.0,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
    )

    engine = make_engine(grid_size=20, blobs=[parent])
    engine.step()

    children = [b for b in engine.blobs if b.id != parent.id]
    assert len(children) == 1
    child = children[0]

    # Child: 3.0 (received) - 0.5 (metabolic) + 0.0 (no cell energy) = 2.5
    assert child.status.energy == pytest.approx(2.5, abs=1e-15)


# ---------------------------------------------------------------------------
# E5: Newborn absorbs from energy-rich cell
# ---------------------------------------------------------------------------

def test_e5_newborn_absorbs_from_grid() -> None:
    """Child placed on energy-rich cell absorbs grid energy."""
    parent = make_blob(
        blob_id=0,
        position=(5, 5),
        energy=10.0,
        action=Action(ActionType.REPRODUCE),
        base_metabolic_cost=0.5,
        reproduce_cost=0.0,
        offspring_energy=3.0,
        reproduction_threshold=5.0,
        max_energy=20.0,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
    )

    # Fill all MOORE neighbors with energy so child absorbs wherever placed
    overrides = {
        (5 + dr, 5 + dc): 2.0
        for dr, dc in MOORE
    }
    engine = make_engine(grid_size=20, blobs=[parent], cell_overrides=overrides)
    engine.step()

    children = [b for b in engine.blobs if b.id != parent.id]
    assert len(children) == 1
    child = children[0]

    # Child: 3.0 - 0.5(metabolic) + absorbed
    # Absorbed = min(2.0, max_energy - (3.0 - 0.5)) = min(2.0, 17.5) = 2.0
    assert child.status.energy == pytest.approx(4.5, abs=1e-15)


# ---------------------------------------------------------------------------
# E5: Newborn action recorded as IDLE with 0 cost
# ---------------------------------------------------------------------------

def test_e5_newborn_action_recorded_idle() -> None:
    """Newborn's BlobRecord has action_taken=IDLE, action_cost=0.0."""
    parent = make_blob(
        blob_id=0,
        position=(5, 5),
        energy=10.0,
        action=Action(ActionType.REPRODUCE),
        base_metabolic_cost=0.5,
        reproduce_cost=0.0,
        offspring_energy=3.0,
        reproduction_threshold=5.0,
        allowed_actions=frozenset({
            ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE,
        }),
    )

    engine = make_engine(grid_size=20, blobs=[parent])
    engine.step()

    children = [b for b in engine.blobs if b.id != parent.id]
    assert len(children) == 1

    # Find child's BlobRecord in recorder
    child_id = children[0].id
    step_records = engine.recorder.blob_records[0]
    child_rec = next(r for r in step_records if r.blob_id == child_id)

    assert child_rec.action_taken == ActionType.IDLE
    assert child_rec.action_cost == pytest.approx(0.0, abs=1e-15)
    assert child_rec.action_succeeded is True


# ---------------------------------------------------------------------------
# D5: Conservative vacancy -- vacated cell still counts as occupied
# ---------------------------------------------------------------------------

def test_d5_conservative_vacancy_full() -> None:
    """A vacates (2,2)->(2,3). B targets (2,2). B FAILS.

    Conservative vacancy: occupied_cells is built BEFORE moves execute,
    so (2,2) is still in occupied_cells when B's claim is resolved.
    """
    blob_a = make_blob(
        blob_id=0,
        position=(2, 2),
        energy=10.0,
        action=Action(ActionType.MOVE, direction=(0, 1)),
        base_metabolic_cost=0.5,
    )
    blob_b = make_blob(
        blob_id=1,
        position=(2, 1),
        energy=10.0,
        action=Action(ActionType.MOVE, direction=(0, 1)),  # targets (2,2)
        base_metabolic_cost=0.5,
    )

    engine = make_engine(grid_size=10, blobs=[blob_a, blob_b])
    engine.step()

    # A should have moved to (2,3)
    assert blob_a.status.position == (2, 3)

    # B's move to (2,2) should FAIL because (2,2) was in occupied_cells
    # at claim time (conservative vacancy). B stays at (2,1).
    assert blob_b.status.position == (2, 1)

    # B still pays move_cost: 10.0 - 0.5 - 0.2 = 9.3
    assert blob_b.status.energy == pytest.approx(9.3, abs=1e-15)
