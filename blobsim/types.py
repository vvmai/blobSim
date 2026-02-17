"""Shared data types for the blob simulation.

Spec reference: tech_spec_mvp.md sections 3, 6, 8, 10, 12, 14.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blobsim.blob import BlobAttributes, BlobStatus


# ---------------------------------------------------------------------------
# Enums (§6, §8)
# ---------------------------------------------------------------------------

class SensingLevel(Enum):
    """What info is visible about neighboring blobs."""
    BASIC = "basic"
    ENERGY = "energy"
    FULL = "full"


class ActionType(Enum):
    """Possible blob actions per step."""
    IDLE = "idle"
    MOVE = "move"
    REPRODUCE = "reproduce"


# ---------------------------------------------------------------------------
# Direction presets & constants (§3, §8)
# ---------------------------------------------------------------------------

VON_NEUMANN: frozenset[tuple[int, int]] = frozenset({
    (-1, 0), (1, 0), (0, -1), (0, 1),
})

MOORE: frozenset[tuple[int, int]] = frozenset({
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
})

DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "NE": (-1, 1),
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
}


# ---------------------------------------------------------------------------
# Action types (§8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """A single blob action. Direction required for MOVE, None otherwise."""
    type: ActionType
    direction: tuple[int, int] | None = None


@dataclass(frozen=True)
class ResolvedAction:
    """Output of conflict resolution; input to EXECUTE phase."""
    blob_id: int
    action: Action
    target: tuple[int, int] | None
    succeeded: bool


# ---------------------------------------------------------------------------
# Conflict resolution (§10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """A blob's claim on a target cell during CLAIM phase."""
    blob_id: int
    target: tuple[int, int]
    action: Action
    priority: float = 0.0


# ---------------------------------------------------------------------------
# Observation & sensing (§6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellView:
    """Frozen snapshot of a single cell as seen by an observer blob."""
    offset: tuple[int, int]
    layers: MappingProxyType[str, float]
    occupied: bool
    occupant_species: str | None = None
    occupant_energy: float | None = None
    occupant_age: int | None = None

    @property
    def energy(self) -> float:
        """Convenience accessor for the energy grid layer."""
        return self.layers["energy"]


@dataclass(frozen=True)
class Observation:
    """Frozen snapshot delivered to a rules engine at start of step.

    Forward refs: BlobStatus, BlobAttributes (blob.py).
    """
    self_status: "BlobStatus"  # noqa: F821
    self_attributes: "BlobAttributes"  # noqa: F821
    neighborhood: tuple[CellView, ...]
    current_cell: CellView
    step: int


# ---------------------------------------------------------------------------
# Observability (§12)
# ---------------------------------------------------------------------------

@dataclass
class StepData:
    """Mutable accumulator for per-blob data collected across phases 5-8."""
    action_taken: ActionType
    action_succeeded: bool
    action_cost: float
    energy_absorbed: float


@dataclass(frozen=True)
class BlobRecord:
    """Immutable per-blob-per-step record for observability export."""
    step: int
    blob_id: int
    species: str
    position: tuple[int, int]
    energy: float
    age: int
    action_taken: ActionType
    action_succeeded: bool
    action_cost: float
    energy_absorbed: float
    alive: bool


@dataclass(frozen=True)
class LedgerSnapshot:
    """Immutable per-step energy ledger snapshot."""
    step: int
    e_blobs: float
    e_grid: float
    e_dissipated: float
    e_injected: float
    n_alive: int
    n_births: int
    n_deaths: int


# ---------------------------------------------------------------------------
# User API (§14)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldState:
    """Public per-step summary yielded by Simulation.iterate()."""
    step: int
    n_alive: int
    n_births: int
    n_deaths: int
    e_blobs: float
    e_grid: float
    e_dissipated: float
    e_injected: float
