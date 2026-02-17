"""Energy ledger and conservation invariant checks.

Provides compensated summation (math.fsum) for energy tracking and
per-step conservation validation. See tech_spec_mvp.md section 11.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from blobsim.blob import Blob
    from blobsim.grid import Grid


class ConservationError(RuntimeError):
    """Raised when energy conservation invariant is violated."""


class InvariantError(RuntimeError):
    """Raised when structural invariants are violated (INV-3, INV-5)."""


class RulesEngineError(RuntimeError):
    """Raised on rules engine contract violations."""


class ConfigError(ValueError):
    """Raised on invalid simulation configuration."""


class EnergyLedger:
    """Tracks energy dissipated and injected across simulation steps.

    Uses math.fsum on delta lists (not running accumulators) to avoid
    floating-point drift over long simulations.
    """

    def __init__(self, initial_total: float) -> None:
        self.initial_total = initial_total
        self._dissipated_deltas: list[float] = []
        self._injected_deltas: list[float] = []

    @property
    def dissipated(self) -> float:
        """Total energy dissipated (metabolism + death), via compensated sum."""
        return math.fsum(self._dissipated_deltas)

    @property
    def injected(self) -> float:
        """Total energy injected (environment replenishment), via compensated sum."""
        return math.fsum(self._injected_deltas)

    def add_dissipated(self, amount: float) -> None:
        """Record a dissipation delta for this step."""
        self._dissipated_deltas.append(amount)

    def add_injected(self, amount: float) -> None:
        """Record an injection delta for this step."""
        self._injected_deltas.append(amount)


def check_conservation(
    blobs: Iterable[Blob],
    grid: Grid,
    ledger: EnergyLedger,
    tolerance: float = 1e-10,
) -> None:
    """Verify energy conservation: E_blobs + E_grid + dissipated - injected == initial.

    Uses math.fsum for blob energy summation.
    Uses if/raise (not assert) so checks survive python -O.

    Args:
        blobs: Iterable of objects with .status.alive and .status.energy.
        grid: Object with .energy ndarray property (numpy sum called externally).
        ledger: EnergyLedger tracking dissipated/injected totals.
        tolerance: Maximum allowed absolute deviation.

    Raises:
        ConservationError: If |delta| > tolerance.
    """
    e_blobs = math.fsum(b.status.energy for b in blobs if b.status.alive)
    e_grid = float(grid.energy.sum())
    lhs = e_blobs + e_grid + ledger.dissipated - ledger.injected
    delta = lhs - ledger.initial_total
    if abs(delta) > tolerance:
        raise ConservationError(
            f"Conservation violated: delta={delta}, "
            f"E_blobs={e_blobs}, E_grid={e_grid}, "
            f"dissipated={ledger.dissipated}, injected={ledger.injected}"
        )


def check_one_per_cell(grid: Grid) -> None:
    """Verify INV-3: no cell contains more than one blob.

    Checks that every non-negative blob_id in the occupancy grid is unique.
    Duplicate blob_ids indicate a single blob mapped to multiple cells,
    which violates the one-blob-per-cell invariant.

    Uses if/raise (not assert) so checks survive python -O.

    Args:
        grid: Object with .occupancy ndarray property (int32, -1 for empty).

    Raises:
        InvariantError: If duplicate blob_ids are found in occupancy grid.
    """
    occupancy = grid.occupancy
    occupied_ids = occupancy[occupancy >= 0].ravel()
    n_occupied = len(occupied_ids)
    n_unique = len(set(occupied_ids.tolist()))
    if n_unique != n_occupied:
        raise InvariantError(
            f"One-per-cell violated: {n_occupied} occupied cells "
            f"but only {n_unique} unique blob IDs"
        )
