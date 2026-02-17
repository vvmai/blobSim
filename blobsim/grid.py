"""Toroidal grid with named float64 layers and int32 occupancy tracking.

Spec reference: tech_spec_mvp.md §4 (Grid & Environment).
"""
from __future__ import annotations

import numpy as np

# Moore neighborhood offsets (Chebyshev distance 1), §4.
MOORE_OFFSETS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class Grid:
    """Toroidal nxn grid with named float64 layers and int32 occupancy.

    Attributes:
        size: Grid dimension (n for nxn).
        layers: Named (n,n) float64 arrays. Includes ``"energy"``.
    """

    def __init__(self, size: int, layers: dict[str, np.ndarray]) -> None:
        """Initialize grid with copied layers and empty occupancy.

        Args:
            size: Grid dimension (nxn).
            layers: Named (n,n) float64 arrays. Must include ``"energy"``.

        Raises:
            ValueError: If ``"energy"`` layer is missing or any layer shape
                does not match ``(size, size)``.
        """
        if "energy" not in layers:
            raise ValueError("layers must include an 'energy' layer")
        for name, arr in layers.items():
            if arr.shape != (size, size):
                raise ValueError(
                    f"Layer '{name}' shape {arr.shape} does not match "
                    f"grid size ({size}, {size})"
                )
        self.size: int = size
        self.layers: dict[str, np.ndarray] = {
            name: arr.astype(np.float64, copy=True)
            for name, arr in layers.items()
        }
        self._occupancy: np.ndarray = np.full((size, size), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def energy(self) -> np.ndarray:
        """``"energy"`` layer (NOT a copy -- mutations affect grid)."""
        return self.layers["energy"]

    @property
    def occupancy(self) -> np.ndarray:
        """Read-only view of occupancy. blob_id or -1 (empty)."""
        view = self._occupancy.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------
    # Mutation methods (engine-only, not called by rules engines)
    # ------------------------------------------------------------------

    def place_blob(self, blob_id: int, pos: tuple[int, int]) -> None:
        """Set occupancy at *pos* to *blob_id*."""
        self._occupancy[pos[0], pos[1]] = blob_id

    def remove_blob(self, pos: tuple[int, int]) -> None:
        """Clear occupancy at *pos* to -1 (empty sentinel)."""
        self._occupancy[pos[0], pos[1]] = -1

    def move_blob(
        self, blob_id: int, old: tuple[int, int], new: tuple[int, int]
    ) -> None:
        """Move blob from *old* to *new*.

        Sets old=-1 THEN new=blob_id, so self-moves (old == new) are safe.
        """
        self._occupancy[old[0], old[1]] = -1
        self._occupancy[new[0], new[1]] = blob_id

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def wrap(self, x: int, y: int) -> tuple[int, int]:
        """Toroidal wrapping via Python modulo (correct for negatives)."""
        return x % self.size, y % self.size
