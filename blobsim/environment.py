"""Pluggable environment functions for grid energy dynamics.

Spec reference: tech_spec_mvp.md section 4 (Grid & Environment).

EnvironmentFn defines the ABC; concrete implementations control how
(and whether) grid energy replenishes each step. The replenish()
return value feeds directly into the energy ledger's E_injected
accumulator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from blobsim.grid import Grid


class EnvironmentFn(ABC):
    """Abstract base for environment replenishment strategies."""

    @abstractmethod
    def replenish(
        self, grid: Grid, step: int, rng: np.random.Generator
    ) -> float:
        """Modify grid layers in-place.

        Returns total energy injected this step.
        """
        ...

    @abstractmethod
    def initial_grid(
        self, grid_size: int
    ) -> dict[str, np.ndarray]:
        """Return layer_name -> (n,n) initial arrays.

        Must include 'energy'.
        """
        ...


class DecayEnvironment(EnvironmentFn):
    """No replenishment. All blobs eventually die.

    Useful for testing energy conservation in closed systems.
    """

    def __init__(self, initial_cell_energy: float) -> None:
        self.initial_cell_energy = initial_cell_energy

    def replenish(
        self, grid: Grid, step: int, rng: np.random.Generator
    ) -> float:
        return 0.0

    def initial_grid(
        self, grid_size: int
    ) -> dict[str, np.ndarray]:
        energy = np.full(
            (grid_size, grid_size), self.initial_cell_energy
        )
        return {"energy": energy}


class RegeneratingEnvironment(EnvironmentFn):
    """Unoccupied cells regrow toward capacity per step.

    - Cells at or above capacity receive no addition (B4).
    - Occupied cells receive no addition (B5).
    - Return value equals exact energy injected (B6).
    """

    def __init__(
        self,
        rate: float,
        capacity: float,
        initial_cell_energy: float,
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self.initial_cell_energy = initial_cell_energy

    def replenish(
        self, grid: Grid, step: int, rng: np.random.Generator
    ) -> float:
        energy = grid.energy             # mutable view (B7)
        unoccupied = grid.occupancy < 0  # read-only mask
        space = self.capacity - energy   # room to grow
        addition = np.minimum(
            self.rate, np.maximum(space, 0.0)
        )
        addition *= unoccupied           # only empty cells
        energy += addition               # mutate in-place
        return float(addition.sum())     # exact injection

    def initial_grid(
        self, grid_size: int
    ) -> dict[str, np.ndarray]:
        energy = np.full(
            (grid_size, grid_size), self.initial_cell_energy
        )
        return {"energy": energy}
