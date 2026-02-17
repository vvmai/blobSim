"""State recording and query helpers for simulation observability.

Spec reference: tech_spec_mvp.md section 12 (Observability Module).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from blobsim.types import ActionType, BlobRecord, LedgerSnapshot, StepData

if TYPE_CHECKING:
    from blobsim.blob import Blob
    from blobsim.grid import Grid
    from blobsim.ledger import EnergyLedger


class StateRecorder:
    """Captures full simulation state per step.

    Dual storage: structured snapshots for analysis + columnar export
    via ``to_dataframes()``.

    ``record_interval`` controls grid snapshot frequency only.
    Blob records and ledger snapshots are captured every step.
    """

    def __init__(self, record_interval: int = 1) -> None:
        self.record_interval = record_interval
        self.blob_records: list[list[BlobRecord]] = []
        self.grid_records: list[np.ndarray | None] = []
        self.ledger_records: list[LedgerSnapshot] = []
        self._initial_total: float | None = None

    def snapshot(
        self,
        step: int,
        blobs: list[Blob],
        grid: Grid,
        ledger: EnergyLedger,
        step_data: dict[int, StepData],
        n_births: int,
        n_deaths: int,
    ) -> None:
        """Record state after a step completes.

        Args:
            step: Current step number.
            blobs: All blobs this step (incl. dead, alive=False).
            grid: Current grid state.
            ledger: Energy ledger with dissipated/injected totals.
            step_data: Per-blob accumulated data from phases 5-8.
            n_births: Number of newborns this step.
            n_deaths: Number of deaths this step.
        """
        if self._initial_total is None:
            self._initial_total = ledger.initial_total

        # Build BlobRecord for each blob
        blob_recs: list[BlobRecord] = []
        for blob in blobs:
            sd = step_data.get(blob.id)
            blob_recs.append(BlobRecord(
                step=step,
                blob_id=blob.id,
                species=blob.attributes.species,
                position=blob.status.position,
                energy=blob.status.energy,
                age=blob.status.age,
                action_taken=sd.action_taken if sd else ActionType.IDLE,
                action_succeeded=sd.action_succeeded if sd else True,
                action_cost=sd.action_cost if sd else 0.0,
                energy_absorbed=sd.energy_absorbed if sd else 0.0,
                alive=blob.status.alive,
            ))
        self.blob_records.append(blob_recs)

        # Grid snapshot (only at intervals)
        if step % self.record_interval == 0:
            self.grid_records.append(grid.energy.copy())
        else:
            self.grid_records.append(None)

        # Ledger snapshot
        e_blobs = math.fsum(b.status.energy for b in blobs if b.status.alive)
        alive_count = sum(1 for b in blobs if b.status.alive)
        self.ledger_records.append(LedgerSnapshot(
            step=step,
            e_blobs=e_blobs,
            e_grid=float(grid.energy.sum()),
            e_dissipated=ledger.dissipated,
            e_injected=ledger.injected,
            n_alive=alive_count,
            n_births=n_births,
            n_deaths=n_deaths,
        ))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def energy_over_time(self) -> np.ndarray:
        """Total energy (blobs + grid) per step. Shape: (T,)."""
        return np.array([
            ls.e_blobs + ls.e_grid for ls in self.ledger_records
        ])

    def population_over_time(self) -> np.ndarray:
        """Alive blob count per step. Shape: (T,)."""
        return np.array([ls.n_alive for ls in self.ledger_records])

    def mean_lifespan(self) -> float:
        """Average steps alive at death, computed from blob_records.

        Returns 0.0 if no deaths have been recorded.
        """
        death_ages: list[int] = []
        for step_recs in self.blob_records:
            for rec in step_recs:
                if not rec.alive:
                    death_ages.append(rec.age)
        if not death_ages:
            return 0.0
        return sum(death_ages) / len(death_ages)

    def species_population(self) -> dict[str, np.ndarray]:
        """Per-species population curves. Shape per species: (T,)."""
        # Collect all species names
        all_species: set[str] = set()
        for step_recs in self.blob_records:
            for rec in step_recs:
                all_species.add(rec.species)

        result: dict[str, list[int]] = {sp: [] for sp in sorted(all_species)}
        for step_recs in self.blob_records:
            counts: dict[str, int] = {sp: 0 for sp in all_species}
            for rec in step_recs:
                if rec.alive:
                    counts[rec.species] += 1
            for sp in all_species:
                result[sp].append(counts[sp])

        return {sp: np.array(vals) for sp, vals in result.items()}

    def conservation_residuals(self) -> np.ndarray:
        """Conservation residual per step. Should be ~0. Shape: (T,).

        Residual = E_blobs + E_grid + E_dissipated
                 - E_injected - initial_total.
        """
        initial = (
            self._initial_total
            if self._initial_total is not None
            else 0.0
        )
        return np.array([
            ls.e_blobs + ls.e_grid + ls.e_dissipated - ls.e_injected - initial
            for ls in self.ledger_records
        ])

    def birth_rate(self) -> np.ndarray:
        """Per-step birth rate (births / max(alive, 1)). Shape: (T,)."""
        return np.array([
            ls.n_births / max(ls.n_alive, 1)
            for ls in self.ledger_records
        ])

    def death_rate(self) -> np.ndarray:
        """Per-step death rate (deaths / max(alive, 1)). Shape: (T,)."""
        return np.array([
            ls.n_deaths / max(ls.n_alive, 1)
            for ls in self.ledger_records
        ])

    def to_dataframes(self) -> dict[str, object]:
        """Export to pandas DataFrames.

        Returns dict with keys ``"blobs"`` and ``"ledger"``.
        Pandas is imported lazily.
        """
        import pandas as pd

        # Flatten blob_records into rows
        blob_rows: list[dict] = []
        for step_recs in self.blob_records:
            for rec in step_recs:
                blob_rows.append({
                    "step": rec.step,
                    "blob_id": rec.blob_id,
                    "species": rec.species,
                    "position_r": rec.position[0],
                    "position_c": rec.position[1],
                    "energy": rec.energy,
                    "age": rec.age,
                    "action_taken": rec.action_taken.value,
                    "action_succeeded": rec.action_succeeded,
                    "action_cost": rec.action_cost,
                    "energy_absorbed": rec.energy_absorbed,
                    "alive": rec.alive,
                })

        # Ledger records
        ledger_rows: list[dict] = []
        for ls in self.ledger_records:
            ledger_rows.append({
                "step": ls.step,
                "e_blobs": ls.e_blobs,
                "e_grid": ls.e_grid,
                "e_dissipated": ls.e_dissipated,
                "e_injected": ls.e_injected,
                "n_alive": ls.n_alive,
                "n_births": ls.n_births,
                "n_deaths": ls.n_deaths,
            })

        return {
            "blobs": pd.DataFrame(blob_rows),
            "ledger": pd.DataFrame(ledger_rows),
        }
