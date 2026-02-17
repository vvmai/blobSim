"""User-facing Simulation class and SimulationResult.

Spec reference: tech_spec_mvp.md section 14 (User API).
"""
from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass

from blobsim.blob import Blob, BlobStatus
from blobsim.config import SimulationConfig, validate_config
from blobsim.engine import SimulationEngine
from blobsim.grid import Grid
from blobsim.ledger import EnergyLedger
from blobsim.observability import StateRecorder
from blobsim.rng import create_rng_hierarchy, make_blob_rng
from blobsim.types import WorldState


@dataclass(frozen=True)
class SimulationResult:
    """Observability data and final state after simulation completes."""

    recorder: StateRecorder
    config: SimulationConfig
    final_step: int


class Simulation:
    """Single-use simulation runner.

    Calling ``run()`` or ``iterate()`` a second time raises
    ``RuntimeError``. Runs all requested steps even if population
    reaches 0 (no early termination).
    """

    def __init__(self, config: SimulationConfig) -> None:
        validate_config(config)

        # Create RNG hierarchy
        engine_rng, env_rng, blob_pool_entropy = (
            create_rng_hierarchy(config.seed)
        )

        # Initialize grid from environment
        layers = config.environment.initial_grid(config.grid_size)
        grid = Grid(config.grid_size, layers)

        # Build species_configs dict
        species_configs = {sc.attributes.species: sc for sc in config.species}

        # Place initial blobs at random positions
        total_blobs = sum(sc.count for sc in config.species)
        all_positions = [
            (r, c)
            for r in range(config.grid_size)
            for c in range(config.grid_size)
        ]
        chosen_indices = engine_rng.choice(
            len(all_positions), size=total_blobs, replace=False,
        )
        chosen_positions = [all_positions[i] for i in chosen_indices]

        blobs: list[Blob] = []
        blob_id = 0
        pos_idx = 0
        for sc in config.species:
            for _ in range(sc.count):
                pos = chosen_positions[pos_idx]
                blob = Blob(
                    blob_id=blob_id,
                    attributes=sc.attributes,
                    status=BlobStatus(
                        energy=sc.initial_energy, age=0, position=pos,
                    ),
                    rules=sc.rules_factory(),
                    rng=make_blob_rng(blob_pool_entropy, blob_id),
                )
                blobs.append(blob)
                grid.place_blob(blob_id, pos)
                blob_id += 1
                pos_idx += 1

        # Initialize ledger
        e_blobs = math.fsum(b.status.energy for b in blobs)
        e_grid = float(grid.energy.sum())
        ledger = EnergyLedger(e_blobs + e_grid)

        # Create recorder
        recorder = StateRecorder(record_interval=config.record_interval)

        # Create engine
        self._engine = SimulationEngine(
            grid=grid,
            blobs=blobs,
            environment=config.environment,
            conflict_resolver=config.conflict_resolver,
            ledger=ledger,
            engine_rng=engine_rng,
            env_rng=env_rng,
            blob_pool_entropy=blob_pool_entropy,
            species_configs=species_configs,
            recorder=recorder,
        )
        self._config = config
        self._used = False

    def run(self, steps: int) -> SimulationResult:
        """Run all steps and return result with full observability data."""
        for _ in self.iterate(steps):
            pass
        return self.result()

    def iterate(self, steps: int) -> Generator[WorldState, None, None]:
        """Yield WorldState after each step. Generator interface.

        Raises:
            RuntimeError: If called more than once.
        """
        if self._used:
            raise RuntimeError("Simulation is single-use")
        self._used = True
        for step_num in range(steps):
            self._engine.step()
            ls = self._engine.recorder.ledger_records[-1]
            yield WorldState(
                step=step_num,
                n_alive=ls.n_alive,
                n_births=ls.n_births,
                n_deaths=ls.n_deaths,
                e_blobs=ls.e_blobs,
                e_grid=ls.e_grid,
                e_dissipated=ls.e_dissipated,
                e_injected=ls.e_injected,
            )

    def result(self) -> SimulationResult:
        """Return observability data and final state."""
        return SimulationResult(
            recorder=self._engine.recorder,
            config=self._config,
            final_step=self._engine._step,
        )
