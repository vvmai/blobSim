"""SC-1: Guaranteed extinction in decay environment.

Analytical basis: With E_injected=0, each alive blob dissipates >= base_metabolic_cost
per step. E_dissipated is monotonically non-decreasing, total energy finite -> extinction
guaranteed within T_max = ceil(E_initial / min_base_metabolic_cost) steps.

IC: DecayEnvironment, 20x20 grid, 2 species (5 walkers + 5 grazers),
initial_cell_energy=2.0, reproduction enabled.

Assertions:
  1. n_alive == 0 within T_max steps (all 10 trials)
  2. Conservation holds at terminus: residual < 1e-9
"""
from __future__ import annotations

import math

import numpy as np

from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic_decay_multi

BASE_SEED = 2000
N_TRIALS = 10
M = 0.1  # min base_metabolic_cost across both species

# E_initial = 10 blobs * 1.0 + 400 cells * 2.0 = 810.0
GRID_SIZE = 20
N_BLOBS = 10
INITIAL_ENERGY = 1.0
CELL_ENERGY = 2.0
E_INITIAL = N_BLOBS * INITIAL_ENERGY + GRID_SIZE ** 2 * CELL_ENERGY
T_MAX = math.ceil(E_INITIAL / M)


def run() -> dict:
    extinctions: list[int] = []
    conservation_ok: list[bool] = []

    for i in range(N_TRIALS):
        config = ic_decay_multi(BASE_SEED + i)
        sim = Simulation(config)

        death_step: int | None = None
        for ws in sim.iterate(T_MAX):
            if ws.n_alive == 0:
                death_step = ws.step + 1
                break

        result = sim.result()
        recorder = result.recorder

        # Check conservation at final recorded step
        residuals = recorder.conservation_residuals()
        final_residual = float(np.abs(residuals[-1]))
        conservation_ok.append(final_residual < 1e-9)

        if death_step is not None:
            extinctions.append(death_step)

    all_extinct = len(extinctions) == N_TRIALS
    all_conserved = all(conservation_ok)
    passed = all_extinct and all_conserved

    return {
        "id": "SC1",
        "passed": passed,
        "n_trials": N_TRIALS,
        "extinctions": len(extinctions),
        "t_max": T_MAX,
        "extinction_steps": sorted(extinctions) if extinctions else [],
        "mean_extinction": round(sum(extinctions) / len(extinctions), 1) if extinctions else None,
        "conservation_ok": sum(conservation_ok),
        "detail": {
            "all_extinct": all_extinct,
            "all_conserved": all_conserved,
        },
    }
