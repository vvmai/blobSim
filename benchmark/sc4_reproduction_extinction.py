"""SC-4: Reproduction then extinction (conservation through lineage).

Analytical basis: Same as SC-1 (E_injected=0 -> finite energy -> extinction) but with
rich grid energy -> population booms via reproduction then crashes to zero.
Reproduction redistributes but never creates energy.

IC: DecayEnvironment, 30x30 grid, initial_cell_energy=10.0, 5 grazers,
low reproduction_threshold.

Assertions:
  1. Peak population > initial count (reproduction happened)
  2. n_alive == 0 eventually (extinction)
  3. max_population <= E_initial / offspring_energy (energy bounds population)
  4. Conservation holds at every recorded step (residuals < 1e-9)
"""
from __future__ import annotations

import math

import numpy as np

from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic_fertile_decay

BASE_SEED = 4000
N_TRIALS = 5
OFFSPRING_ENERGY = 0.5
M = 0.1

# E_initial = 5 blobs * 1.0 + 900 cells * 10.0 = 9005.0
GRID_SIZE = 30
N_BLOBS = 5
INITIAL_ENERGY_PER_BLOB = 1.0
CELL_ENERGY = 10.0
E_INITIAL = N_BLOBS * INITIAL_ENERGY_PER_BLOB + GRID_SIZE ** 2 * CELL_ENERGY
POP_BOUND = E_INITIAL / OFFSPRING_ENERGY
T_MAX = math.ceil(E_INITIAL / M)


def run() -> dict:
    trial_results: list[dict] = []

    for i in range(N_TRIALS):
        config = ic_fertile_decay(BASE_SEED + i)
        sim = Simulation(config)

        # Iterate until extinction or T_MAX
        death_step: int | None = None
        for ws in sim.iterate(T_MAX):
            if ws.n_alive == 0:
                death_step = ws.step + 1
                break

        result = sim.result()
        recorder = result.recorder
        pop = recorder.population_over_time()
        residuals = recorder.conservation_residuals()

        peak_pop = int(pop.max())
        went_extinct = death_step is not None
        reproduced = peak_pop > N_BLOBS
        pop_within_bound = peak_pop <= POP_BOUND
        max_residual = float(np.abs(residuals).max())
        conserved = max_residual < 1e-9

        trial_results.append({
            "seed": BASE_SEED + i,
            "peak_pop": peak_pop,
            "extinct": went_extinct,
            "extinction_step": death_step,
            "reproduced": reproduced,
            "pop_within_bound": pop_within_bound,
            "max_residual": max_residual,
            "conserved": conserved,
        })

    all_reproduced = all(t["reproduced"] for t in trial_results)
    all_extinct = all(t["extinct"] for t in trial_results)
    all_bounded = all(t["pop_within_bound"] for t in trial_results)
    all_conserved = all(t["conserved"] for t in trial_results)
    passed = all_reproduced and all_extinct and all_bounded and all_conserved

    return {
        "id": "SC4",
        "passed": passed,
        "n_trials": N_TRIALS,
        "e_initial": E_INITIAL,
        "pop_bound": POP_BOUND,
        "detail": {
            "all_reproduced": all_reproduced,
            "all_extinct": all_extinct,
            "all_bounded": all_bounded,
            "all_conserved": all_conserved,
        },
        "trials": trial_results,
    }
