"""SC-6: Single blob energy steady-state (regenerating environment).

Analytical basis: Single blob on large grid with capacity >> effective_cost.
Blob rarely revisits cells -> always finds full cells -> absorbs capacity per step.
Energy converges to max_energy.

IC: RegeneratingEnvironment(rate=1.0, capacity=5.0), 50x50 grid, single random walker,
reproduction disabled (threshold=inf).
  base_metabolic=0.1, move_cost=0.2, effective_cost ~ 0.278
  capacity (5.0) >> effective_cost (0.278) -> energy -> max_energy

Assertions:
  1. Blob alive at step 500
  2. Blob energy ~ max_energy (within 1.0) averaged over last 100 steps
  3. Energy variance over last 100 steps < 1.0 (stable, not fluctuating)
"""
from __future__ import annotations

import numpy as np

from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic_single_regen

BASE_SEED = 6000
N_TRIALS = 10
N_STEPS = 500
TAIL_WINDOW = 100

MAX_ENERGY = 20.0
ENERGY_TOL = 1.0
VARIANCE_TOL = 1.0


def run() -> dict:
    trial_results: list[dict] = []

    for i in range(N_TRIALS):
        config = ic_single_regen(BASE_SEED + i)
        sim = Simulation(config)
        sim.run(N_STEPS)

        recorder = sim.result().recorder

        # Extract per-step energy for the single blob (blob_id=0)
        # blob_records is list[list[BlobRecord]] — one list per step
        energies: list[float] = []
        alive_at_end = False
        for step_records in recorder.blob_records:
            for br in step_records:
                if br.blob_id == 0:
                    energies.append(br.energy)
                    if br.alive:
                        alive_at_end = True
                    break

        tail = np.array(energies[-TAIL_WINDOW:]) if len(energies) >= TAIL_WINDOW else np.array(energies)
        mean_energy = float(tail.mean()) if len(tail) > 0 else 0.0
        energy_var = float(tail.var()) if len(tail) > 0 else float("inf")

        # Check blob alive at final step
        final_pop = recorder.population_over_time()
        alive_at_end = int(final_pop[-1]) > 0

        near_max = abs(mean_energy - MAX_ENERGY) < ENERGY_TOL
        stable = energy_var < VARIANCE_TOL

        trial_results.append({
            "seed": BASE_SEED + i,
            "alive_at_end": alive_at_end,
            "mean_energy_tail": round(mean_energy, 2),
            "energy_var_tail": round(energy_var, 4),
            "near_max": near_max,
            "stable": stable,
        })

    all_alive = all(t["alive_at_end"] for t in trial_results)
    all_near_max = all(t["near_max"] for t in trial_results)
    all_stable = all(t["stable"] for t in trial_results)
    passed = all_alive and all_near_max and all_stable

    return {
        "id": "SC6",
        "passed": passed,
        "n_trials": N_TRIALS,
        "n_steps": N_STEPS,
        "max_energy": MAX_ENERGY,
        "energy_tol": ENERGY_TOL,
        "variance_tol": VARIANCE_TOL,
        "detail": {
            "all_alive": all_alive,
            "all_near_max": all_near_max,
            "all_stable": all_stable,
        },
        "trials": trial_results,
    }
