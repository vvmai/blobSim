"""SC-5: Carrying capacity convergence (regenerating environment).

Analytical basis: Density-dependent feedback — more blobs -> fewer regenerating cells ->
less food -> more deaths. Equilibrium: N* ~ rate * G^2 / (rate + effective_cost).

IC: RegeneratingEnvironment(rate=0.1, capacity=5.0), 20x20 grid, random walkers.
  base_metabolic=0.1, move_cost=0.2 -> effective_cost = 0.1 + (8/9)*0.2 ~ 0.278
  N* ~ 0.1 * 400 / (0.1 + 0.278) ~ 106

Two convergence runs per trial:
  - Sparse start: 10 blobs -> should grow toward N*
  - Dense start: 300 blobs -> should decline toward N*

Assertions:
  1. Mean population over last 200 steps of both runs within 30% of each other
  2. Both means within 50% of analytical N*
  3. Neither run goes extinct (population > 0 at end)
"""
from __future__ import annotations

import numpy as np

from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic_carrying_capacity

BASE_SEED = 5000
N_TRIALS = 5
N_STEPS = 2000
TAIL_WINDOW = 200

# Analytical prediction
RATE = 0.1
GRID_SIZE = 20
M = 0.1
MOVE_COST = 0.2
EFFECTIVE_COST = M + (8 / 9) * MOVE_COST
N_STAR = RATE * GRID_SIZE ** 2 / (RATE + EFFECTIVE_COST)

SPARSE_COUNT = 10
DENSE_COUNT = 300

# Tolerances
CONVERGENCE_TOL = 0.30  # sparse/dense means within 30%
ANALYTICAL_TOL = 0.50   # both means within 50% of N*


def _run_trial(seed: int, count: int) -> np.ndarray:
    """Run a single trial, return population_over_time array."""
    config = ic_carrying_capacity(seed, count=count)
    sim = Simulation(config)
    sim.run(N_STEPS)
    return sim.result().recorder.population_over_time()


def run() -> dict:
    trial_results: list[dict] = []

    for i in range(N_TRIALS):
        seed = BASE_SEED + i

        pop_sparse = _run_trial(seed, SPARSE_COUNT)
        pop_dense = _run_trial(seed + 1000, DENSE_COUNT)

        mean_sparse = float(pop_sparse[-TAIL_WINDOW:].mean())
        mean_dense = float(pop_dense[-TAIL_WINDOW:].mean())

        # Check convergence: means within 30% of each other
        avg = (mean_sparse + mean_dense) / 2
        if avg > 0:
            relative_diff = abs(mean_sparse - mean_dense) / avg
        else:
            relative_diff = float("inf")
        converged = relative_diff < CONVERGENCE_TOL

        # Check analytical: both within 50% of N*
        sparse_near = abs(mean_sparse - N_STAR) / N_STAR < ANALYTICAL_TOL
        dense_near = abs(mean_dense - N_STAR) / N_STAR < ANALYTICAL_TOL

        # Check survival
        sparse_alive = int(pop_sparse[-1]) > 0
        dense_alive = int(pop_dense[-1]) > 0

        trial_results.append({
            "seed": seed,
            "mean_sparse": round(mean_sparse, 1),
            "mean_dense": round(mean_dense, 1),
            "relative_diff": round(relative_diff, 3),
            "converged": converged,
            "sparse_near_nstar": sparse_near,
            "dense_near_nstar": dense_near,
            "sparse_alive": sparse_alive,
            "dense_alive": dense_alive,
        })

    all_converged = all(t["converged"] for t in trial_results)
    all_near_nstar = all(
        t["sparse_near_nstar"] and t["dense_near_nstar"]
        for t in trial_results
    )
    all_alive = all(
        t["sparse_alive"] and t["dense_alive"]
        for t in trial_results
    )
    passed = all_converged and all_near_nstar and all_alive

    return {
        "id": "SC5",
        "passed": passed,
        "n_trials": N_TRIALS,
        "n_steps": N_STEPS,
        "n_star": round(N_STAR, 1),
        "convergence_tol": CONVERGENCE_TOL,
        "analytical_tol": ANALYTICAL_TOL,
        "detail": {
            "all_converged": all_converged,
            "all_near_nstar": all_near_nstar,
            "all_alive": all_alive,
        },
        "trials": trial_results,
    }
