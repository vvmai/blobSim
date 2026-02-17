"""B2a: Stochastic mean lifespan matches theoretical prediction.

Calibrates: energy drain rate, affordability gate, death boundary.

Two-phase model:
  Phase 1 (stochastic): energy >= B, random move/idle at E[c] per step.
  Phase 2 (deterministic): energy < B, forced IDLE at M per step.

  E[tau] = (E_0 - B)/E[c] + B/M + E[R]*(1/E[c] - 1/M)
  B = M + move_cost, E[c] = M + (8/9)*move_cost, E[R] = E[c^2]/(2*E[c])

Note: tau = 1000 - 5*n_moves (algebraic identity), so observed values
are always multiples of 5. No parametric test needed.
"""
from __future__ import annotations

from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic1

BASE_SEED = 1000
N_TRIALS = 50
MOVE_COST = 0.5
E_0 = 100.0
MAX_ENERGY = 200.0
M = 0.1  # base_metabolic

# Analytical prediction
B = M + MOVE_COST
E_C = M + (8 / 9) * MOVE_COST
E_C2 = (1 / 9) * M**2 + (8 / 9) * (M + MOVE_COST) ** 2
E_R = E_C2 / (2 * E_C)
PREDICTED_TAU = (E_0 - B) / E_C + B / M + E_R * (1 / E_C - 1 / M)

TOLERANCE = 3.0
MAX_STEPS = int(PREDICTED_TAU) + 50


def run() -> dict:
    taus: list[int] = []

    for i in range(N_TRIALS):
        config = ic1(
            BASE_SEED + i,
            move_cost=MOVE_COST,
            initial_energy=E_0,
            max_energy=MAX_ENERGY,
        )
        sim = Simulation(config)

        death_step = MAX_STEPS
        for ws in sim.iterate(MAX_STEPS):
            if ws.n_alive == 0:
                death_step = ws.step + 1
                break
        taus.append(death_step)

    mean = sum(taus) / len(taus)
    passed = abs(mean - PREDICTED_TAU) < TOLERANCE

    return {
        "id": "B2a",
        "passed": passed,
        "predicted": round(PREDICTED_TAU, 2),
        "observed_mean": round(mean, 2),
        "observed_values": sorted(set(taus)),
        "tolerance": TOLERANCE,
    }
