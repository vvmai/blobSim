"""C2: Extinction boundary (phase transition).

Two-phase benchmark: calibration sweeps regeneration rates to empirically
locate the critical rate r_c, then main phase validates the phase transition
at multiples of r_c.

Analytical basis:
    Below a critical regeneration rate r_c, blobs cannot sustain a population:
    per-capita energy intake is insufficient to offset metabolic cost and
    stochastic death. Above r_c, survival is possible. The critical rate
    is a phase transition in population dynamics.

C2a (calibration):
    3 walkers with reproduction on 20x20 grid, sweep across 5 rates
    [0.03, 0.12]. 3 trials per rate, 500 steps. Identifies r_c via
    linear interpolation of P(survival) = 0.5.

C2 (main):
    Validates the transition at 3 rate multiples of r_c (0.5x, 1.0x, 2.0x).
    8 trials per rate, 500 steps.

Pass criteria:
    1. At 0.5 * r_c: P(survival) < 30%
    2. At 2.0 * r_c: P(survival) > 70%

Usage: python3 -m benchmark.c2_extinction_boundary
"""
from __future__ import annotations

import sys

import numpy as np

from blobsim.ledger import ConservationError
from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic_extinction_boundary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Calibration (coarse sweep to locate transition)
CAL_SEED = 7000
CAL_RATES = [0.03, 0.05, 0.07, 0.09, 0.12]
CAL_TRIALS = 3
CAL_STEPS = 500

# Main phase (validate at multiples of r_c)
MAIN_SEED = 7100
RATE_MULTIPLIERS = [0.5, 1.0, 2.0]
N_TRIALS = 8
MAIN_STEPS = 500

# Pass thresholds
P_EXTINCT_THRESHOLD = 0.30
P_SURVIVE_THRESHOLD = 0.70


def _run_trial(seed: int, rate: float, steps: int) -> bool:
    """Run single trial, return True if population survives."""
    config = ic_extinction_boundary(seed, rate=rate)
    sim = Simulation(config)
    try:
        for ws in sim.iterate(steps):
            if ws.n_alive == 0:
                return False
    except ConservationError:
        return False
    return True


def _calibrate() -> tuple[float, list[dict]]:
    """C2a: Coarse sweep to empirically locate r_c (P(survival) = 50%)."""
    cal_data: list[dict] = []

    for rate in CAL_RATES:
        n_survived = sum(
            _run_trial(CAL_SEED + t, rate, CAL_STEPS)
            for t in range(CAL_TRIALS)
        )
        p_survival = n_survived / CAL_TRIALS
        cal_data.append({
            "rate": rate,
            "n_survived": n_survived,
            "n_trials": CAL_TRIALS,
            "p_survival": p_survival,
        })

    # Interpolate to find r_c where P(survival) = 0.5
    rates_arr = np.array([d["rate"] for d in cal_data])
    p_arr = np.array([d["p_survival"] for d in cal_data])
    r_c = float(np.interp(0.5, p_arr, rates_arr))

    return r_c, cal_data


def _run_main(r_c: float) -> tuple[bool, list[dict]]:
    """C2 main: validate phase transition at multiples of r_c."""
    sweep_data: list[dict] = []

    for mult in RATE_MULTIPLIERS:
        rate = r_c * mult
        n_survived = sum(
            _run_trial(MAIN_SEED + int(mult * 100) + t, rate, MAIN_STEPS)
            for t in range(N_TRIALS)
        )
        p_survival = n_survived / N_TRIALS
        sweep_data.append({
            "multiplier": mult,
            "rate": rate,
            "n_survived": n_survived,
            "n_trials": N_TRIALS,
            "p_survival": p_survival,
        })

    # Check pass criteria
    p_by_mult = {d["multiplier"]: d["p_survival"] for d in sweep_data}

    low_rate_ok = p_by_mult[0.5] < P_EXTINCT_THRESHOLD
    high_rate_ok = p_by_mult[2.0] > P_SURVIVE_THRESHOLD

    passed = low_rate_ok and high_rate_ok
    return passed, sweep_data


def run() -> dict:
    """Run C2 benchmark (calibration + main)."""
    r_c, cal_data = _calibrate()
    print(f"    C2a calibration: r_c = {r_c:.4f}")

    passed, sweep_data = _run_main(r_c)
    p_by_mult = {d["multiplier"]: d["p_survival"] for d in sweep_data}

    return {
        "id": "C2",
        "passed": passed,
        "r_c": r_c,
        "detail": {
            "low_rate_extinct": p_by_mult[0.5] < P_EXTINCT_THRESHOLD,
            "high_rate_survives": p_by_mult[2.0] > P_SURVIVE_THRESHOLD,
        },
        "calibration": cal_data,
        "sweep": sweep_data,
    }


if __name__ == "__main__":
    result = run()
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\nC2 Extinction Boundary: {status}")
    print(f"  r_c = {result['r_c']:.4f}")
    print("  Calibration:")
    for d in result["calibration"]:
        print(f"    rate={d['rate']:.3f}: P={d['p_survival']:.0%}")
    print("  Main sweep:")
    for d in result["sweep"]:
        print(f"    {d['multiplier']:.1f}x r_c (rate={d['rate']:.4f}): "
              f"P(survival)={d['p_survival']:.0%} ({d['n_survived']}/{d['n_trials']})")
    print(f"  Detail: {result['detail']}")
    sys.exit(0 if result["passed"] else 1)
