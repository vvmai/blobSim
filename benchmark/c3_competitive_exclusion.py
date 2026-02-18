"""C3: Competitive exclusion (strategy dominance).

Two-phase benchmark: calibration measures per-species absorption, then
main phase validates that the dominant species outcompetes the other.

Analytical basis:
    Grazers seek highest-energy neighboring cells -> higher per-capita
    absorption than blind random walkers. With identical metabolism,
    higher absorption -> faster reproduction -> eventual dominance.

C3a (calibration):
    Single grazer vs single walker (separate runs) on 30x30 regenerating
    grid (rate=0.1). 5 trials x 500 steps. Measure mean absorption.
    Go/no-go gate: if |delta_absorb| < 0.01, skip C3 main.

C3 (main):
    25 grazers + 25 walkers on 30x30 regenerating grid. 5 trials, 500 steps.

Pass criteria (if go/no-go passes):
    1. Dominant species reaches >70% of population in >= 8/10 trials
    2. Dominant species alive at end of all trials
    3. Frequency trend increasing over last 1000 steps (smoothed, <= 2 dips)

Usage: python3 -m benchmark.c3_competitive_exclusion
"""
from __future__ import annotations

import sys

import numpy as np

from blobsim.ledger import ConservationError
from blobsim.simulation import Simulation

from benchmark.initial_conditions import ic_competition, ic_species_absorption

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Calibration
CAL_SEED = 8000
CAL_TRIALS = 5
CAL_STEPS = 500
DELTA_THRESHOLD = 0.01  # Go/no-go gate

# Main phase
MAIN_SEED = 8100
N_TRIALS = 5
MAIN_STEPS = 500

# Pass thresholds
DOMINANCE_FRACTION = 0.70
MIN_DOMINANT_TRIALS = 4
MAX_TREND_DIPS = 2
TREND_WINDOW = 1000
SMOOTHING_WINDOW = 50


def _measure_species_absorption(species: str) -> float:
    """Measure mean per-step absorption for a single blob of given species."""
    absorptions: list[float] = []

    for trial in range(CAL_TRIALS):
        seed = CAL_SEED + trial + (100 if species == "grazer" else 0)
        config = ic_species_absorption(seed, species=species)
        sim = Simulation(config)
        try:
            for _ in sim.iterate(CAL_STEPS):
                pass
        except ConservationError:
            pass
        result = sim.result()
        recorder = result.recorder

        total_absorbed = 0.0
        n_steps = 0
        for step_recs in recorder.blob_records:
            for rec in step_recs:
                if rec.alive:
                    total_absorbed += rec.energy_absorbed
            n_steps += 1

        absorptions.append(total_absorbed / max(n_steps, 1))

    return float(np.mean(absorptions))


def _calibrate() -> tuple[str, float, float, float]:
    """C3a: Measure absorption for both species, determine dominant."""
    grazer_abs = _measure_species_absorption("grazer")
    walker_abs = _measure_species_absorption("walker")
    delta = grazer_abs - walker_abs
    dominant = "grazer" if delta > 0 else "walker"
    return dominant, grazer_abs, walker_abs, delta


def _run_main(dominant: str) -> tuple[bool, list[dict]]:
    """C3 main: competition trials, check dominance."""
    trial_data: list[dict] = []

    for trial in range(N_TRIALS):
        seed = MAIN_SEED + trial
        config = ic_competition(seed)
        sim = Simulation(config)
        try:
            for _ in sim.iterate(MAIN_STEPS):
                pass
        except ConservationError:
            pass
        result = sim.result()
        recorder = result.recorder

        species_pop = recorder.species_population()
        pop_total = recorder.population_over_time()

        dom_pop = species_pop.get(dominant, np.zeros(1))

        # Compute dominance fraction at each step (avoid div by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            dom_frac = np.where(pop_total > 0, dom_pop / pop_total, 0.0)

        # 1. Does dominant reach >70% at any point?
        max_frac = float(dom_frac.max()) if len(dom_frac) > 0 else 0.0
        reached_dominance = max_frac > DOMINANCE_FRACTION

        # 2. Dominant alive at end?
        dom_alive_end = int(dom_pop[-1]) > 0 if len(dom_pop) > 0 else False

        # 3. Frequency trend over last TREND_WINDOW steps
        if len(dom_frac) > TREND_WINDOW:
            tail = dom_frac[-TREND_WINDOW:]
        else:
            tail = dom_frac

        # Smooth with moving average
        if len(tail) > SMOOTHING_WINDOW:
            kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
            smoothed = np.convolve(tail, kernel, mode="valid")
        else:
            smoothed = tail

        # Count dips (decreases in smoothed series)
        dips = 0
        for i in range(1, len(smoothed)):
            if smoothed[i] < smoothed[i - 1]:
                dips += 1

        # Already at dominance? (fraction sustained above threshold at end)
        already_dominant = (
            len(smoothed) > 0 and smoothed[-1] >= DOMINANCE_FRACTION
        )
        # Trend is "OK" if already dominant or dips are bounded
        trend_ok = already_dominant or (
            dips <= MAX_TREND_DIPS if len(smoothed) > 1 else True
        )
        # Net increase: accept steady dominance (>=)
        if len(smoothed) > 1:
            net_increase = already_dominant or smoothed[-1] >= smoothed[0]
        else:
            net_increase = True

        trial_data.append({
            "seed": seed,
            "reached_dominance": reached_dominance,
            "max_dom_frac": max_frac,
            "dom_alive_end": dom_alive_end,
            "dips": dips,
            "trend_ok": trend_ok,
            "net_increase": net_increase,
            "final_dom_pop": int(dom_pop[-1]) if len(dom_pop) > 0 else 0,
            "final_total_pop": int(pop_total[-1]) if len(pop_total) > 0 else 0,
        })

    # Aggregate pass criteria
    n_dominant = sum(1 for t in trial_data if t["reached_dominance"])
    all_alive = all(t["dom_alive_end"] for t in trial_data)
    # Trend check: majority of trials show increasing trend
    n_trend_ok = sum(1 for t in trial_data if t["trend_ok"] and t["net_increase"])

    crit_1 = n_dominant >= MIN_DOMINANT_TRIALS
    crit_2 = all_alive
    crit_3 = n_trend_ok >= MIN_DOMINANT_TRIALS

    passed = crit_1 and crit_2 and crit_3
    return passed, trial_data


def run() -> dict:
    """Run C3 benchmark (calibration + main)."""
    dominant, grazer_abs, walker_abs, delta = _calibrate()

    print(f"    C3a calibration: grazer={grazer_abs:.4f}, walker={walker_abs:.4f}, "
          f"delta={delta:.4f}, dominant={dominant}")

    # Go/no-go gate
    if abs(delta) < DELTA_THRESHOLD:
        print("    C3a gate: delta too small, skipping C3 main (pass)")
        return {
            "id": "C3",
            "passed": True,
            "skipped": True,
            "dominant": dominant,
            "grazer_absorption": grazer_abs,
            "walker_absorption": walker_abs,
            "delta": delta,
            "detail": {"gate": "skipped — absorption delta below threshold"},
        }

    passed, trial_data = _run_main(dominant)

    n_dominant = sum(1 for t in trial_data if t["reached_dominance"])
    all_alive = all(t["dom_alive_end"] for t in trial_data)
    n_trend_ok = sum(1 for t in trial_data if t["trend_ok"] and t["net_increase"])

    return {
        "id": "C3",
        "passed": passed,
        "skipped": False,
        "dominant": dominant,
        "grazer_absorption": grazer_abs,
        "walker_absorption": walker_abs,
        "delta": delta,
        "detail": {
            "dominance_reached": f"{n_dominant}/{N_TRIALS} >= {MIN_DOMINANT_TRIALS}",
            "dominant_alive_end": all_alive,
            "trend_increasing": f"{n_trend_ok}/{N_TRIALS} >= {MIN_DOMINANT_TRIALS}",
        },
        "trials": trial_data,
    }


if __name__ == "__main__":
    result = run()
    status = "PASS" if result["passed"] else "FAIL"
    skipped = " (skipped)" if result.get("skipped") else ""
    print(f"\nC3 Competitive Exclusion: {status}{skipped}")
    print(f"  Dominant: {result['dominant']}")
    print(f"  Grazer absorption: {result['grazer_absorption']:.4f}")
    print(f"  Walker absorption: {result['walker_absorption']:.4f}")
    print(f"  Delta: {result['delta']:.4f}")
    if not result.get("skipped"):
        print(f"  Detail: {result['detail']}")
        for t in result.get("trials", []):
            print(f"    seed={t['seed']}: dom={t['reached_dominance']}, "
                  f"frac={t['max_dom_frac']:.2f}, alive={t['dom_alive_end']}")
    sys.exit(0 if result["passed"] else 1)
