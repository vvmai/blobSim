"""Run calibration and convergence benchmarks.

Usage: python -m benchmark.run_all
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from benchmark.b2a_single_lifespan import run as run_b2a
from benchmark.c2_extinction_boundary import run as run_c2
from benchmark.c3_competitive_exclusion import run as run_c3
from benchmark.sc1_decay_extinction import run as run_sc1
from benchmark.sc4_reproduction_extinction import run as run_sc4
from benchmark.sc5_carrying_capacity import run as run_sc5
from benchmark.sc6_energy_equilibrium import run as run_sc6

BENCHMARKS: list[tuple[str, Callable[[], dict]]] = [
    # Calibration
    ("B2a", run_b2a),
    # Convergence
    ("SC1", run_sc1),
    ("SC4", run_sc4),
    ("SC5", run_sc5),
    ("SC6", run_sc6),
    ("C2", run_c2),
    ("C3", run_c3),
]

RESULTS_PATH = Path("benchmark/results/benchmark_results.json")


def main() -> int:
    results: list[dict] = []
    total_start = time.perf_counter()

    for name, run_fn in BENCHMARKS:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = run_fn()
        elapsed = time.perf_counter() - t0
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)

    total = time.perf_counter() - total_start
    all_passed = all(r["passed"] for r in results)

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "results": results,
    }, indent=2))

    n_pass = sum(1 for r in results if r["passed"])
    status = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"\n  {n_pass}/{len(results)} ({total:.1f}s) — {status}")
    print(f"  Results: {RESULTS_PATH}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
