"""Generate GIF visualizations for benchmark scenarios.

Produces one GIF per convergence scenario, stored in benchmark/gifs/.
Each GIF shows the simulation evolving toward its convergence condition.

Usage: python3 -m benchmark.generate_gifs
"""
from __future__ import annotations

import time
from pathlib import Path

from blobsim.rendering import render_gif
from blobsim.simulation import Simulation

from benchmark.initial_conditions import (
    ic_carrying_capacity,
    ic_competition,
    ic_decay_multi,
    ic_extinction_boundary,
    ic_fertile_decay,
)

GIF_DIR = Path(__file__).parent / "gifs"
SEED = 42


def _generate(name: str, sim: Simulation, steps: int, **kwargs) -> Path:
    """Run simulation and render GIF."""
    t0 = time.perf_counter()
    result = sim.run(steps)
    t_sim = time.perf_counter() - t0

    out = GIF_DIR / f"{name}.gif"
    t1 = time.perf_counter()
    path = render_gif(result, out, **kwargs)
    t_render = time.perf_counter() - t1

    n_frames = len(result.recorder.blob_records) // kwargs.get("sample_every", 1)
    print(f"  {name}: {n_frames} frames, sim={t_sim:.1f}s, render={t_render:.1f}s -> {path}")
    return path


def main() -> None:
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating GIFs in {GIF_DIR}/\n")

    # SC-1: Decay to extinction (2 species, ~35 steps)
    print("[SC-1] Decay extinction — 2 species consuming finite grid energy")
    sim = Simulation(ic_decay_multi(SEED))
    _generate("sc1_decay_extinction", sim, steps=40, fps=4, cell_pixels=14)

    # SC-4: Boom-bust extinction (grazers on rich grid, ~130 steps)
    print("[SC-4] Boom-bust — population explosion then crash")
    sim = Simulation(ic_fertile_decay(SEED))
    _generate("sc4_boom_bust", sim, steps=50, fps=5, cell_pixels=10)

    # SC-5 sparse: 10 blobs growing toward carrying capacity
    print("[SC-5] Carrying capacity — sparse start (10 blobs)")
    sim = Simulation(ic_carrying_capacity(SEED, count=10))
    _generate("sc5_sparse", sim, steps=500, fps=15, cell_pixels=14, sample_every=5)

    # SC-5 dense: 300 blobs declining toward carrying capacity
    print("[SC-5] Carrying capacity — dense start (300 blobs)")
    sim = Simulation(ic_carrying_capacity(SEED, count=300))
    _generate("sc5_dense", sim, steps=500, fps=15, cell_pixels=14, sample_every=5)

    # C2: Extinction boundary — two regimes of the phase transition
    # r_c ≈ 0.075; show 0.5x (extinction) vs 2.0x (survival)
    print("[C2] Extinction boundary — below critical rate (extinction)")
    sim = Simulation(ic_extinction_boundary(SEED, rate=0.038))
    _generate("c2_extinction", sim, steps=200, fps=10, cell_pixels=14, sample_every=2)

    print("[C2] Extinction boundary — above critical rate (survival)")
    sim = Simulation(ic_extinction_boundary(SEED, rate=0.15))
    _generate("c2_survival", sim, steps=500, fps=12, cell_pixels=14, sample_every=5)

    # C3: Competitive exclusion — grazers vs walkers (dominance by ~step 100)
    print("[C3] Competitive exclusion — grazers outcompeting walkers")
    sim = Simulation(ic_competition(SEED))
    _generate("c3_competition", sim, steps=60, fps=5, cell_pixels=10, sample_every=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
