"""Cyclic 3-species ouroboros: A->B->C->A (rock-paper-scissors).

Each species is an omnivore (grazes grass, preys opportunistically on the ONE
species it dominates). Cyclic dominance provides the chirality that two
species lack, so the lattice self-organizes into rotating spiral waves -- a
genuine tail-eating ouroboros. Random initial mix (Simulation's default
placement) is the classic spiral-forming initial condition.

Run: .venv/bin/python -m scenarios.cyclic_ouroboros
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blobsim.config import SimulationConfig, SpeciesConfig  # noqa: E402
from blobsim.environment import RegeneratingEnvironment  # noqa: E402
from blobsim.ledger import ConservationError, conservation_tolerance  # noqa: E402
from blobsim.rendering import render_gif  # noqa: E402
from blobsim.simulation import Simulation  # noqa: E402

from blobsim.species.omnivore import OmnivoreRules  # noqa: E402
from scenarios.spatial_ouroboros import fighter  # noqa: E402

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)
COL = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


def lead_lag(x, y, max_lag=60):
    a = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-9)
    b = (np.asarray(y, float) - np.mean(y)) / (np.std(y) + 1e-9)
    best_lag, best_c = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.mean(a[:len(a) - lag] * b[lag:]) if lag < len(a) else 0.0
        else:
            c = np.mean(a[-lag:] * b[:len(b) + lag])
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


def run():
    print("\n=== CYCLIC OUROBOROS: A->B->C->A omnivores (rock-paper-scissors) ===")
    gs = 60
    env = RegeneratingEnvironment(0.2, 5.0, 5.0)
    # A eats B, B eats C, C eats A
    specs = [
        ("A", "B"), ("B", "C"), ("C", "A"),
    ]
    n = (gs * gs) // 12  # ~8% each -> ~24% occupancy
    species = [
        SpeciesConfig(
            OmnivoreRules,
            fighter(s, prey, thr=8.0, off=2.0, atk=0.1, metab=0.15, max_e=18.0),
            n, 8.0)
        for s, prey in specs
    ]
    cfg = SimulationConfig(grid_size=gs, environment=env, species=species, seed=7)
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(1500):
            pass
    except ConservationError as e:
        print(f"  !! ConservationError: {e}")
    res = sim.result()
    rec = res.recorder
    spp = rec.species_population()
    A, B, C = (spp.get(k, np.zeros(1)) for k in ("A", "B", "C"))
    resid = float(np.abs(rec.conservation_residuals()).max())
    print(f"  init {n} each | final A={int(A[-1])} B={int(B[-1])} C={int(C[-1])} "
          f"| all coexist={min(A[-1], B[-1], C[-1]) > 0} | maxResid={resid:.1e}")

    # cyclic temporal signature: A should lead B should lead C should lead A
    tail = slice(len(A) // 3, None)
    lab, _ = lead_lag(A[tail], B[tail])
    lbc, _ = lead_lag(B[tail], C[tail])
    lca, _ = lead_lag(C[tail], A[tail])
    print(f"  lead-lag (steps): A->B={lab}  B->C={lbc}  C->A={lca}  "
          f"(consistent sign => cyclic rotation)")

    _metrics(rec, "Cyclic 3-species ouroboros (A->B->C->A)",
             FIGS / "cyclic_ouroboros_metrics.png", (lab, lbc, lca))
    _frames(rec, gs, "Cyclic ouroboros", FIGS / "cyclic_ouroboros_frames.png")
    render_gif(res, FIGS / "cyclic_ouroboros.gif", fps=12, sample_every=8,
               cell_fill=True, cell_pixels=10)
    return res


def _metrics(rec, title, path, lags):
    spp = rec.species_population()
    A, B, C = (spp.get(k, np.zeros(1)) for k in ("A", "B", "C"))
    steps = np.arange(len(A))
    resid = np.abs(rec.conservation_residuals())
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=13, weight="bold")
    for k, s in (("A", A), ("B", B), ("C", C)):
        ax[0, 0].plot(steps, s, color=COL[k], label=k)
    ax[0, 0].set_title("populations (cyclic oscillation if coexisting)")
    ax[0, 0].set_xlabel("step"); ax[0, 0].legend(fontsize=8)
    # simplex-style: fractions
    tot = np.clip(A + B + C, 1, None)
    ax[0, 1].plot(A / tot, B / tot, lw=0.5, color="k")
    ax[0, 1].scatter((A / tot)[0], (B / tot)[0], c="g", s=30, zorder=3, label="start")
    ax[0, 1].set_title("composition orbit (A frac vs B frac)")
    ax[0, 1].set_xlabel("A fraction"); ax[0, 1].set_ylabel("B fraction")
    ax[0, 1].legend(fontsize=8)
    lab, lbc, lca = lags
    ax[1, 0].text(0.5, 0.5,
                  f"lead-lag (steps)\nA->B = {lab}\nB->C = {lbc}\nC->A = {lca}\n\n"
                  "same-sign lags => the three\nchase each other (rotation)",
                  ha="center", va="center", fontsize=12, transform=ax[1, 0].transAxes)
    ax[1, 0].axis("off")
    ax[1, 1].semilogy(steps, np.clip(resid, 1e-16, None), color="#9467bd")
    tol = np.array([
        conservation_tolerance(ls.e_blobs, ls.e_grid, ls.e_dissipated, ls.e_injected)
        for ls in rec.ledger_records
    ])
    ax[1, 1].plot(steps, tol, color="r", ls=":", lw=1, label="INV-1 tol")
    ax[1, 1].set_title("|conservation residual|"); ax[1, 1].set_xlabel("step")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=90); plt.close(fig)


def _frames(rec, gs, title, path):
    T = len(rec.blob_records)
    steps = [min(3, T - 1), T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T - 1]
    fig, axes = plt.subplots(1, len(steps), figsize=(2.7 * len(steps), 3.0))
    for ax, t in zip(axes, steps):
        for sp, c in COL.items():
            ys = [r.position[0] for r in rec.blob_records[t] if r.alive and r.species == sp]
            xs = [r.position[1] for r in rec.blob_records[t] if r.alive and r.species == sp]
            ax.scatter(xs, ys, s=4, c=c)
        ax.set_xlim(0, gs); ax.set_ylim(0, gs); ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"t={t}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + "  (A=blue, B=orange, C=green)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path, dpi=100); plt.close(fig)


if __name__ == "__main__":
    run()
    print(f"\nfigures in {FIGS}")
