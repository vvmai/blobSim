"""Visual verification of discovery scenarios.

Produces, per scenario: an animated GIF (grid energy + blobs) and a
multi-panel metrics PNG, plus a montage of key spatial frames for
inspection. Outputs to scenarios/figs/.

Run: .venv/bin/python -m scenarios.verify
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blobsim.config import SimulationConfig, SpeciesConfig  # noqa: E402
from blobsim.environment import RegeneratingEnvironment  # noqa: E402
from blobsim.rendering import render_gif  # noqa: E402
from blobsim.simulation import Simulation  # noqa: E402
from blobsim.species.grazer import GrazerRules  # noqa: E402
from blobsim.species.random_walker import RandomWalkerRules  # noqa: E402

from scenarios.discover import attrs, FarsightGrazerRules, FORAGE  # noqa: E402

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

SPCOL = {"grazer": "#d62728", "walker": "#1f77b4"}


def metrics_png(rec, title, path):
    """4-panel: population, blob/grid energy, species fractions, conservation."""
    led = rec.ledger_records
    steps = np.arange(len(led))
    e_blobs = np.array([s.e_blobs for s in led])
    e_grid = np.array([s.e_grid for s in led])
    pop = rec.population_over_time()
    resid = np.abs(rec.conservation_residuals())
    spp = rec.species_population()

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=13, weight="bold")

    ax[0, 0].plot(steps, pop, color="k")
    ax[0, 0].set_title("population")
    ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("n_alive")

    ax[0, 1].plot(steps, e_blobs, label="E_blobs", color="#2ca02c")
    ax[0, 1].plot(steps, e_grid, label="E_grid (grass)", color="#8c564b")
    ax[0, 1].plot(steps, e_blobs + e_grid, label="E_total", color="k", ls="--", lw=1)
    ax[0, 1].set_title("energy partition"); ax[0, 1].set_xlabel("step")
    ax[0, 1].legend(fontsize=8)

    for name, series in sorted(spp.items()):
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(pop > 0, series / pop, 0.0)
        ax[1, 0].plot(steps, frac, label=name, color=SPCOL.get(name))
    ax[1, 0].set_title("species fraction"); ax[1, 0].set_xlabel("step")
    ax[1, 0].set_ylim(-0.02, 1.02); ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogy(steps, np.clip(resid, 1e-16, None), color="#9467bd")
    ax[1, 1].axhline(1e-10, color="r", ls=":", lw=1, label="INV-1 tol 1e-10")
    ax[1, 1].set_title("|conservation residual|"); ax[1, 1].set_xlabel("step")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return float(resid.max())


def frame_montage(rec, steps, title, path):
    """Row of grid-energy heatmaps with blob positions overlaid."""
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    vmax = max((g.max() for g in rec.grid_records if g is not None), default=1.0)
    for ax, t in zip(axes, steps):
        g = rec.grid_records[t]
        ax.imshow(g, cmap="YlGn", vmin=0, vmax=vmax, origin="upper")
        xs, ys, cs = [], [], []
        for r in rec.blob_records[t]:
            if r.alive:
                ys.append(r.position[0]); xs.append(r.position[1])
                cs.append(SPCOL.get(r.species, "k"))
        ax.scatter(xs, ys, s=10, c=cs, edgecolors="white", linewidths=0.2)
        n_alive = sum(1 for r in rec.blob_records[t] if r.alive)
        ax.set_title(f"t={t}  n={n_alive}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + "  (green=grass energy, dots=blobs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def key_steps(rec, k=4):
    pop = rec.population_over_time()
    T = len(pop)
    peak = int(np.argmax(pop))
    return sorted(set([min(5, T - 1), peak, (peak + T) // 2, T - 1]))[:k]


# ---------------------------------------------------------------------------
# Scenario 1: grazer vs walker — competitive exclusion (verifies N3/C3)
# ---------------------------------------------------------------------------

def sc_exclusion():
    print("\n[1] competitive exclusion (grazer vs walker)")
    cfg = SimulationConfig(
        grid_size=30,
        environment=RegeneratingEnvironment(0.1, 5.0, 5.0),
        species=[
            SpeciesConfig(GrazerRules, attrs("grazer"), 25, 1.0),
            SpeciesConfig(RandomWalkerRules, attrs("walker"), 25, 1.0),
        ],
        seed=601,
    )
    res = Simulation(cfg).run(600)
    rec = res.recorder
    r = metrics_png(rec, "Exclusion: grazer vs walker (regen r=0.1)",
                    FIGS / "sc1_exclusion_metrics.png")
    frame_montage(rec, key_steps(rec),
                  "Exclusion", FIGS / "sc1_exclusion_frames.png")
    render_gif(res, FIGS / "sc1_exclusion.gif", fps=12, sample_every=4)
    spp = rec.species_population()
    print(f"    final grazer={int(spp['grazer'][-1])} walker={int(spp['walker'][-1])} "
          f"| max|resid|={r:.1e}")


# ---------------------------------------------------------------------------
# Scenario 2: grazer-only — boom-bust then damped equilibrium
# ---------------------------------------------------------------------------

def sc_boombust():
    print("\n[2] boom-bust -> damped carrying-capacity equilibrium (grazer only)")
    cfg = SimulationConfig(
        grid_size=36,
        environment=RegeneratingEnvironment(0.05, 5.0, 5.0),
        species=[SpeciesConfig(GrazerRules, attrs("grazer", thr=2.0), 50, 2.0)],
        seed=77,
    )
    res = Simulation(cfg).run(700)
    rec = res.recorder
    r = metrics_png(rec, "Boom-bust -> equilibrium (grazer, regen r=0.05)",
                    FIGS / "sc2_boombust_metrics.png")
    frame_montage(rec, key_steps(rec),
                  "Boom-bust", FIGS / "sc2_boombust_frames.png")
    render_gif(res, FIGS / "sc2_boombust.gif", fps=12, sample_every=4)
    pop = rec.population_over_time()
    print(f"    peak={int(pop.max())} final={int(pop[-1])} "
          f"overshoot={pop.max()/max(pop[-1],1):.2f}x | max|resid|={r:.1e}")


# ---------------------------------------------------------------------------
# Scenario 3: N4 redesigned — vision under SCARCITY (food limiting)
# ---------------------------------------------------------------------------

def sc_vision_scarcity():
    print("\n[3] N4 redux: value of vision under scarcity (low regen, no repro)")
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for radius, col in ((1, "#1f77b4"), (3, "#ff7f0e"), (6, "#2ca02c")):
        means = []
        last_rec = rec = None
        for seed in range(5):
            cfg = SimulationConfig(
                grid_size=40,
                environment=RegeneratingEnvironment(0.02, 5.0, 1.0),
                species=[SpeciesConfig(
                    FarsightGrazerRules,
                    attrs("grazer", obs=radius, actions=FORAGE,
                          max_e=20.0, metab=0.15), 1, 8.0)],
                seed=700 + seed,
            )
            rec = Simulation(cfg).run(600).recorder
            # mean energy of the (single) blob while alive
            e = [r.energy for step in rec.blob_records for r in step if r.alive]
            means.append(float(np.mean(e)) if e else 0.0)
            last_rec = rec
        # plot one representative energy trajectory
        traj = [step[0].energy if step and step[0].alive else 0.0
                for step in last_rec.blob_records]
        ax.plot(traj, color=col, label=f"radius={radius} (mean E={np.mean(means):.2f})")
        rows.append((radius, float(np.mean(means))))
        print(f"    radius={radius}  mean blob energy={np.mean(means):.3f}")
    ax.set_title("N4: single forager energy under scarcity (regen r=0.02)")
    ax.set_xlabel("step"); ax.set_ylabel("blob energy"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "sc3_vision_scarcity.png", dpi=90)
    plt.close(fig)
    assert last_rec is not None
    gains = rows[-1][1] - rows[0][1]
    print(f"    -> vision benefit (r=6 vs r=1): {gains:+.3f} mean energy")


if __name__ == "__main__":
    sc_exclusion()
    sc_boombust()
    sc_vision_scarcity()
    print(f"\nfigures in {FIGS}")
