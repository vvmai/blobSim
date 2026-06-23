"""Ouroboros: produce a predator-prey Lotka-Volterra cycle with the new
ATTACK mechanism, then a mutual A<->B yin-yang variant.

Diagnostics: peak counting, predator-lags-prey cross-correlation, conservation.
Visuals: GIF + metrics PNG (predator vs prey, phase portrait, conservation).

Run: .venv/bin/python -m scenarios.ouroboros
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blobsim.blob import BlobAttributes  # noqa: E402
from blobsim.config import SimulationConfig, SpeciesConfig  # noqa: E402
from blobsim.environment import RegeneratingEnvironment  # noqa: E402
from blobsim.ledger import ConservationError  # noqa: E402
from blobsim.rendering import render_gif  # noqa: E402
from blobsim.simulation import Simulation  # noqa: E402
from blobsim.species.grazer import GrazerRules  # noqa: E402
from blobsim.species.predator import PredatorRules  # noqa: E402
from blobsim.types import ActionType, MOORE, SensingLevel  # noqa: E402

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

PREY_ACTS = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE})
PRED_ACTS = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
                       ActionType.REPRODUCE})
COL = {"grazer": "#2ca02c", "predator": "#d62728", "A": "#1f77b4", "B": "#ff7f0e"}


def prey_attrs(species="grazer", *, thr=6.0, off=1.0):
    return BlobAttributes.create(
        species=species, max_energy=15.0, offspring_energy=off,
        reproduction_threshold=thr, base_metabolic_cost=0.1, move_cost=0.1,
        reproduce_cost=0.0, observation_radius=1, allowed_actions=PREY_ACTS,
        allowed_directions=MOORE)


def pred_attrs(species="predator", *, prey, thr=14.0, off=3.0, atk=0.3,
               metab=0.3, obs=2):
    return BlobAttributes.create(
        species=species, max_energy=25.0, offspring_energy=off,
        reproduction_threshold=thr, base_metabolic_cost=metab, move_cost=0.1,
        reproduce_cost=0.0, attack_cost=atk, observation_radius=obs,
        sensing_level=SensingLevel.ENERGY, preys_on=frozenset({prey}),
        allowed_actions=PRED_ACTS, allowed_directions=MOORE)


def run(cfg, steps):
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(steps):
            pass
    except ConservationError as e:
        print(f"    !! ConservationError: {e}")
    rec = sim.result().recorder
    resid = np.abs(rec.conservation_residuals())
    return sim.result(), (float(resid.max()) if len(resid) else 0.0)


# --- diagnostics ----------------------------------------------------------

def count_peaks(series, smooth=11, min_rel=0.10):
    a = np.asarray(series, float)
    if len(a) < 3 * smooth:
        return 0
    s = np.convolve(a, np.ones(smooth) / smooth, mode="valid")
    rng = s.max() - s.min()
    if rng < 1e-9:
        return 0
    peaks = 0
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1]:
            left = s[max(0, i - 25):i].min()
            right = s[i + 1:i + 26].min()
            if s[i] - max(left, right) > min_rel * rng:
                peaks += 1
    return peaks


def lead_lag(prey, pred, max_lag=40):
    """Lag L (>0) maximizing corr(prey[t], pred[t+L]); predator lags prey."""
    a = np.asarray(prey, float)
    b = np.asarray(pred, float)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best_lag, best_c = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.mean(a[:len(a) - lag] * b[lag:]) if lag < len(a) else 0
        else:
            c = np.mean(a[-lag:] * b[:len(b) + lag])
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


def metrics_png(res, prey_sp, pred_sp, title, path):
    rec = res.recorder
    spp = rec.species_population()
    prey = spp.get(prey_sp, np.zeros(1))
    pred = spp.get(pred_sp, np.zeros(1))
    steps = np.arange(len(prey))
    resid = np.abs(rec.conservation_residuals())

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=13, weight="bold")

    ax[0, 0].plot(steps, prey, color=COL.get(prey_sp, "g"), label=f"{prey_sp} (prey)")
    ax[0, 0].plot(steps, pred, color=COL.get(pred_sp, "r"), label=f"{pred_sp} (pred)")
    ax[0, 0].set_title("populations"); ax[0, 0].set_xlabel("step")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(prey, pred, lw=0.6, color="k")
    ax[0, 1].scatter(prey[0], pred[0], c="g", s=30, zorder=3, label="start")
    ax[0, 1].set_title("phase portrait (the ouroboros loop)")
    ax[0, 1].set_xlabel(f"{prey_sp} (prey)"); ax[0, 1].set_ylabel(f"{pred_sp} (pred)")
    ax[0, 1].legend(fontsize=8)

    lag, c = lead_lag(prey[len(prey)//4:], pred[len(pred)//4:])
    ax[1, 0].plot(steps, prey / (prey.max() or 1), color=COL.get(prey_sp, "g"),
                  label="prey (norm)")
    ax[1, 0].plot(steps, pred / (pred.max() or 1), color=COL.get(pred_sp, "r"),
                  label="pred (norm)")
    ax[1, 0].set_title(f"normalized — predator lag={lag} steps (corr={c:.2f})")
    ax[1, 0].set_xlabel("step"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].semilogy(steps, np.clip(resid, 1e-16, None), color="#9467bd")
    ax[1, 1].axhline(1e-10, color="r", ls=":", lw=1, label="INV-1 tol")
    ax[1, 1].set_title("|conservation residual|"); ax[1, 1].set_xlabel("step")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return lag, c


# --- scenario 1: classic predator-prey ------------------------------------

def predator_prey():
    print("\n=== OUROBOROS-1: predator vs grazer-prey (Lotka-Volterra) ===")
    best = None
    for seed in (11, 12, 13, 14, 15):
        cfg = SimulationConfig(
            grid_size=40,
            environment=RegeneratingEnvironment(0.08, 5.0, 5.0),
            species=[
                SpeciesConfig(GrazerRules, prey_attrs(), 200, 4.0),
                SpeciesConfig(PredatorRules, pred_attrs(prey="grazer"), 25, 12.0),
            ],
            seed=seed,
        )
        res, resid = run(cfg, 1500)
        spp = res.recorder.species_population()
        prey, pred = spp.get("grazer", np.zeros(1)), spp.get("predator", np.zeros(1))
        coexist = prey[-1] > 0 and pred[-1] > 0
        pk = count_peaks(prey)
        print(f"  seed={seed} final prey={int(prey[-1])} pred={int(pred[-1])} "
              f"coexist={coexist} prey_peaks={pk} resid={resid:.1e}")
        if coexist and (best is None or pk > best[1]):
            best = (res, pk, seed)
    if best is None:
        print("  -> no coexisting cycle found in sweep")
        return
    res, pk, seed = best
    lag, c = metrics_png(res, "grazer", "predator",
                         f"Predator-prey Lotka-Volterra (seed {seed})",
                         FIGS / "ouroboros_predprey_metrics.png")
    render_gif(res, FIGS / "ouroboros_predprey.gif", fps=12, sample_every=6)
    verdict = ("CYCLE" if pk >= 2 and lag > 0 else
               "coexist, weak/no clear cycle")
    print(f"  -> BEST seed={seed}: prey_peaks={pk}, predator lag={lag} (corr {c:.2f})"
          f"  => {verdict}")


# --- scenario 2: mutual A<->B yin-yang ------------------------------------

def yin_yang():
    print("\n=== OUROBOROS-2: mutual A<->B predation (yin-yang) ===")
    a = pred_attrs("A", prey="B", thr=16.0, off=3.0, atk=0.2, metab=0.15, obs=2)
    b = pred_attrs("B", prey="A", thr=16.0, off=3.0, atk=0.2, metab=0.15, obs=2)
    cfg = SimulationConfig(
        grid_size=40,
        environment=RegeneratingEnvironment(0.1, 5.0, 5.0),
        species=[
            SpeciesConfig(PredatorRules, a, 80, 8.0),
            SpeciesConfig(PredatorRules, b, 80, 8.0),
        ],
        seed=21,
    )
    res, resid = run(cfg, 1200)
    spp = res.recorder.species_population()
    A, B = spp.get("A", np.zeros(1)), spp.get("B", np.zeros(1))
    print(f"  final A={int(A[-1])} B={int(B[-1])} coexist={A[-1]>0 and B[-1]>0} "
          f"A_peaks={count_peaks(A)} resid={resid:.1e}")
    lag, c = metrics_png(res, "A", "B", "Mutual A<->B predation (yin-yang)",
                         FIGS / "ouroboros_yinyang_metrics.png")
    render_gif(res, FIGS / "ouroboros_yinyang.gif", fps=12, sample_every=5)
    print(f"  -> A/B lead-lag={lag} (corr {c:.2f})")


if __name__ == "__main__":
    predator_prey()
    yin_yang()
    print(f"\nfigures in {FIGS}")
