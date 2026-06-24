"""Cyclic 3-species ouroboros: A->B->C->A (rock-paper-scissors).

Each species is an omnivore (grazes grass, preys opportunistically on the ONE
species it dominates). Cyclic dominance provides the chirality that two
species lack, so the lattice self-organizes into rotating spiral waves -- a
genuine tail-eating ouroboros. Random initial mix (Simulation's default
placement) is the classic spiral-forming initial condition.

Run: .venv/bin/python -m scenarios.cyclic_ouroboros
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from blobsim.config import SimulationConfig, SpeciesConfig  # noqa: E402
from blobsim.environment import RegeneratingEnvironment  # noqa: E402
from blobsim.ledger import ConservationError, conservation_tolerance  # noqa: E402
from blobsim.rendering import render_gif  # noqa: E402
from blobsim.simulation import Simulation  # noqa: E402

from blobsim.species.omnivore import OmnivoreRules  # noqa: E402
from scenarios.discover import occupancy_map  # noqa: E402
from scenarios.spatial_ouroboros import fighter  # noqa: E402

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)
COL = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

# Density of the cyclic pattern world. High food + cheap metabolism push the
# steady-state occupancy to ~85-90% so the rotating domains read clearly (a
# sparse grid hides the structure). These are display knobs for the *qualitative*
# pattern -- they don't affect the benchmark base-cases, which keep their own
# analytically-meaningful rates.
RATE = 2.0
METAB = 0.10
# Collapse demo runs below the spiral wavelength (lambda ~ 33 cells, ~invariant
# to density). At the high display density a few losers cling on near gs=22, so
# the demo uses a grid well below lambda for a clean single-survivor collapse.
COLLAPSE_GS = 18
COLLAPSE_SEED = 2


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
    env = RegeneratingEnvironment(RATE, 5.0, 5.0)
    # A eats B, B eats C, C eats A
    specs = [
        ("A", "B"), ("B", "C"), ("C", "A"),
    ]
    n = (gs * gs) // 12
    species = [
        SpeciesConfig(
            OmnivoreRules,
            fighter(s, prey, thr=8.0, off=2.0, atk=0.1, metab=METAB, max_e=18.0),
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
    render_gif(res, FIGS / "cyclic_ouroboros.gif", fps=12, sample_every=15,
               cell_fill=True, alpha_by_energy=True, cell_pixels=8)
    _orbit(rec, FIGS / "cyclic_orbit.gif")
    return res


def _orbit(rec, path, *, step=10, tail=300, fps=15):
    """Animate the population composition as a point wandering the A/B/C
    simplex (the cyclic rotation traces a loop around the centroid). A line
    plot, not a grid render -- the energy-opacity rule does not apply here.
    """
    spp = rec.species_population()
    A, B, C = (spp.get(k, np.zeros(1)).astype(float) for k in ("A", "B", "C"))
    tot = np.clip(A + B + C, 1, None)
    b, c = B / tot, C / tot
    X = b + 0.5 * c
    Y = (math.sqrt(3) / 2) * c
    T = len(X)
    cmap = colormaps["viridis"]
    pal = {n: colormaps["tab10"](i) for i, n in enumerate(sorted(("A", "B", "C")))}
    tri = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2], [0, 0]])
    frames = []
    for k in range(2, T, step):
        fig = Figure(figsize=(5.0, 4.9), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes((0.02, 0.04, 0.84, 0.92))
        ax.plot(tri[:, 0], tri[:, 1], color="0.55", lw=1.6)
        ax.plot(X[:k], Y[:k], color="0.78", lw=0.9, zorder=1)
        lo = max(0, k - tail)
        ax.scatter(X[lo:k], Y[lo:k], c=np.arange(lo, k), cmap=cmap,
                   vmin=0, vmax=T, s=11, zorder=2)
        ax.scatter([X[k]], [Y[k]], color=cmap(k / T), s=110,
                   edgecolors="black", linewidths=1.0, zorder=3)
        ax.plot(0.5, math.sqrt(3) / 6, marker="+", color="k", ms=14, mew=2, zorder=1)
        for (px, py), lab in [((0, 0), "A"), ((1, 0), "B"), ((0.5, math.sqrt(3) / 2), "C")]:
            ax.annotate(lab, (px, py), textcoords="offset points",
                        xytext=(-16 if px < 0.4 else (16 if px > 0.6 else 0),
                                -20 if py < 0.1 else 14),
                        fontsize=22, color=pal[lab], weight="bold", ha="center")
        ax.text(0.5, -0.02, f"step {k}", transform=ax.transAxes, ha="center",
                fontsize=15, color="0.25")
        ax.set_xlim(-0.10, 1.10)
        ax.set_ylim(-0.12, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba()).copy())
        fig.clear()
    Image.fromarray(frames[0], "RGBA").save(
        path, save_all=True,
        append_images=[Image.fromarray(f, "RGBA") for f in frames[1:]],
        duration=1000 // fps, loop=0, disposal=2)
    return path


def collapse(seed: int = COLLAPSE_SEED, steps: int = 400):
    """Same A->B->C->A omnivores, but on a grid SMALLER than the spiral
    wavelength (~33 cells). Below that critical size the rotating waves cannot
    fit, so cyclic dominance collapses to a single surviving species -- the
    counter-example to coexistence. Renders cyclic_collapse.gif.
    """
    print("\n=== CYCLIC COLLAPSE: grid below the spiral wavelength ===")
    gs = COLLAPSE_GS
    env = RegeneratingEnvironment(RATE, 5.0, 5.0)
    n = (gs * gs) // 12
    species = [
        SpeciesConfig(
            OmnivoreRules,
            fighter(s, prey, thr=8.0, off=2.0, atk=0.1, metab=METAB, max_e=18.0),
            n, 8.0)
        for s, prey in (("A", "B"), ("B", "C"), ("C", "A"))
    ]
    cfg = SimulationConfig(grid_size=gs, environment=env, species=species, seed=seed)
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(steps):
            pass
    except ConservationError as e:
        print(f"  !! ConservationError: {e}")
    res = sim.result()
    spp = res.recorder.species_population()
    A, B, C = (spp.get(k, np.zeros(1)) for k in ("A", "B", "C"))
    surv = sum(1 for v in (A[-1], B[-1], C[-1]) if v > 0)
    print(f"  final A={int(A[-1])} B={int(B[-1])} C={int(C[-1])} "
          f"| survivors={surv}/3 (collapse if <3)")
    render_gif(res, FIGS / "cyclic_collapse.gif", fps=10, sample_every=6,
               cell_fill=True, alpha_by_energy=True, cell_pixels=12)
    return res


def _wavelength(rec, gs, species):
    """Domain wavelength (cells) from the first zero-crossing of the radial
    autocorrelation of a species' occupancy map, averaged over the tail."""
    lams = []
    T = len(rec.blob_records)
    for t in range(max(0, T - 100), T, 20):
        field = occupancy_map(rec.blob_records, t, gs, species).astype(float)
        field = field - field.mean()
        ft = np.fft.fft2(field)
        ac = np.fft.fftshift(np.fft.ifft2(ft * np.conj(ft)).real)
        ac /= ac.max()
        cy, cx = np.array(ac.shape) // 2
        yy, xx = np.indices(ac.shape)
        r = np.hypot(xx - cx, yy - cy).astype(int)
        rad = np.bincount(r.ravel(), ac.ravel()) / np.bincount(r.ravel())
        zero = np.where(rad < 0)[0]
        if len(zero):
            lams.append(2 * zero[0])
    return float(np.mean(lams)) if lams else float("nan")


def five_species(seed: int = 7, steps: int = 1200):
    """Does cyclic dominance generalise past three species? Five omnivores in a
    ring A->B->C->D->E->A (each preys on the next, is preyed on by the prior).
    They still coexist, but the extra species pack the lattice into finer, more
    fragmented domains -- a different (coarser-measured) wavelength than the
    3-cycle. Renders cyclic_five.gif.
    """
    print("\n=== FIVE-SPECIES CYCLE: A->B->C->D->E->A ===")
    gs = 72
    env = RegeneratingEnvironment(RATE, 5.0, 5.0)
    ring = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
    n = (gs * gs) // 20
    species = [
        SpeciesConfig(
            OmnivoreRules,
            fighter(s, prey, thr=8.0, off=2.0, atk=0.1, metab=METAB, max_e=18.0),
            n, 8.0)
        for s, prey in ring
    ]
    cfg = SimulationConfig(grid_size=gs, environment=env, species=species, seed=seed)
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(steps):
            pass
    except ConservationError as e:
        print(f"  !! ConservationError: {e}")
    res = sim.result()
    rec = res.recorder
    spp = rec.species_population()
    fin = {k: int(spp.get(k, np.zeros(1))[-1]) for k in "ABCDE"}
    resid = float(np.abs(rec.conservation_residuals()).max())
    print(f"  final {fin} | all coexist={min(fin.values()) > 0} | maxResid={resid:.1e}")
    print(f"  domain wavelength lambda(A) ~ {_wavelength(rec, gs, 'A'):.0f} cells "
          f"(3-species ~33)")
    render_gif(res, FIGS / "cyclic_five.gif", fps=12, sample_every=16,
               cell_fill=True, alpha_by_energy=True, cell_pixels=7)
    return res


def asymmetric(seed: int = 7, steps: int = 1200):
    """Break the symmetry: make A a stronger predator of B (max_energy 26 vs the
    others' 18) so A wins more of its fights. The cycle is robust -- all three
    still coexist, the rotation just runs a little lopsided. Renders asym_3.gif.
    """
    print("\n=== ASYMMETRIC 3-CYCLE: A overpowered (max_e=26) ===")
    gs = 60
    env = RegeneratingEnvironment(RATE, 5.0, 5.0)
    n = (gs * gs) // 12
    maxe = {"A": 26.0, "B": 18.0, "C": 18.0}
    species = [
        SpeciesConfig(
            OmnivoreRules,
            fighter(s, prey, thr=8.0, off=2.0, atk=0.1, metab=METAB, max_e=maxe[s]),
            n, 8.0)
        for s, prey in (("A", "B"), ("B", "C"), ("C", "A"))
    ]
    cfg = SimulationConfig(grid_size=gs, environment=env, species=species, seed=seed)
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(steps):
            pass
    except ConservationError as e:
        print(f"  !! ConservationError: {e}")
    res = sim.result()
    spp = res.recorder.species_population()
    fin = {k: int(spp.get(k, np.zeros(1))[-1]) for k in "ABC"}
    resid = float(np.abs(res.recorder.conservation_residuals()).max())
    print(f"  final {fin} | all coexist={min(fin.values()) > 0} | maxResid={resid:.1e}")
    render_gif(res, FIGS / "asym_3.gif", fps=12, sample_every=15,
               cell_fill=True, alpha_by_energy=True, cell_pixels=8)
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
    collapse()
    five_species()
    asymmetric()
    print(f"\nfigures in {FIGS}")
