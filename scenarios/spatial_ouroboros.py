"""Spatial ouroboros: two mutually-predating populations (A eats B, B eats A)
seeded on opposite halves of a toroidal grid. Question: do they form a
circulating, tail-eating interface (an ouroboros), or does one side just win?

Because blobsim.Simulation places blobs at random positions, this module
builds the SimulationEngine directly so A and B can be seeded on opposite
sides. No changes to blobsim.

Run: .venv/bin/python -m scenarios.spatial_ouroboros
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blobsim.blob import Blob, BlobAttributes, BlobStatus  # noqa: E402
from blobsim.config import SimulationConfig, SpeciesConfig  # noqa: E402
from blobsim.environment import RegeneratingEnvironment  # noqa: E402
from blobsim.grid import Grid  # noqa: E402
from blobsim.ledger import EnergyLedger  # noqa: E402
from blobsim.observability import StateRecorder  # noqa: E402
from blobsim.engine import SimulationEngine  # noqa: E402
from blobsim.conflict import RandomResolver  # noqa: E402
from blobsim.rng import create_rng_hierarchy, make_blob_rng  # noqa: E402
from blobsim.simulation import SimulationResult  # noqa: E402
from blobsim.rendering import render_gif  # noqa: E402
from blobsim.species.omnivore import OmnivoreRules  # noqa: E402
from blobsim.types import ActionType, MOORE, SensingLevel  # noqa: E402

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

PRED = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
                  ActionType.REPRODUCE})


def fighter(species, prey, *, thr=10.0, off=2.0, atk=0.1, metab=0.2,
            max_e=18.0, obs=1):
    """A mutual-predation 'fighter': eats grass + reproduces in own territory,
    attacks the enemy species at the interface."""
    return BlobAttributes.create(
        species=species, max_energy=max_e, offspring_energy=off,
        reproduction_threshold=thr, base_metabolic_cost=metab, move_cost=0.1,
        reproduce_cost=0.0, attack_cost=atk, observation_radius=obs,
        sensing_level=SensingLevel.ENERGY, preys_on=frozenset({prey}),
        allowed_actions=PRED, allowed_directions=MOORE)


def build_split(grid_size, env, scA, scB, nA, nB, e0, seed):
    """Build a SimulationEngine with A seeded on the left half, B on the right.

    Returns (engine, config, recorder).
    """
    engine_rng, env_rng, blob_pool_entropy = create_rng_hierarchy(seed)
    grid = Grid(grid_size, env.initial_grid(grid_size))
    place_rng = np.random.default_rng(seed ^ 0x5EED)

    half = grid_size // 2
    left = [(r, c) for r in range(grid_size) for c in range(half)]
    right = [(r, c) for r in range(grid_size) for c in range(half, grid_size)]
    posA = [left[i] for i in place_rng.choice(len(left), nA, replace=False)]
    posB = [right[i] for i in place_rng.choice(len(right), nB, replace=False)]

    species_configs = {scA.attributes.species: scA, scB.attributes.species: scB}
    blobs: list[Blob] = []
    bid = 0
    for sc, positions in ((scA, posA), (scB, posB)):
        for pos in positions:
            blobs.append(Blob(
                blob_id=bid, attributes=sc.attributes,
                status=BlobStatus(energy=e0, age=0, position=pos),
                rules=sc.rules_factory(),
                rng=make_blob_rng(blob_pool_entropy, bid)))
            grid.place_blob(bid, pos)
            bid += 1

    e_blobs = math.fsum(b.status.energy for b in blobs)
    ledger = EnergyLedger(e_blobs + float(grid.energy.sum()))
    recorder = StateRecorder(record_interval=1)
    config = SimulationConfig(grid_size=grid_size, environment=env,
                              species=[scA, scB], seed=seed)
    engine = SimulationEngine(
        grid=grid, blobs=blobs, environment=env,
        conflict_resolver=RandomResolver(), ledger=ledger,
        engine_rng=engine_rng, env_rng=env_rng,
        blob_pool_entropy=blob_pool_entropy,
        species_configs=species_configs, recorder=recorder)
    return engine, config, recorder


# --- circulation diagnostics ----------------------------------------------

def circular_mean_col(records_t, species, gs):
    """Circular mean column (torus angle) of a species' blobs at one step."""
    angs = [2 * math.pi * r.position[1] / gs
            for r in records_t if r.alive and r.species == species]
    if not angs:
        return None
    s = sum(math.sin(a) for a in angs)
    c = sum(math.cos(a) for a in angs)
    return math.atan2(s, c)  # radians in (-pi, pi]


def unwrap_drift(series):
    """Total unwrapped angular drift (radians). |drift| large => rotation."""
    vals = [v for v in series if v is not None]
    if len(vals) < 2:
        return 0.0
    return float(np.unwrap(np.array(vals))[-1] - np.unwrap(np.array(vals))[0])


# --- experiment -----------------------------------------------------------

def run_split(grid_size, env, scA, scB, nA, nB, e0, seed, steps):
    engine, config, rec = build_split(grid_size, env, scA, scB, nA, nB, e0, seed)
    for _ in range(steps):
        engine.step()
    return SimulationResult(recorder=rec, config=config, final_step=steps), rec


def two_species(seed=1, steps=900):
    print("\n=== SPATIAL OUROBOROS: A(left) <-> B(right), mutual predation ===")
    gs = 40
    env = RegeneratingEnvironment(0.2, 5.0, 5.0)
    fa = fighter("A", "B", thr=8.0, off=2.0, atk=0.1, metab=0.15, max_e=18.0)
    fb = fighter("B", "A", thr=8.0, off=2.0, atk=0.1, metab=0.15, max_e=18.0)
    scA = SpeciesConfig(OmnivoreRules, fa, 0, 8.0)
    scB = SpeciesConfig(OmnivoreRules, fb, 0, 8.0)
    n = (gs * gs) // 10  # ~1/5 of each half
    res, rec = run_split(gs, env, scA, scB, n, n, 8.0, seed, steps)

    spp = rec.species_population()
    A, B = spp.get("A", np.zeros(1)), spp.get("B", np.zeros(1))
    resid = float(np.abs(rec.conservation_residuals()).max())
    print(f"  init A={n} B={n} | final A={int(A[-1])} B={int(B[-1])} "
          f"coexist={A[-1] > 0 and B[-1] > 0} | maxResid={resid:.1e}")

    angA = [circular_mean_col(rec.blob_records[t], "A", gs)
            for t in range(len(rec.blob_records))]
    driftA = unwrap_drift(angA)
    print(f"  A centroid angular drift over run: {driftA:+.2f} rad "
          f"({driftA / (2 * math.pi):+.2f} full turns)")

    render_gif(res, FIGS / "spatial_ouroboros_lr.gif", fps=12, sample_every=4)
    _metrics(rec, gs, "A<->B mutual predation (A left, B right)",
             FIGS / "spatial_ouroboros_lr_metrics.png")
    _frames(rec, "A<->B mutual predation",
            FIGS / "spatial_ouroboros_lr_frames.png")
    return driftA


def _metrics(rec, gs, title, path):
    spp = rec.species_population()
    A, B = spp.get("A", np.zeros(1)), spp.get("B", np.zeros(1))
    steps = np.arange(len(A))
    resid = np.abs(rec.conservation_residuals())
    angA = [circular_mean_col(rec.blob_records[t], "A", gs) for t in steps]
    angB = [circular_mean_col(rec.blob_records[t], "B", gs) for t in steps]
    ua = np.unwrap([a if a is not None else 0 for a in angA])
    ub = np.unwrap([a if a is not None else 0 for a in angB])

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=13, weight="bold")
    ax[0, 0].plot(steps, A, color="#1f77b4", label="A"); ax[0, 0].plot(steps, B, color="#ff7f0e", label="B")
    ax[0, 0].set_title("populations"); ax[0, 0].set_xlabel("step"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(steps, ua, color="#1f77b4", label="A centroid angle (unwrapped)")
    ax[0, 1].plot(steps, ub, color="#ff7f0e", label="B centroid angle")
    ax[0, 1].set_title("territory angular position (rotation if it drifts)")
    ax[0, 1].set_xlabel("step"); ax[0, 1].set_ylabel("radians"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(A, B, lw=0.6, color="k"); ax[1, 0].scatter(A[0], B[0], c="g", s=30, zorder=3)
    ax[1, 0].set_title("phase portrait A vs B"); ax[1, 0].set_xlabel("A"); ax[1, 0].set_ylabel("B")
    ax[1, 1].semilogy(steps, np.clip(resid, 1e-16, None), color="#9467bd")
    ax[1, 1].axhline(1e-10, color="r", ls=":", lw=1, label="INV-1 tol")
    ax[1, 1].set_title("|conservation residual|"); ax[1, 1].set_xlabel("step"); ax[1, 1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=90); plt.close(fig)


def _frames(rec, title, path):
    T = len(rec.blob_records)
    steps = [min(3, T - 1), T // 4, T // 2, 3 * T // 4, T - 1]
    col = {"A": "#1f77b4", "B": "#ff7f0e"}
    fig, axes = plt.subplots(1, len(steps), figsize=(3.0 * len(steps), 3.4))
    vmax = max((g.max() for g in rec.grid_records if g is not None), default=1.0)
    for ax, t in zip(axes, steps):
        g = rec.grid_records[t]
        if g is not None:
            ax.imshow(g, cmap="Greys", vmin=0, vmax=vmax, origin="upper", alpha=0.5)
        for sp, c in col.items():
            ys = [r.position[0] for r in rec.blob_records[t] if r.alive and r.species == sp]
            xs = [r.position[1] for r in rec.blob_records[t] if r.alive and r.species == sp]
            ax.scatter(xs, ys, s=8, c=c)
        ax.set_title(f"t={t}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + "  (blue=A, orange=B)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=90); plt.close(fig)


if __name__ == "__main__":
    two_species()
    print(f"\nfigures in {FIGS}")
