"""Autonomous exploration of blobsim patterns built on the cyclic predation
world. Two questions:

1. Finite-size biodiversity collapse — does a small grid lose species (one wins)
   while a large grid keeps all three (spirals protect diversity)?
2. A single rotating spiral — does seeding the 3 species in 3 angular sectors
   produce one big ouroboros core?

Run: .venv/bin/python -m scenarios.explore
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from blobsim.blob import Blob, BlobStatus
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.conflict import RandomResolver
from blobsim.environment import RegeneratingEnvironment
from blobsim.grid import Grid
from blobsim.ledger import ConservationError, EnergyLedger
from blobsim.observability import StateRecorder
from blobsim.engine import SimulationEngine
from blobsim.rng import create_rng_hierarchy, make_blob_rng
from blobsim.simulation import Simulation, SimulationResult
from blobsim.species.omnivore import OmnivoreRules

from scenarios.spatial_ouroboros import fighter

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

CYCLE = [("A", "B"), ("B", "C"), ("C", "A")]


def cyclic_species(counts):
    return [
        SpeciesConfig(
            OmnivoreRules,
            fighter(s, p, thr=8.0, off=2.0, atk=0.1, metab=0.15, max_e=18.0),
            counts, 8.0)
        for s, p in CYCLE
    ]


def run_random(gs, steps, seed):
    """3-cycle, random initial mix (Simulation default placement)."""
    n = (gs * gs) // 12
    cfg = SimulationConfig(
        grid_size=gs, environment=RegeneratingEnvironment(0.2, 5.0, 5.0),
        species=cyclic_species(n), seed=seed)
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(steps):
            pass
    except ConservationError as e:
        print(f"    !! {e}")
    return sim.result()


def n_alive_by_species(rec):
    spp = rec.species_population()
    return {k: int(v[-1]) for k, v in spp.items()}


# ---------------------------------------------------------------------------
# 1. Finite-size biodiversity collapse
# ---------------------------------------------------------------------------

def finite_size():
    print("\n=== finite-size biodiversity: 3-cycle, random init, 600 steps ===")
    print("grid | survived-all-3 / trials | typical final counts")
    rows = []
    for gs in (16, 24, 32, 48, 64):
        allthree = 0
        last = {}
        for seed in range(5):
            rec = run_random(gs, 600, 100 + seed).recorder
            counts = n_alive_by_species(rec)
            n_surv = sum(1 for v in counts.values() if v > 0)
            if n_surv == 3:
                allthree += 1
            last = counts
        rows.append((gs, allthree))
        print(f"{gs:>4} | {allthree}/5{'':18} | {last}")
    return rows


# ---------------------------------------------------------------------------
# 2. Single rotating spiral via angular-sector seeding
# ---------------------------------------------------------------------------

def build_sectors(gs, env, species_configs, count_each, e0, seed):
    """Seed each species in its own angular wedge around the grid centre."""
    engine_rng, env_rng, pool = create_rng_hierarchy(seed)
    grid = Grid(gs, env.initial_grid(gs))
    place_rng = np.random.default_rng(seed ^ 0xA5A5)
    cx = cy = (gs - 1) / 2.0
    K = len(species_configs)

    # bucket every cell by angular sector
    buckets = {k: [] for k in range(K)}
    for r in range(gs):
        for c in range(gs):
            ang = math.atan2(r - cy, c - cx) % (2 * math.pi)
            buckets[int(ang / (2 * math.pi) * K) % K].append((r, c))

    cfgmap = {sc.attributes.species: sc for sc in species_configs}
    blobs, bid = [], 0
    for k, sc in enumerate(species_configs):
        cells = buckets[k]
        idx = place_rng.choice(len(cells), min(count_each, len(cells)), replace=False)
        for i in idx:
            pos = cells[i]
            blobs.append(Blob(bid, sc.attributes,
                              BlobStatus(energy=e0, age=0, position=pos),
                              sc.rules_factory(), make_blob_rng(pool, bid)))
            grid.place_blob(bid, pos)
            bid += 1

    e_blobs = math.fsum(b.status.energy for b in blobs)
    ledger = EnergyLedger(e_blobs + float(grid.energy.sum()))
    rec = StateRecorder(record_interval=1)
    cfg = SimulationConfig(grid_size=gs, environment=env,
                           species=species_configs, seed=seed)
    engine = SimulationEngine(
        grid=grid, blobs=blobs, environment=env,
        conflict_resolver=RandomResolver(), ledger=ledger,
        engine_rng=engine_rng, env_rng=env_rng, blob_pool_entropy=pool,
        species_configs=cfgmap, recorder=rec)
    return engine, cfg, rec


def single_spiral(gs=60, steps=1500, seed=3):
    print("\n=== single spiral: 3 sectors meeting at centre ===")
    env = RegeneratingEnvironment(0.2, 5.0, 5.0)
    species = cyclic_species((gs * gs) // 12)
    eng, cfg, rec = build_sectors(gs, env, species, (gs * gs) // 9, 8.0, seed)
    try:
        for _ in range(steps):
            eng.step()
    except ConservationError as e:
        print(f"    !! {e}")
    res = SimulationResult(recorder=rec, config=cfg, final_step=steps)
    print("   final:", n_alive_by_species(rec),
          "maxResid=%.1e" % float(np.abs(rec.conservation_residuals()).max()))
    from blobsim.rendering import render_gif
    render_gif(res, FIGS / "single_spiral.gif", fps=12, sample_every=8,
               cell_fill=True, cell_pixels=10)
    return res


if __name__ == "__main__":
    finite_size()
    print("\nfigures in", FIGS)
