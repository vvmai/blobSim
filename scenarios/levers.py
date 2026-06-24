"""Lever sweeps in the cyclic 3-species (rock-paper-scissors) world.

Each run is small (grid 40, 400 steps, 2 seeds) so we can test many levers fast.
For each lever value we report the coexistence regime (how many of the 3 species
survive), the final populations, an oscillation measure (CV of total population
over the tail), and spatial clustering (mean occupancy Moran's I).

Run: .venv/bin/python -m scenarios.levers
"""
from __future__ import annotations

import numpy as np

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.environment import RegeneratingEnvironment
from blobsim.ledger import ConservationError
from blobsim.simulation import Simulation
from blobsim.species.omnivore import OmnivoreRules
from blobsim.types import ActionType, MOORE, VON_NEUMANN, SensingLevel

from scenarios.discover import morans_I, occupancy_map

PRED = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
                  ActionType.REPRODUCE})
CYCLE = [("A", "B"), ("B", "C"), ("C", "A")]

BASE = dict(gs=40, regen=0.2, cap=5.0, init=5.0, e0=8.0, steps=400,
            thr=8.0, off=2.0, atk=0.1, metab=0.15, max_e=18.0, move=0.1,
            obs=1, sensing=SensingLevel.ENERGY, dirs=MOORE)


def attrs(species, prey, P):
    return BlobAttributes.create(
        species=species, max_energy=P["max_e"], offspring_energy=P["off"],
        reproduction_threshold=P["thr"], base_metabolic_cost=P["metab"],
        move_cost=P["move"], reproduce_cost=0.0, attack_cost=P["atk"],
        observation_radius=P["obs"], sensing_level=P["sensing"],
        preys_on=frozenset({prey}), allowed_actions=PRED,
        allowed_directions=P["dirs"])


def run_one(P, seed):
    gs = P["gs"]
    n = (gs * gs) // 12
    species = [SpeciesConfig(OmnivoreRules, attrs(s, p, P), n, P["e0"])
               for s, p in CYCLE]
    cfg = SimulationConfig(
        grid_size=gs,
        environment=RegeneratingEnvironment(P["regen"], P["cap"], P["init"]),
        species=species, seed=seed)
    sim = Simulation(cfg)
    try:
        for _ in sim.iterate(P["steps"]):
            pass
    except ConservationError:
        pass
    rec = sim.result().recorder
    spp = rec.species_population()
    finals = {k: int(spp.get(k, np.zeros(1))[-1]) for k in ("A", "B", "C")}
    survivors = sum(1 for v in finals.values() if v > 0)
    total = sum(spp.get(k, np.zeros(len(rec.ledger_records))) for k in ("A", "B", "C"))
    tail = total[-150:]
    cv = float(tail.std() / tail.mean()) if tail.mean() > 0 else 0.0
    # clustering: mean Moran's I of A-occupancy over a few tail steps
    Tn = len(rec.blob_records)
    mi = np.mean([morans_I(occupancy_map(rec.blob_records, t, gs, "A"))
                  for t in range(Tn - 60, Tn, 20)])
    return survivors, finals, cv, float(mi)


def sweep(name, key, values, fmt=str):
    print(f"\n=== lever: {name} ===")
    print(f"{'value':>10} | regime(surv/3) | osc-CV | clusterI | typical finals")
    for v in values:
        P = dict(BASE)
        P[key] = v
        survs, finals, cvs, mis = [], None, [], []
        for seed in (0, 1):
            s, f, cv, mi = run_one(P, 200 + seed)
            survs.append(s); cvs.append(cv); mis.append(mi); finals = f
        regime = np.mean(survs)
        print(f"{fmt(v):>10} | {regime:>12.1f} | {np.mean(cvs):6.2f} | "
              f"{np.mean(mis):+7.3f} | {finals}")


if __name__ == "__main__":
    sweep("mobility (move_cost)", "move", [0.0, 0.05, 0.1, 0.3, 0.6])
    sweep("neighbourhood", "dirs", [MOORE, VON_NEUMANN],
          fmt=lambda d: "Moore" if d is MOORE else "VonNeu")
    sweep("attack_cost", "atk", [0.0, 0.1, 0.5, 1.0, 2.0])
    sweep("sensing", "sensing", [SensingLevel.BASIC, SensingLevel.ENERGY],
          fmt=lambda s: s.value)
    sweep("observation_radius", "obs", [1, 2, 3])
    sweep("regen rate", "regen", [0.05, 0.1, 0.2, 0.4])
    sweep("grass capacity", "cap", [2.0, 5.0, 10.0, 20.0])
    sweep("reproduction_threshold", "thr", [4.0, 8.0, 12.0, 16.0])
    print("\nDONE")
