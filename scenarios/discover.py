"""Scenario discovery harness — tests theses A3, N2, N3, N4.

Pure-numpy diagnostics (sparklines, Moran's I); no plotting dependency.
Run: .venv/bin/python -m scenarios.discover
"""
from __future__ import annotations

import numpy as np

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.environment import RegeneratingEnvironment
from blobsim.ledger import ConservationError
from blobsim.rules import RulesEngine
from blobsim.simulation import Simulation
from blobsim.species.grazer import GrazerRules
from blobsim.species.random_walker import RandomWalkerRules
from blobsim.types import Action, ActionType, MOORE

REPRO = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.REPRODUCE})
FORAGE = frozenset({ActionType.IDLE, ActionType.MOVE})


def attrs(species, *, max_e=20.0, off=0.5, thr=1.0, metab=0.1, move=0.2,
          obs=1, actions=REPRO):
    return BlobAttributes.create(
        species=species, max_energy=max_e, offspring_energy=off,
        reproduction_threshold=thr, base_metabolic_cost=metab,
        move_cost=move, reproduce_cost=0.0, observation_radius=obs,
        allowed_actions=actions, allowed_directions=MOORE,
    )


def run(config, steps):
    sim = Simulation(config)
    max_resid = 0.0
    try:
        for _ in sim.iterate(steps):
            pass
    except ConservationError as e:
        print(f"    !! ConservationError: {e}")
    rec = sim.result().recorder
    r = rec.conservation_residuals()
    if len(r):
        max_resid = float(np.abs(r).max())
    return rec, max_resid


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def spark(series, width=58):
    a = np.asarray(series, dtype=float)
    if len(a) == 0:
        return ""
    if len(a) > width:
        a = a[np.linspace(0, len(a) - 1, width).astype(int)]
    lo, hi = float(a.min()), float(a.max())
    ch = "▁▂▃▄▅▆▇█"
    if hi - lo < 1e-12:
        return ch[0] * len(a)
    return "".join(ch[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in a)


def occupancy_map(blob_records, step, gs, species=None):
    m = np.zeros((gs, gs), dtype=bool)
    for rec in blob_records[step]:
        if not rec.alive:
            continue
        if species is not None and rec.species != species:
            continue
        r, c = rec.position
        m[r, c] = True
    return m


def morans_I(field):
    """Toroidal 4-neighbour spatial autocorrelation. >0 clustered, ~0 random."""
    x = field.astype(float)
    z = x - x.mean()
    denom = (z * z).sum()
    if denom < 1e-12:
        return 0.0
    num = sum((z * np.roll(z, s, axis=ax)).sum()
              for ax in (0, 1) for s in (1, -1))
    return float((x.size / (4 * x.size)) * (num / denom))


# ---------------------------------------------------------------------------
# Farsight grazer for N4: actually USES observation_radius (stock GrazerRules
# only ever consults its 8 Moore neighbours, so radius is inert for it).
# Picks the richest unoccupied cell within the full radius, steps one Moore
# cell toward it.
# ---------------------------------------------------------------------------

class FarsightGrazerRules(RulesEngine):
    """Sound gradient-follower that actually USES observation_radius.

    Priority: (1) reproduce if fat with an empty Moore neighbor; (2) eat the
    best ADJACENT unoccupied cell with energy>0 (never forgo reachable food);
    (3) only if no adjacent food, path one step toward the best distant cell.

    NOTE: an earlier naive version skipped step (2) — it chased the global max
    in radius and starved en route, producing a spurious "vision is harmful"
    cliff. That was a rule bug, not a sim property. See DISCOVERY.md N4.
    """

    def decide(self, observation, rng):
        a = observation.self_attributes
        s = observation.self_status
        adj_dir = a.allowed_directions

        if (ActionType.REPRODUCE in a.allowed_actions
                and s.energy >= a.reproduction_threshold):
            if any(not c.occupied and c.offset in adj_dir
                   for c in observation.neighborhood
                   if abs(c.offset[0]) <= 1 and abs(c.offset[1]) <= 1):
                return Action(type=ActionType.REPRODUCE)

        adjacent = [c for c in observation.neighborhood
                    if not c.occupied and c.offset in adj_dir
                    and abs(c.offset[0]) <= 1 and abs(c.offset[1]) <= 1
                    and c.energy > 0]
        if adjacent:
            best = max(adjacent, key=lambda c: c.energy)
            return Action(type=ActionType.MOVE, direction=best.offset)

        far = [c for c in observation.neighborhood
               if not c.occupied and c.energy > 0]
        if not far:
            return Action(type=ActionType.IDLE)
        b = max(far, key=lambda c: c.energy)
        step = (int(np.sign(b.offset[0])), int(np.sign(b.offset[1])))
        if step == (0, 0) or step not in adj_dir:
            return Action(type=ActionType.IDLE)
        return Action(type=ActionType.MOVE, direction=step)


# ===========================================================================
# A3: spatial self-organisation — do grazers cluster, walkers stay uniform?
# ===========================================================================

def exp_a3():
    print("\n=== A3: spatial clustering (grazer fronts vs walker diffusion) ===")
    out = {}
    for species, rules in (("grazer", GrazerRules), ("walker", RandomWalkerRules)):
        cfg = SimulationConfig(
            grid_size=36,
            environment=RegeneratingEnvironment(0.05, 5.0, 5.0),
            species=[SpeciesConfig(rules, attrs(species, thr=2.0), 50, 2.0)],
            seed=77,
        )
        rec, resid = run(cfg, 700)
        gs = 36
        T = len(rec.blob_records)
        occ_I, en_I = [], []
        for t in range(T - 200, T, 20):
            occ_I.append(morans_I(occupancy_map(rec.blob_records, t, gs, species)))
            g = rec.grid_records[t]
            if g is not None:
                en_I.append(morans_I(g))
        oi, ei = float(np.mean(occ_I)), float(np.mean(en_I))
        out[species] = (oi, ei)
        print(f"  {species:<7} occ Moran I={oi:+.3f}  energy-field Moran I={ei:+.3f}"
              f"   (resid {resid:.1e})")
        print(f"    pop: {spark(rec.population_over_time())}")
    verdict = "REPRODUCED" if out["grazer"][0] > out["walker"][0] + 0.05 else "NOT shown"
    print(f"  -> grazer more clustered than walker? {verdict}")
    return out


# ===========================================================================
# N2: prudent-predator / tragedy of the commons (interior optimum in threshold)
# ===========================================================================

def exp_n2():
    print("\n=== N2: prudent-predator optimum (sweep reproduction_threshold) ===")
    rows = []
    for thr in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        tails, exts = [], 0
        for seed in range(4):
            cfg = SimulationConfig(
                grid_size=30,
                environment=RegeneratingEnvironment(0.1, 5.0, 5.0),
                species=[SpeciesConfig(
                    GrazerRules, attrs("grazer", thr=thr, off=0.5), 20, 3.0)],
                seed=300 + seed,
            )
            rec, _ = run(cfg, 800)
            pop = rec.population_over_time()
            tails.append(float(pop[-200:].mean()))
            if int(pop[-1]) == 0:
                exts += 1
        mt = float(np.mean(tails))
        rows.append((thr, mt, exts))
        print(f"  thr={thr:<5} mean_tail_pop={mt:6.1f}  extinct={exts}/4")
    peak = max(rows, key=lambda r: r[1])
    interior = rows[0][1] < peak[1] and rows[-1][1] < peak[1]
    print(f"  -> peak at thr={peak[0]} (pop {peak[1]:.1f}); "
          f"interior optimum? {'YES (hump)' if interior else 'NO (monotone)'}")
    return rows


# ===========================================================================
# N3: coexistence pocket — any (rate,density) where both species persist?
# ===========================================================================

def exp_n3():
    print("\n=== N3: coexistence pocket (grazer + walker) ===")
    found = []
    for rate in (0.05, 0.15, 0.4):
        for count in (15, 40):
            fracs, both_alive = [], 0
            for seed in range(3):
                cfg = SimulationConfig(
                    grid_size=30,
                    environment=RegeneratingEnvironment(rate, 5.0, 5.0),
                    species=[
                        SpeciesConfig(GrazerRules, attrs("grazer"), count, 1.0),
                        SpeciesConfig(RandomWalkerRules, attrs("walker"), count, 1.0),
                    ],
                    seed=600 + seed,
                )
                rec, _ = run(cfg, 800)
                sp = rec.species_population()
                g = sp.get("grazer", np.zeros(1))
                w = sp.get("walker", np.zeros(1))
                tail_g, tail_w = g[-150:], w[-150:]
                if tail_g.min() > 0 and tail_w.min() > 0:
                    both_alive += 1
                tot = g[-1] + w[-1]
                fracs.append(g[-1] / tot if tot > 0 else 0.5)
            gf = float(np.mean(fracs))
            coex = both_alive == 3 and 0.2 <= gf <= 0.8
            tag = "  <-- COEXIST" if coex else ""
            print(f"  rate={rate:<5} count={count:<3} grazer_frac={gf:.2f} "
                  f"both_alive_allseeds={both_alive}/3{tag}")
            if coex:
                found.append((rate, count, gf))
    print(f"  -> coexistence pocket(s): {found if found else 'NONE (exclusion wins)'}")
    return found


# ===========================================================================
# N4: diminishing value of vision (farsight grazer, sweep observation_radius)
# ===========================================================================

def exp_n4():
    """Value of vision UNDER COMPETITION.

    A single forager is never food-limited (it pins at max_energy regardless
    of radius), so vision can only matter when foragers compete for scarce,
    locally-depleted food. 80 farsight grazers, no reproduction, scarce regen.
    """
    print("\n=== N4: value of vision under competition (80 foragers, scarce) ===")
    rows = []
    for radius in (1, 2, 3, 6):
        surv, meanE = [], []
        for seed in range(5):
            cfg = SimulationConfig(
                grid_size=30,
                environment=RegeneratingEnvironment(0.03, 5.0, 1.0),
                species=[SpeciesConfig(
                    FarsightGrazerRules,
                    attrs("grazer", obs=radius, actions=FORAGE,
                          max_e=10.0, metab=0.15), 80, 5.0)],
                seed=800 + seed,
            )
            rec, _ = run(cfg, 400)
            pop = rec.population_over_time()
            surv.append(pop[-1] / 80.0)
            e = [r.energy for r in rec.blob_records[-1] if r.alive]
            meanE.append(float(np.mean(e)) if e else 0.0)
        s, m = float(np.mean(surv)), float(np.mean(meanE))
        rows.append((radius, s, m))
        print(f"  radius={radius}  survival={s:.2f}  mean_final_energy={m:.2f}")
    survivals = [r[1] for r in rows]
    spread = max(survivals) - min(survivals)
    if spread < 0.05:
        verdict = "NEUTRAL — vision inert (sound rule eats best adjacent first)"
    elif survivals[0] == max(survivals):
        verdict = "vision HARMFUL (check rule for chasing-distant-maxima artifact)"
    else:
        verdict = "vision helps"
    print(f"  -> survival spread={spread:.2f}: {verdict}")
    return rows


if __name__ == "__main__":
    exp_a3()
    exp_n2()
    exp_n3()
    exp_n4()
    print("\nDONE")
