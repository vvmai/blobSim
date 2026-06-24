"""Predation comparison cases: how the SAME cyclic A->B->C->A world looks
under different blob 'intelligence'. All scenes share one initial condition
(seed, placement, grid) and differ only in the decision rule / sensing, so the
GIFs are directly comparable side by side.

Two families:
  * rule    -> case_basic, case_energy, case_blind
  * sensing -> sense_basic, sense_energy, sense_full

Renders with the unified cell-fill + energy-opacity style (>=50% energy = full
colour, below fades toward the background).

Run: .venv/bin/python -m scenarios.comparison_cases
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from blobsim.blob import BlobAttributes
from blobsim.config import SimulationConfig, SpeciesConfig
from blobsim.environment import RegeneratingEnvironment
from blobsim.rendering import render_gif
from blobsim.rules import RulesEngine
from blobsim.simulation import Simulation
from blobsim.species.omnivore import OmnivoreRules
from blobsim.types import Action, ActionType, MOORE, SensingLevel

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

PRED = frozenset({ActionType.IDLE, ActionType.MOVE, ActionType.ATTACK,
                  ActionType.REPRODUCE})
CYCLE = [("A", "B"), ("B", "C"), ("C", "A")]


class BlindOmnivore(RulesEngine):
    """Attacks ANY adjacent enemy with no energy check -- so it works even at
    BASIC sensing (which cannot read neighbour energy). The engine still only
    kills when the attacker is stronger; losing attacks just waste attack_cost.
    Otherwise: reproduce when fat, else graze the richest adjacent grass, else
    idle."""

    def decide(self, observation, rng):  # noqa: ARG002 (rng unused, deterministic)
        a = observation.self_attributes
        me = observation.self_status.energy
        adj = a.allowed_directions
        moore = lambda c: abs(c.offset[0]) <= 1 and abs(c.offset[1]) <= 1
        prey = [c for c in observation.neighborhood
                if c.occupied and c.offset in adj and moore(c)
                and c.occupant_species in a.preys_on]
        if prey:
            return Action(type=ActionType.ATTACK,
                          direction=sorted(prey, key=lambda c: c.offset)[0].offset)
        if me >= a.reproduction_threshold and any(
                not c.occupied and c.offset in adj and moore(c)
                for c in observation.neighborhood):
            return Action(type=ActionType.REPRODUCE)
        graze = [c for c in observation.neighborhood
                 if not c.occupied and c.offset in adj and moore(c) and c.energy > 0]
        if graze:
            return Action(type=ActionType.MOVE,
                          direction=max(graze, key=lambda c: c.energy).offset)
        return Action(type=ActionType.IDLE)


# High food + cheap metabolism so the grid fills to ~85-90% and the domains
# read clearly (matches the cyclic pattern world; see cyclic_ouroboros.RATE).
RATE = 2.0
METAB = 0.10


def _attrs(species, prey, sensing, move_cost):
    return BlobAttributes.create(
        species=species, max_energy=18.0, offspring_energy=2.0,
        reproduction_threshold=8.0, base_metabolic_cost=METAB, move_cost=move_cost,
        reproduce_cost=0.0, attack_cost=0.1, observation_radius=1,
        sensing_level=sensing, preys_on=frozenset({prey}),
        allowed_actions=PRED, allowed_directions=MOORE)


def _render(rules, sensing, move_cost, name, *, seed, gs=48, steps=500):
    n = (gs * gs) // 12
    species = [SpeciesConfig(rules, _attrs(s, p, sensing, move_cost), n, 8.0)
               for s, p in CYCLE]
    cfg = SimulationConfig(grid_size=gs,
                           environment=RegeneratingEnvironment(RATE, 5.0, 5.0),
                           species=species, seed=seed)
    sim = Simulation(cfg)
    for _ in sim.iterate(steps):
        pass
    rec = sim.result().recorder
    spp = rec.species_population()
    fin = {k: int(spp.get(k, np.zeros(1))[-1]) for k in ("A", "B", "C")}
    path = render_gif(sim.result(), FIGS / f"{name}.gif", fps=10, sample_every=10,
                      cell_fill=True, alpha_by_energy=True, cell_pixels=8)
    print(f"  {name}: finals={fin}  ({path.stat().st_size // 1024} KB)")


def run(seed: int = 0):
    print("=== rule comparison (basic omnivore / energy omnivore / blind attacker) ===")
    _render(OmnivoreRules, SensingLevel.BASIC, 0.1, "case_basic", seed=seed)
    _render(OmnivoreRules, SensingLevel.ENERGY, 0.1, "case_energy", seed=seed)
    _render(BlindOmnivore, SensingLevel.BASIC, 0.1, "case_blind", seed=seed)

    print("=== sensing comparison (high move_cost=0.6, omnivore) ===")
    _render(OmnivoreRules, SensingLevel.BASIC, 0.6, "sense_basic", seed=seed)
    _render(OmnivoreRules, SensingLevel.ENERGY, 0.6, "sense_energy", seed=seed)
    _render(OmnivoreRules, SensingLevel.FULL, 0.6, "sense_full", seed=seed)


if __name__ == "__main__":
    run()
    print(f"\nfigures in {FIGS}")
