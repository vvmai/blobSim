# Predation Specification

## Table of Contents
1. [Overview](#1-overview)
2. [Data Model](#2-data-model)
3. [Action Flow](#3-action-flow)
4. [CLAIM Phase](#4-claim-phase)
5. [RESOLVE Phase](#5-resolve-phase)
6. [EXECUTE Phase](#6-execute-phase)
7. [Invariants](#7-invariants)
8. [Test Coverage](#8-test-coverage)

---

## 1. Overview

Predation is the mechanism by which one blob kills another and absorbs its
energy. A blob may ATTACK a neighboring cell if that cell's occupant belongs
to a species listed in the attacker's `preys_on` set.

Attack success is determined by **energy comparison** at resolution time
(Phase 5: RESOLVE, before Phase 7: CHARGE applies costs).

---

## 2. Data Model

| Field | Type | Constraint |
|---|---|---|
| `preys_on` | `frozenset[str]` | Non-empty iff ATTACK in `allowed_actions`; cannot include own species |
| `action_costs[ATTACK]` | `float` | >= 0 |
| `allowed_actions` | `frozenset[ActionType]` | ATTACK requires non-empty `preys_on` |

Config errors raised at `BlobAttributes.create` time (B-P1 through B-P4).

---

## 3. Action Flow

```
OBSERVE → DECIDE → VALIDATE → CLAIM → RESOLVE → EXECUTE → CHARGE → ABSORB → DEATH_CHECK
                                                    ↑
                                          energy comparison here
                                          (pre-CHARGE energies)
```

Affordability gate (VALIDATE): if `blob.energy < base_metabolic_cost +
attack_cost`, ATTACK is downgraded to IDLE. No claim is emitted.

---

## 4. CLAIM Phase

- Attacker emits an ATTACK claim into `attack_claims_by_prey[prey_pos]`.
- If the target cell is empty at CLAIM time, no claim is emitted (defensive guard).
- ATTACK claims are kept in a separate channel from MOVE/REPRODUCE claims
  (FLAG-5: `resolve_all` rejects occupied-cell claims).

---

## 5. RESOLVE Phase

### 5.1 Energy-Comparison Winner Selection

For each prey cell (iterated in sorted order for INV-2 determinism):

1. Read `victim_energy = prey.status.energy` (current value, before CHARGE).
2. **Qualifying attackers**: all claimants with `attacker.energy > victim_energy`.
3. If no qualifier exists → all claims `succeeded=False`.
4. Among qualifiers, select the one with maximum energy as winner.
5. Ties among the strongest are broken by `resolver.resolve(tied_claims, engine_rng)`.
6. Winner: `succeeded=True`. All others (qualified or not): `succeeded=False`.

### 5.2 Mutual A↔B Combat

When A attacks B and B attacks A simultaneously:
- Each is resolved independently per its prey cell.
- If `A.energy > B.energy`: A's attack on B succeeds; B's attack on A fails
  (B.energy <= A.energy so B does not qualify as an attacker of A).
- If `A.energy == B.energy`: both attacks fail; both survive.
- No reliance on sorted-cell ordering to determine the winner — energy alone
  decides. Sorted order remains for INV-2 (deterministic RNG consumption).

### 5.3 Chain Predation

When attacks form a chain (A→B→C): a blob killed this step does not execute
its own attack. Cancellation does not cascade past a surviving (freed) blob.
Outcome is determined by energy alone, independent of cell coordinates.

Formally: blob `x` is *killed* iff its unique winning attacker *executes*;
an attacker *executes* iff it is not itself killed. For chain A→B→C→D:
A executes (kills B); B killed → B's attack cancelled → C not killed → C
executes (kills D). Survivors: {A, C}; dead: {B, D}.

---

## 6. EXECUTE Phase

### 6.1 Successful Attacks

ATTACK kills execute before MOVE/REPRODUCE (ATTACK-first ordering, §8.2):

1. Compute `headroom = attacker.max_energy - attacker.status.energy`.
2. `transfer = min(prey.energy, max(0, headroom))`.
3. `dissipate = prey.energy - transfer`.
4. `attacker.energy += transfer`; `ledger.add_dissipated(dissipate)`.
5. Set `prey.energy = 0`, `prey.alive = False`; remove from grid.

### 6.2 Failed Attacks

A failed attack (attacker.energy <= victim.energy, or lost energy tie-break):
- Attacker pays `attack_cost` in the normal CHARGE phase — no special handling needed.
- No energy transfer occurs; victim is unaffected.
- Attacker's `step_data.energy_gained_from_predation` remains 0.

---

## 7. Invariants

| ID | Invariant | Check |
|---|---|---|
| INV-1 | Energy conservation: `E_blobs + E_grid + dissipated - injected == initial_total` | `check_conservation` after every step; residual < `conservation_tolerance` (= `1e-10 + 1e-12 * throughput`, scales with accumulated energy so long runs don't trip on roundoff) |
| INV-2 | Determinism: same seed → identical survivor set | Prey cells sorted; RNG consumed in fixed order |
| INV-3 | One blob per cell | `check_one_per_cell` after every step |

---

## 8. Test Coverage

| Test | Section |
|---|---|
| B-P1..B-P4, `attack_cost`, `preys_on` immutability | §14.1 |
| Single kill: energy transfer, conservation, grid cleanup | §14.2 |
| N predators on one prey: exactly one winner via energy+resolver | §14.3 |
| ATTACK-first: prey MOVE/REPRODUCE cancelled by kill | §14.4 |
| Predator low energy: downgrade to IDLE, starved after zero-gain kill | §14.5 |
| Mutual A↔B: stronger kills weaker deterministically | §14.6 |
| Weak attacker fails, victim survives unchanged | §14.7 |
| Equal-energy mutual attack: both fail, both survive | §14.7 |
| Three attackers: strongest wins, others each paid attack_cost | §14.7 |
| Conservation on failed attack (residual < 1e-10) | §14.7 |
| Stronger A kills B deterministically across two seeds | §14.7 |
| Chain A→B→C: survivors {A,C} coordinate-independent | §5.3 |
| Chain A→B→C→D: survivors {A,C}, dead {B,D} both layouts | §5.3 |
| Chain conservation holds after step | §5.3 |
| Cancelled attacker records action_succeeded=False | §5.3 |
| Mutual A↔B plus B→C tail: C survives (cancellation non-cascading) | §5.3 |
