# Benchmark Suite

**Source**: `tech_spec_mvp.md` §2, §3, §4, §7, §9

## Contents

- [Purpose](#purpose)
- [Design Principles](#design-principles)
- [Coverage Matrix](#coverage-matrix)
- [Initial Conditions](#initial-conditions)
- [Calibration Benchmarks](#calibration-benchmarks)
- [Convergence Benchmarks](#convergence-benchmarks)
- [Candidate Benchmarks](#candidate-benchmarks)
- [Analytical Reference](#analytical-reference)
- [Traceability](#traceability)
- [File Structure](#file-structure)

## Purpose

Verifies the simulation produces correct emergent behavior: given known inputs, observed outputs match analytically predicted values. The system is treated as a black box.

**Prerequisites**: Integration tests pass (`tests/test_integration_mechanical.py`).

| Layer | Scope | Tests |
| ----- | ----- | ----- |
| Unit/integration | Single-step mechanical correctness | `tests/` |
| Calibration | Single-agent multi-step predictions | B2a |
| Convergence | Population-level stable-state dynamics | SC-1, SC-4, SC-5, SC-6 |

## Design Principles

A benchmark measures an **emergent quantity** requiring multiple interacting subsystems. If predictable from one function's source, it belongs in `tests/`.

1. **First-principles prediction** — derived from theory, never curve-fit to output
2. **Convergent quantity** — equilibrium, asymptotic, or time-averaged (not transient snapshots)
3. **Adequate power** — 10% miscalibration in any subsystem must fail the test
4. **Minimal assumptions** — prefer exact predictions (conservation laws, algebraic identities) over approximations; bound errors when approximations are necessary

## Coverage Matrix

Which analytical properties each benchmark validates. Each row is a testable property; columns indicate which benchmark exercises it.

| Analytical property | B2a | SC-1 | SC-4 | SC-5 | SC-6 |
| ------------------- | --- | ---- | ---- | ---- | ---- |
| Energy conservation | | x | x | | |
| Metabolism (drain rate) | x | x | x | x | x |
| Affordability gate | x | | | | |
| Death boundary | x | x | x | | |
| Reproduction | | x | x | x | |
| Spatial exclusion | | | x | x | |
| Environment: decay | x | x | x | | |
| Environment: regeneration | | | | x | x |
| Extinction (finite energy) | | x | x | | |
| Carrying capacity (mean-field) | | | | x | |
| Energy steady-state (absorption) | | | | | x |
| Multi-species interaction | | x | | | |

## Initial Conditions

All IC factories are in `benchmark/initial_conditions.py`. Each takes `seed: int` and returns `SimulationConfig`.

### IC-1: Decay Grid, Isolated Walker

Single `RandomWalkerRules` blob, 20x20 grid, `DecayEnvironment(initial_cell_energy=0.0)`, `base_metabolic=0.1`, reproduction disabled. Configurable: `move_cost`, `initial_energy`, `max_energy`.

**Used by**: B2a

### IC-2: Decay Grid, Multi-Species

20x20 grid, `DecayEnvironment(initial_cell_energy=2.0)`, 5 walkers + 5 grazers. `base_metabolic=0.1`, `move_cost=0.2`, `reproduction_threshold=1.0`, `offspring_energy=0.5`. Reproduction enabled, zero energy injection.

**Used by**: SC-1

### IC-3: Rich Decay Grid, Fertile Grazers

30x30 grid, `DecayEnvironment(initial_cell_energy=10.0)`, 5 grazers. Same metabolic params as IC-2. Low `reproduction_threshold=1.0` drives aggressive reproduction while grid energy lasts. Zero energy injection.

**Used by**: SC-4

### IC-4: Regenerating Grid, Variable Density

20x20 grid, `RegeneratingEnvironment(rate=0.1, capacity=5.0, initial_cell_energy=5.0)`, random walkers. `count` is a parameter: 10 (sparse) or 300 (dense). Reproduction enabled.

**Used by**: SC-5

### IC-5: Large Regenerating Grid, Single Walker

50x50 grid, `RegeneratingEnvironment(rate=1.0, capacity=5.0, initial_cell_energy=5.0)`, 1 random walker, reproduction disabled (`threshold=inf`).

**Used by**: SC-6

## Calibration Benchmarks

Single-agent dynamics where exact analytical predictions are available.

### B2a: Mean Lifespan

**Validates**: energy drain rate + affordability gate + death boundary

With `move_cost > 0`, the affordability gate (§9 Phase 3) creates two regimes: stochastic (can afford MOVE) then deterministic (forced IDLE). Predicted via Wald's equation with renewal overshoot:

```text
E[tau] = (E_0 - B)/E[c] + B/M + E[R]*(1/E[c] - 1/M)
B = M + move_cost,  E[c] = M + (8/9)*move_cost,  E[R] = E[c²]/(2*E[c])
```

Note: `tau = 1000 - 5*n_moves` (algebraic identity for these parameters), so observed values are always multiples of 5.

**Setup**: IC-1, `move_cost=0.5`, 50 trials. Predicted `E[tau]=186.16`.

**Pass**: `|observed_mean - predicted| < 3.0`

## Convergence Benchmarks

Population-level dynamics where stable-state behavior is predictable from first principles. These test multi-component interactions that calibration benchmarks don't reach: reproduction, spatial exclusion, environment regeneration, and density-dependent feedback.

### SC-1: Guaranteed Extinction (Decay Environment)

**Validates**: finite energy implies extinction — conservation law + metabolism + death + reproduction (reproduction redistributes but cannot create energy)

**Analytical basis**: With `E_injected=0`, each alive blob dissipates `>= base_metabolic_cost` per step. `E_dissipated` is monotonically non-decreasing, total energy finite. Therefore extinction is guaranteed within `T_max = ceil(E_initial / min_base_metabolic_cost)` steps. This holds regardless of reproduction dynamics — energy is conserved through lineage.

**Setup**: IC-2, 10 trials, `E_initial = 810.0`, `T_max = 8100`.

**Pass**:

1. All trials reach `n_alive == 0` within T_max
2. Conservation holds at terminus (`|residual| < 1e-9`)

### SC-4: Reproduction then Extinction (Boom-Bust Conservation)

**Validates**: population explosion + crash dynamics under energy conservation — reproduction drives growth, finite energy drives extinction, conservation holds through the entire lifecycle

**Analytical basis**: Same energy argument as SC-1, but rich grid energy (`initial_cell_energy=10.0` on 30x30 = 9000 total grid energy) creates a population boom before the inevitable crash. Reproduction redistributes energy among offspring but the system total `E_blobs + E_grid + E_dissipated` is invariant at every step.

**What distinguishes this from SC-1**: SC-1 has modest grid energy (800) so population growth is limited. SC-4 creates dramatic boom-bust dynamics with peak populations potentially in the hundreds, testing conservation under high-throughput reproduction and death cascades.

Energy bound on population: at any instant, each alive blob has energy `> 0`, so `peak_population * offspring_energy <= E_initial`.

**Setup**: IC-3, 5 trials, `E_initial = 9005.0`, `pop_bound = 18010`.

**Pass**:

1. Peak population > initial count (reproduction happened)
2. All trials reach extinction
3. `peak_population <= E_initial / offspring_energy`
4. Conservation at every step (`max |residual| < 1e-9`)

### SC-5: Carrying Capacity Convergence

**Validates**: density-dependent equilibrium — environment regeneration + metabolism + reproduction + death + spatial exclusion produce a stable population

**Analytical basis**: Mean-field energy balance at equilibrium. Energy input = energy dissipation:

```text
rate × (G² - N) = N × c_avg
N* ≈ rate × G² / (rate + c_avg)
```

With `rate=0.1`, `G²=400`, `c_avg = M + (8/9)*move_cost = 0.278`:

```text
N* ≈ 0.1 × 400 / (0.1 + 0.278) ≈ 106
```

**Approximation bounds**: Mean-field assumes well-mixed population. Spatial clustering creates depletion halos (blobs deplete nearby cells, reducing absorption for neighbors). Expect `N_observed < N*`; the 50% tolerance accommodates this spatial correction.

**Convergence test design**: Two runs from opposite extremes (10 sparse, 300 dense) must converge to similar equilibrium. This eliminates sensitivity to initial conditions and proves the equilibrium is an attractor.

**Setup**: IC-4, 5 trials × 2 runs (sparse=10, dense=300), 2000 steps each.

**Pass**:

1. Mean population (last 200 steps) of sparse and dense runs within 30% of each other
2. Both means within 50% of analytical N*
3. Neither run goes extinct

### SC-6: Single Blob Energy Steady-State

**Validates**: energy absorption converges to expected rate — regeneration + absorption + metabolism reach equilibrium for a single agent

**Analytical basis**: Single blob on a 50x50 grid with `rate=1.0` (fast regeneration) and `capacity=5.0`. The blob moves randomly, visiting ~2500 unique positions before revisiting. Since `rate=1.0` and the grid is large, every cell the blob visits is at or near capacity. Net energy per step:

```text
absorbed - dissipated ≈ capacity - effective_cost = 5.0 - 0.278 = 4.722
```

Energy climbs rapidly to `max_energy=20.0` and stays there (capped).

**Setup**: IC-5, 10 trials, 500 steps.

**Pass**:

1. Blob alive at step 500
2. Mean energy (last 100 steps) within 1.0 of `max_energy` (20.0)
3. Energy variance (last 100 steps) < 1.0

## Candidate Benchmarks

Analytical handles not yet covered by implemented benchmarks. Listed in order of tractability.

### C2: Extinction Boundary (Phase Transition)

**Analytical handle**: [Extinction boundary](#extinction-boundary)

**Approach**: Sweep `environment.rate`. Identify the critical rate where `P(extinction within T)` transitions from ~1 to <1. Compare to the Galton-Watson criticality prediction (`μ=1` boundary).

**Key decision**: Critical rate depends on per-step absorption, which depends on movement pattern and grid state. May require Monte Carlo estimation of single-agent energy intake as an intermediate calibration step.

### C3: Competitive Exclusion (Strategy Dominance)

**Analytical handle**: [Competitive exclusion](#competitive-exclusion)

**Approach**: Equal initial populations of Grazer and RandomWalker. Predict dominant species from per-capita energy intake differential. Measure species frequency ratio over time.

**Key decision**: Requires estimating `Δ_absorb` analytically or via single-species calibration. Most meaningful test but most assumptions.

## Analytical Reference

Background theory for the testable properties used above. Each involves `>=2` interacting subsystems and produces a numerical prediction.

### Carrying Capacity

In `RegeneratingEnvironment(rate, capacity)`, at population equilibrium `energy_in = energy_out`:

```text
rate × (G² - N) = N × c_avg + Φ_death
```

where `G²` = grid cells, `N` = population, `c_avg` = mean per-step dissipation, `Φ_death` = death-remainder flux. Ignoring `Φ_death`:

```text
N_eq ≈ rate × G² / (rate + c_avg)
```

**Tested by**: SC-5

### Extinction Boundary

Below a critical `environment.rate`, mean offspring per lineage `μ < 1` → certain extinction (Galton-Watson). Above it, survival is possible. The critical rate is a phase transition.

**Tested by**: C2 (candidate)

### Density-Dependent Penalty

Failed claims (§9 Phase 5) pay full action cost with zero benefit. At density `ρ = N/G²`, failure probability `≈ ρ`. Emergent penalty grows quadratically with N, amplifying crashes and delaying recovery.

**Tested by**: SC-4 (implicitly — high density causes failed claims), SC-5 (density feedback)

### Competitive Exclusion

Grazers (energy-seeking) outcompete RandomWalkers (blind) via higher per-capita absorption. Mean-field predicts fixation when `Δ_absorb > 0`.

**Tested by**: C3 (candidate)

### Boom-Bust Oscillation

When reproduction rate >> regeneration rate: overshoot → depletion → die-off → recovery → repeat. Whether convergence is monotone or oscillatory depends on the feedback loop gain.

**Tested by**: SC-4 (boom-bust dynamics observed in trajectory)

## Traceability

| Test | Spec sections | Analytical property |
| ---- | ------------- | ------------------- |
| B2a | §9 (engine), §3 (affordability), §4 (decay env) | Wald's equation for lifespan |
| SC-1 | §4 (decay env), §9 (metabolism, death, reproduction) | Finite energy → extinction |
| SC-4 | §3 (conservation), §4 (decay env), §9 (reproduction, death) | Conservation through boom-bust |
| SC-5 | §4 (regen env), §9 (all phases) | Mean-field carrying capacity |
| SC-6 | §4 (regen env), §9 (absorption, metabolism) | Energy absorption steady-state |

## File Structure

```text
benchmark/
├── __init__.py
├── initial_conditions.py          # IC factories (IC-1 through IC-5)
├── b2a_single_lifespan.py         # Calibration: mean lifespan
├── sc1_decay_extinction.py        # Convergence: guaranteed extinction
├── sc4_reproduction_extinction.py # Convergence: boom-bust extinction
├── sc5_carrying_capacity.py       # Convergence: carrying capacity
├── sc6_energy_equilibrium.py      # Convergence: energy steady-state
├── run_all.py                     # Runner (all benchmarks)
└── results/
    └── benchmark_results.json
```
