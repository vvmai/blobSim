# BlobSim

Cellular automaton on a 2D toroidal grid. Entities ("blobs") carry individual energy, choose actions via pluggable rules engines, and compete for grid resources. The environment evolves independently through pluggable dynamics. All energy flows are tracked by a central ledger enforcing conservation.

## Table of Contents

- [BlobSim](#blobsim)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
    - [Simulation loop (per step)](#simulation-loop-per-step)
  - [Core Invariants](#core-invariants)
  - [Module Map](#module-map)
  - [Extending](#extending)
    - [Add a new species](#add-a-new-species)
    - [Add a new environment](#add-a-new-environment)
  - [Testing \& Benchmarks](#testing--benchmarks)
  - [Commands](#commands)

## Architecture

Custom NumPy implementation — no ABM framework. Key design choices:

- **Energy ledger** — compensated summation (`math.fsum`) prevents float drift; conservation checked every step
- **Seeded RNG hierarchy** — deterministic: same seed → identical state at every timestep
- **Pluggable rules** — species behavior defined by `RulesEngine.decide(observation, rng) -> Action`
- **Pluggable environments** — `DecayEnvironment`, `RegeneratingEnvironment`, or custom `EnvironmentFn`
- **Pluggable conflict resolution** — default `RandomResolver`; swap via `SimulationConfig`

### Simulation loop (per step)

```
OBSERVE → DECIDE → RESOLVE CONFLICTS → CHARGE → EXECUTE → ABSORB → ENVIRONMENT → DEATH CHECK → RECORD
```

## Core Invariants

| ID | Rule |
|----|------|
| INV-1 | `E_blobs + E_grid + E_dissipated - E_injected = E_total(0)` (tol: 1e-10) |
| INV-2 | Same seed → identical state at every timestep |
| INV-3 | One blob per cell at all times |
| INV-4 | EXECUTE phase is zero-sum for blob energy (reproduction) |
| INV-5 | After DEATH CHECK, surviving blobs have `energy > DEATH_THRESHOLD` |

## Module Map

```
blobsim/
├── blob.py            BlobAttributes (frozen), BlobStatus, Blob
├── config.py          SimulationConfig, SpeciesConfig, validate_config()
├── conflict.py        ConflictResolver protocol, RandomResolver
├── engine.py          SimulationEngine (step loop, DEATH_THRESHOLD)
├── environment.py     EnvironmentFn protocol, DecayEnvironment, RegeneratingEnvironment
├── grid.py            Grid (toroidal 2D, named layers)
├── ledger.py          EnergyLedger, ConfigError, ConservationError
├── observability.py   StateRecorder (population, energy, lifespans)
├── rendering.py       Rendering utilities
├── rng.py             create_rng_hierarchy(), make_blob_rng()
├── rules.py           RulesEngine base class
├── simulation.py      Simulation (user API), SimulationResult
├── types.py           ActionType, Action, Observation, CellView, SensingLevel, ...
└── species/
    ├── grazer.py      GrazerRules — energy-aware movement
    └── random_walker.py  RandomWalkerRules — uniform random movement
```

## Extending

### Add a new species

1. Subclass `RulesEngine` in `blobsim/species/your_species.py`
2. Implement `decide(observation, rng) -> Action`
3. Register in `blobsim/species/__init__.py`
4. Pass a `SpeciesConfig(rules_factory=YourRules, ...)` to `SimulationConfig`

### Add a new environment

1. Implement the `EnvironmentFn` protocol: `initial_grid(size) -> dict[str, ndarray]` and `step(grid, rng)`
2. Pass the instance to `SimulationConfig(environment=...)`

## Testing & Benchmarks

| Layer | Scope | Location |
|-------|-------|----------|
| Unit / integration | Single-step mechanical correctness | `tests/` |
| Calibration | Single-agent multi-step predictions | `benchmark/b2a_*` |
| Convergence | Population-level stable-state dynamics | `benchmark/sc*` |

See [`benchmark/benchmark.md`](benchmark/benchmark.md) for analytical derivations and coverage.

## Commands

```bash
# Run simulation
python main.py

# Tests
python -m pytest tests/ -v

# Benchmarks
python -m benchmark.run_all

# Type check
pyright

# Lint & format
ruff check .
ruff format .
```
