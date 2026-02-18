# GOL — Blob Simulation

## Purpose
Cellular automaton on a 2D toroidal grid. Entities ("blobs") have individual energy, pluggable behavior (rules engines), and modular action sets. Grid environment evolves independently via pluggable functions. Core invariant: **energy conservation**.

## Tech Stack
- **Language**: Python 3.12+
- **Dependencies**: NumPy (core engine), Pyright (type checking)
- **Platform**: macOS (Darwin)
- **No framework** — custom NumPy implementation for domain-specific conflict resolution, energy ledger, per-blob RNG.

## Core Invariants
1. **INV-1 Energy Conservation**: `E_blobs(t) + E_grid(t) + E_dissipated(t) - E_injected(t) = E_total(0)` (tolerance 1e-10)
2. **INV-2 Reproducibility**: Same seed → identical state at every timestep
3. **INV-3 One Blob Per Cell**: No cell contains >1 blob
4. **INV-4 No Energy Creation by Blobs**: EXECUTE phase is zero-sum for blob energy during reproduction
5. **INV-5 Non-Negative Energy**: After DEATH CHECK, all surviving blobs have energy > DEATH_THRESHOLD > 0

## Codebase Structure
```
gol/
├── main.py                   # Entry point — runs 200-step demo simulation
├── tech_spec_mvp.md          # Full technical specification
├── pyrightconfig.json        # Type checker config (Python 3.12, Darwin)
├── blobsim/                  # Core simulation package
│   ├── __init__.py           # Public API re-exports
│   ├── blob.py               # BlobAttributes, BlobStatus, Blob
│   ├── config.py             # SimulationConfig, SpeciesConfig, validate_config()
│   ├── conflict.py           # ConflictResolver, RandomResolver
│   ├── engine.py             # SimulationEngine (DEATH_THRESHOLD constant)
│   ├── environment.py        # EnvironmentFn, DecayEnvironment, RegeneratingEnvironment
│   ├── grid.py               # Grid (toroidal 2D, energy layer)
│   ├── ledger.py             # EnergyLedger, error types (ConfigError, ConservationError, etc.)
│   ├── observability.py      # StateRecorder
│   ├── rendering.py          # Rendering utilities
│   ├── rng.py                # RNG hierarchy (create_rng_hierarchy, make_blob_rng)
│   ├── rules.py              # RulesEngine base class
│   ├── simulation.py         # Simulation, SimulationResult (user-facing)
│   ├── types.py              # ActionType, Action, Observation, CellView, etc.
│   └── species/              # Pluggable species implementations
│       ├── grazer.py         # GrazerRules
│       └── random_walker.py  # RandomWalkerRules
├── benchmark/                # Calibration & convergence benchmarks
│   ├── run_all.py            # Runner (python -m benchmark.run_all)
│   ├── initial_conditions.py # Shared IC helpers
│   ├── b2a_single_lifespan.py
│   ├── sc1_decay_extinction.py
│   ├── sc4_reproduction_extinction.py
│   ├── sc5_carrying_capacity.py
│   ├── sc6_energy_equilibrium.py
│   └── results/              # JSON benchmark results
└── tests/                    # pytest test suite
    ├── conftest.py           # Shared fixtures (FixedActionRules, make_engine, make_blob, etc.)
    ├── test_config_validation.py
    ├── test_engine_mechanics.py
    ├── test_environment.py
    ├── test_integration_conservation.py
    ├── test_integration_mechanical.py
    └── test_species_rules.py
```
