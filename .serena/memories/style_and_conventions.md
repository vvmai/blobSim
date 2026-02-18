# Code Style & Conventions

## General
- **`from __future__ import annotations`** at top of every module
- **Type hints** on all function signatures (return types included)
- **Docstrings**: module-level docstring with purpose + spec reference. Class/function docstrings with Args/Returns/Raises where appropriate.
- **Dataclasses** for data containers (`@dataclass` or `@dataclass(frozen=True)`)
- **Immutable containers**: `MappingProxyType` for dicts, `frozenset` for sets in frozen dataclasses

## Imports
- `from __future__ import annotations` first
- stdlib → third-party → local, alphabetical within groups
- `TYPE_CHECKING` block for circular/type-only imports
- Absolute imports only, no wildcards

## Naming
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE` for module-level constants (e.g. `DEATH_THRESHOLD`, `MOORE`, `VON_NEUMANN`)

## Error handling
- Custom exceptions in `ledger.py`: `ConfigError`, `ConservationError`, `InvariantError`, `RulesEngineError`
- Conservation checks use `if/raise` (not `assert`) — must survive `python -O`
- Tolerance: `1e-10` for floating-point comparisons
- `math.fsum` for compensated summation to prevent float drift

## Testing
- pytest with fixtures in `conftest.py`
- `FixedActionRules` for deterministic testing
- Factory helpers: `make_engine()`, `make_blob()`, `make_observation()`, `make_cell_view()`

## Architecture patterns
- **Pluggable rules**: `RulesEngine` base class, species implement `decide(observation, rng) -> Action`
- **Pluggable environments**: `EnvironmentFn` protocol, implementations: `DecayEnvironment`, `RegeneratingEnvironment`
- **Pluggable conflict resolution**: `ConflictResolver` protocol, default `RandomResolver`
- **Energy ledger**: Central accounting for all energy flows, enforces conservation invariant
- **RNG hierarchy**: Seeded deterministic RNG — `create_rng_hierarchy(seed)` → engine_rng, env_rng, blob_pool_entropy
