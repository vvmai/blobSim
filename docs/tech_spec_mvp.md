# Technical Specification: Blob Simulation MVP

## Table of Contents
- [1. Context](#1-context)
- [2. Invariants](#2-invariants)
- [3. Data Model](#3-data-model)
- [4. Grid & Environment](#4-grid--environment)
- [5. Blob Architecture](#5-blob-architecture)
- [6. Observation & Sensing Model](#6-observation--sensing-model)
- [7. Rules Engine](#7-rules-engine)
- [8. Action System](#8-action-system)
- [9. Simulation Engine](#9-simulation-engine)
- [10. Conflict Resolution](#10-conflict-resolution)
- [11. Energy Accounting](#11-energy-accounting)
- [12. Observability Module](#12-observability-module)
- [13. RNG Architecture](#13-rng-architecture)
- [14. User API](#14-user-api)
- [15. MVP Scope](#15-mvp-scope)
- [16. Module Structure](#16-module-structure)
- [17. Verification](#17-verification)
- [Appendix A: Blob-Blob Interactions (v2 Design Notes)](#appendix-a-blob-blob-interactions-v2-design-notes)
- [Appendix B: Config Validation Preconditions](#appendix-b-config-validation-preconditions)

---

## 1. Context

Cellular automaton on a 2D toroidal grid. Entities ("blobs") have individual energy, pluggable behavior (rules engines), and modular action sets. Grid environment evolves independently via pluggable functions. Core invariant: energy conservation — all tests derive from it.

**Decision**: Custom NumPy implementation. No ABM framework. Rationale: domain-specific conflict resolution, energy ledger, per-blob RNG. ~500-800 LOC for core engine.

---

## 2. Invariants

### INV-1: Energy Conservation
```
E_blobs(t) + E_grid(t) + E_dissipated(t) - E_injected(t) = E_total(0)
```
Where:
- `E_blobs(t)` = sum of all living blob energies
- `E_grid(t)` = sum of all cell values on the `"energy"` grid layer only
- `E_dissipated(t)` = cumulative energy lost to metabolism + action costs + death remainders
- `E_injected(t)` = cumulative energy added by environment replenishment
- `E_total(0)` = `E_blobs(0) + E_grid(0)` at simulation start

Checked every step via explicit `if/raise` (not `assert` — must survive `python -O`).
Tolerance: `1e-10`. Ledger uses compensated summation (`math.fsum` on delta lists) to prevent floating-point drift over long runs.

### INV-2: Reproducibility
Same seed → identical state at every timestep. ALL RNG flows through seeded generators. No implicit randomness.

### INV-3: One Blob Per Cell
`len(occupied_cells) == len(alive_blobs)` at all times. No cell contains >1 blob.

### INV-4: No Energy Creation by Blobs
The EXECUTE phase is zero-sum for blob energy during reproduction: `parent.energy -= offspring_energy`, `child.energy = offspring_energy`. Action costs are separately accounted in the CHARGE phase and go to `E_dissipated`.

### INV-5: Non-Negative Energy (end-of-step)
After DEATH CHECK, all surviving blobs have `energy > DEATH_THRESHOLD > 0`. Energy MAY be transiently negative between CHARGE and ABSORB (when `blob.energy < base_metabolic_cost`). This is an internal engine state — never observable outside the step loop. ABSORB may recover the blob; if not, DEATH CHECK removes it. No clamping — full cost always deducted in CHARGE for correct conservation accounting.

---

## 3. Data Model

### BlobAttributes (frozen — set at creation, never changes)
```python
@dataclass(frozen=True)
class BlobAttributes:
    species: str                                        # species identifier
    max_energy: float                                   # energy cap (absorption limited to this)
    offspring_energy: float                              # energy transferred to child on reproduction
    reproduction_threshold: float                       # min energy to attempt reproduction
    base_metabolic_cost: float                          # energy/step (always paid)
    action_costs: MappingProxyType[ActionType, float]   # immutable cost lookup per action type
    observation_radius: int                             # Moore neighborhood radius (≥1); controls Observation field of view
    sensing_level: SensingLevel                         # what info is visible about neighbors
    allowed_actions: frozenset[ActionType]              # what this species can do
    allowed_directions: frozenset[tuple[int, int]]      # legal (dx,dy) for MOVE; e.g. VON_NEUMANN or MOORE
    extra: MappingProxyType[str, Any]                   # arbitrary species-specific params (immutable)
```

**Immutability**: `action_costs` and `extra` use `types.MappingProxyType` to enforce deep immutability. A factory classmethod handles wrapping at construction time.

**Note on hashability**: `MappingProxyType` is immutable but NOT hashable. `BlobAttributes` uses `frozen=True` for shallow immutability enforcement, but instances are not hashable (cannot be used as dict keys or set members). This is acceptable — `BlobAttributes` are compared by identity, not hashed.

### `BlobAttributes.create()` Factory
```python
@classmethod
def create(
    cls,
    species: str,
    max_energy: float,
    offspring_energy: float,
    reproduction_threshold: float,
    base_metabolic_cost: float,
    move_cost: float,                          # → action_costs[MOVE]
    reproduce_cost: float,                     # → action_costs[REPRODUCE]
    observation_radius: int,
    allowed_actions: frozenset[ActionType],
    allowed_directions: frozenset[tuple[int, int]],
    idle_cost: float = 0.0,                    # → action_costs[IDLE]; default 0
    sensing_level: SensingLevel = SensingLevel.BASIC,
    extra: dict[str, Any] | None = None,
) -> "BlobAttributes":
    """Build BlobAttributes with validation. Wraps dicts → MappingProxyType."""
```

Factory responsibilities:
1. Builds `action_costs = MappingProxyType({IDLE: idle_cost, MOVE: move_cost, REPRODUCE: reproduce_cost})`
2. Wraps `extra` (or `{}`) into `MappingProxyType`
3. Validates: `ActionType.IDLE in allowed_actions` (mandatory)
4. Validates: `offspring_energy > base_metabolic_cost` (child survives first step)
5. Returns frozen `BlobAttributes` instance

### Direction Presets
```python
VON_NEUMANN = frozenset({(-1,0), (1,0), (0,-1), (0,1)})           # 4-connected
MOORE = frozenset({(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)})  # 8-connected
```
`allowed_directions` controls which `(dx,dy)` values are legal for MOVE actions. Species with `VON_NEUMANN` can only move orthogonally; species with `MOORE` can also move diagonally. Custom direction sets (e.g., knight moves) are supported by passing arbitrary offsets.

### BlobStatus (mutable — updated each step)
```python
@dataclass
class BlobStatus:
    energy: float
    age: int                  # steps survived (0 on birth step, incremented in POST-STEP)
    position: tuple[int, int]
    alive: bool = True
```

`age` starts at 0 when created (initial placement or reproduction). Incremented once per step in POST-STEP, after DEATH CHECK. A newborn created at step t has `age=0` during step t's CHARGE/ABSORB, `age=1` at the start of step t+1.

### BlobID
Monotonically increasing int. Never reused. Dead blob IDs persist in observability history. The engine maintains a `_next_blob_id: int` counter, incremented on every blob creation (initial placement and reproduction). IDs are assigned in creation order: initial blobs first (species list order, then within-species order), then births in step-execution order.

### Blob
```python
class Blob:
    id: int
    attributes: BlobAttributes
    status: BlobStatus
    rules: RulesEngine        # stateful — persists across steps, can maintain internal memory
    rng: np.random.Generator  # per-blob, seeded from (global_seed, blob_id)
```

---

## 4. Grid & Environment

### Grid (generic layer system)
```python
class Grid:
    size: int                                    # n for nxn
    layers: dict[str, np.ndarray]                # named (n,n) float64 arrays
    _occupancy: np.ndarray                       # (n,n) int32 — blob_id or -1
```

The `"energy"` layer is required and drives absorption. Additional layers (e.g., `"temperature"`, `"toxicity"`) can be registered by environment functions and are passed to blobs via `CellView.layers` (see §6 CellView).

### Grid Public API
```python
# Read-only properties
grid.energy    → grid.layers["energy"]       # convenience for energy layer
grid.occupancy → grid._occupancy (read-only view)  # np.ndarray, blob_id or -1

# Mutation methods (engine-only, not called by rules engines)
grid.place_blob(blob_id: int, pos: tuple[int, int])    # set _occupancy[pos] = blob_id
grid.remove_blob(pos: tuple[int, int])                  # set _occupancy[pos] = -1
grid.move_blob(blob_id: int, old: tuple[int, int], new: tuple[int, int])
    # _occupancy[old] = -1; _occupancy[new] = blob_id
```

Toroidal wrapping: `(x + dx) % size, (y + dy) % size`

Precomputed neighbor offsets for Moore neighborhood (radius=1):
```python
MOORE_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
```
Radius > 1: all cells within Chebyshev distance r.

### EnvironmentFn (ABC)
```python
class EnvironmentFn(ABC):
    @abstractmethod
    def replenish(self, grid: Grid, step: int, rng: np.random.Generator) -> float:
        """Modify grid layers in-place. Return total energy injected this step."""
        ...

    @abstractmethod
    def initial_grid(self, grid_size: int) -> dict[str, np.ndarray]:
        """Return dict of layer_name → (n,n) initial arrays. Must include 'energy'."""
        ...
```

The `replenish()` contract:
- Receives full Grid (can modify any layer) + environment RNG
- Returns **exact** amount of energy added to the `"energy"` layer (for `E_injected` ledger)
- Called once per step, after death check
- MVP implementations ignore `rng`; stochastic environments (v2+) use it
- ENVIRONMENT runs after DEATH CHECK, so cells vacated by dead blobs are unoccupied and eligible for replenishment in the same step

### MVP Implementations

**DecayEnvironment**: No replenishment. All blobs eventually die.
```python
class DecayEnvironment(EnvironmentFn):
    def __init__(self, initial_cell_energy: float):
        self.initial_cell_energy = initial_cell_energy

    def replenish(self, grid, step, rng):
        return 0.0
    def initial_grid(self, grid_size):
        return {"energy": np.full((grid_size, grid_size), self.initial_cell_energy)}
```

**RegeneratingEnvironment**: Unoccupied cells regrow toward capacity at rate `r`.
```python
class RegeneratingEnvironment(EnvironmentFn):
    def __init__(self, rate: float, capacity: float, initial_cell_energy: float):
        self.rate = rate
        self.capacity = capacity
        self.initial_cell_energy = initial_cell_energy

    def replenish(self, grid, step, rng):
        energy = grid.energy
        unoccupied = grid.occupancy < 0
        space = self.capacity - energy
        addition = np.minimum(self.rate, np.maximum(space, 0.0))
        addition *= unoccupied
        energy += addition
        return float(addition.sum())

    def initial_grid(self, grid_size):
        return {"energy": np.full((grid_size, grid_size), self.initial_cell_energy)}
```

---

## 5. Blob Architecture

### Composition (not inheritance)
```
Blob (concrete, owns state + RNG)
 ├── has-a: BlobAttributes (frozen)
 ├── has-a: BlobStatus (mutable)
 └── has-a: RulesEngine (ABC — pluggable brain, stateful)
```

Blob is NOT subclassed per species. Species differentiation is via:
1. `BlobAttributes` (different costs, thresholds, allowed actions, sensing level)
2. `RulesEngine` implementation (different decision logic, can maintain internal memory)

This keeps the engine species-agnostic. The engine never inspects species type — it calls `blob.rules.decide()` and processes the returned action uniformly.

### Blob Lifecycle
```
BIRTH → ALIVE (energy > ε) → DEATH (energy ≤ ε)
```
- Birth: from initial placement or reproduction
- Death: starvation only (MVP). Remaining energy → `E_dissipated`
- No resurrection. Dead blobs removed from grid, preserved in observability history.

---

## 6. Observation & Sensing Model

### Information Boundary

The rules engine operates under **fog of war**. The Observation is a frozen snapshot taken at the START of the step (Phase 1), before any blob has acted. The rules engine cannot see:
- What other blobs will do (simultaneous decisions)
- Whether its own action will succeed (conflict resolution hasn't happened)
- Who will die this step
- What the environment will inject

This is enforced structurally: Observation is built in Phase 1; rules engine runs in Phase 2.

### Sensing Levels

Different species can perceive different amounts of information about neighbors.

```python
class SensingLevel(Enum):
    BASIC = "basic"      # occupied + species only
    ENERGY = "energy"    # + occupant energy
    FULL = "full"        # + occupant energy, age
```

The engine builds `CellView` according to the observing blob's `sensing_level`. Fields not covered by the blob's sensing level are `None`. This is enforced by the engine, not by the rules engine.

**Smoke test**: Given blob A (`BASIC`) and blob B (`ENERGY`) both adjacent to blob C (energy=5.0, age=3, species="prey"):
- A's CellView for C: `occupied=True, occupant_species="prey", occupant_energy=None, occupant_age=None`
- B's CellView for C: `occupied=True, occupant_species="prey", occupant_energy=5.0, occupant_age=None`
- A `FULL` observer would see: `occupant_energy=5.0, occupant_age=3`

### CellView
```python
@dataclass(frozen=True)
class CellView:
    offset: tuple[int, int]                # relative to observer: (0,0) = self, (-1,1) = NE neighbor
    layers: MappingProxyType[str, float]    # grid layer values at this cell (energy, temperature, etc.)
    occupied: bool
    occupant_species: str | None = None     # always visible if occupied
    occupant_energy: float | None = None    # visible only if sensing >= ENERGY
    occupant_age: int | None = None         # visible only if sensing >= FULL
```

**Relative coordinates only**: `offset` is the `(dx, dy)` displacement from the observer's position. The rules engine never sees absolute grid coordinates — the engine handles all `offset → absolute` conversion using `(pos + offset) % grid_size`. This makes rules engines grid-size-agnostic and toroidal wrapping invisible.

`CellView.layers` contains values from all registered grid layers at this cell (see §4 Grid). Convenience: `cell.energy` property → `cell.layers["energy"]`.

### Observation
```python
@dataclass(frozen=True)
class Observation:
    self_status: BlobStatus                 # COPY via dataclasses.replace() (mutations have no effect)
    self_attributes: BlobAttributes         # blob's own attributes (frozen, safe to share)
    neighborhood: tuple[CellView, ...]      # visible cells within observation_radius (see below)
    current_cell: CellView                  # cell blob is standing on (offset=(0,0))
    step: int                               # current simulation step
```

### Configurable View Radius

The `neighborhood` field contains ALL cells within Chebyshev distance `observation_radius` of the blob (Moore neighborhood), **excluding the blob's own cell** (which is in `current_cell`).

- `observation_radius=1`: 8 neighbors (standard Moore)
- `observation_radius=2`: 24 neighbors (5x5 minus center)
- `observation_radius=r`: `(2r+1)² - 1` neighbors

`current_cell` always has `offset=(0,0)`, `occupied=True`, `occupant_species=blob.attributes.species`. Self-sensing follows the blob's own `sensing_level` (a `BASIC` blob does NOT see its own energy in `current_cell`).

The engine builds the neighborhood in Phase 1 (OBSERVE) by iterating offsets `(dx, dy)` where `max(|dx|, |dy|) <= observation_radius` and `(dx,dy) != (0,0)`. Each cell becomes a `CellView` with `offset=(dx,dy)`. The rules engine receives the full visible field and can use it for directional decision-making (e.g., move toward richest cell within view).

Species with larger `observation_radius` see farther but the data volume grows quadratically. This is a per-species tunable on `BlobAttributes` — different species can have different radii in the same simulation.

---

## 7. Rules Engine

```python
class RulesEngine(ABC):
    @abstractmethod
    def decide(self, observation: Observation, rng: np.random.Generator) -> Action:
        """Given what the blob sees, return one action."""
        ...
```

Contract:
- Receives full `Observation` + blob's personal RNG
- Returns exactly one `Action`
- Must only return actions from `observation.self_attributes.allowed_actions`
- MOVE direction MUST be from `observation.self_attributes.allowed_directions` (engine crashes otherwise)
- MUST check feasibility: REPRODUCE only if empty neighbor exists (engine crashes otherwise)
- SHOULD NOT assume its action will succeed (conflicts can still happen)
- Violations of MUST constraints raise `RulesEngineError` with blob_id, action, and reason

### Decision-Making Guidance (for rules engine implementers)
The rules engine sees which neighbors are empty/occupied and their grid layer values. It makes *informed* decisions:
- No empty neighbors in `allowed_directions` → MUST NOT output REPRODUCE (contract violation → crash)
- High-energy cell nearby → MOVE toward it
- Low energy → IDLE to conserve (only pays base metabolic, `action_costs[IDLE] = 0`)
- IDLE is a first-class deliberate action, not just a collision fallback
- Energy below `reproduction_threshold` → don't attempt REPRODUCE (affordability gate would reject anyway)

### Statefulness

`RulesEngine` instances persist across steps and are owned by a single blob. Implementations MAY maintain internal state (observation history, learned parameters, etc.) for memory and learning:

```python
class StatefulExample(RulesEngine):
    def __init__(self):
        self.history = []
    def decide(self, observation, rng):
        self.history.append(observation)
        # use history for decisions...
```

No spec change needed for RL — the ABC already supports stateful implementations.

### MVP Implementations

**RandomWalkerRules**: Decision priority:
1. If `energy >= reproduction_threshold` AND empty neighbor exists in `allowed_directions` → `Action(REPRODUCE)`
2. Otherwise, pick uniformly at random from `allowed_directions` offsets (as MOVE) + IDLE. Total options = `len(allowed_directions) + 1`, equally weighted.

**GrazerRules**: Decision priority:
1. If `energy >= reproduction_threshold` AND empty neighbor exists in `allowed_directions` → `Action(REPRODUCE)`
2. Find unoccupied cell in neighborhood with highest `cell.energy`. If tie, break uniformly at random using `rng`.
3. If best cell has energy > 0 and is in `allowed_directions` → `Action(MOVE, offset)`
4. Otherwise → `Action(IDLE)`

### Future (v2+): RL-Compatible Interface
```python
class RLRulesEngine(RulesEngine):
    """Wraps an RL policy. observation → feature vector → policy → action."""
    def __init__(self, policy: Policy):
        self.policy = policy
        self.obs_buffer = []
    def decide(self, observation, rng):
        self.obs_buffer.append(observation)
        features = self._encode(self.obs_buffer[-5:])
        action_probs = self.policy(features)
        return self._sample(action_probs, rng)
```

---

## 8. Action System

### ActionType Enum
```python
class ActionType(Enum):
    IDLE = "idle"
    MOVE = "move"
    REPRODUCE = "reproduce"
```

### Action
```python
@dataclass(frozen=True)
class Action:
    type: ActionType
    direction: tuple[int, int] | None = None  # (dx, dy) for MOVE; None for IDLE/REPRODUCE
```

**MOVE** carries a direction `(dx, dy)`. The rules engine picks the direction from `allowed_directions`; Phase 3 validates it; Phase 4 computes the wrapped absolute target cell via `(pos + direction) % grid_size`. The result MUST map to a valid grid coordinate.

**REPRODUCE** has `direction=None`. The engine picks the target cell (random empty neighbor) during CLAIM.

**IDLE** has `direction=None`. No target cell.

This avoids enum explosion for non-adjacent moves (v2: LUNGE = `Action(MOVE, (2, 0))`).

### Standard Direction Offsets
```python
DIRECTIONS = {
    "N":  (-1,  0), "NE": (-1,  1), "E": ( 0,  1), "SE": ( 1,  1),
    "S":  ( 1,  0), "SW": ( 1, -1), "W": ( 0, -1), "NW": (-1, -1),
}
```

### ResolvedAction (output of RESOLVE, input to EXECUTE)
```python
@dataclass(frozen=True)
class ResolvedAction:
    blob_id: int
    action: Action
    target: tuple[int, int] | None   # resolved target cell (for MOVE and REPRODUCE)
    succeeded: bool
```

Decouples the Claim (conflict input) from the execution instruction. The engine uses `ResolvedAction` in EXECUTE and CHARGE.

### Affordability Gate

The affordability check is a **worst-case gate** — it verifies the blob can pay the full cost if the action succeeds. It is NOT the charge formula.

```python
def can_afford(blob: Blob, action: Action) -> bool:
    """Gate check: can blob afford this action if it succeeds?"""
    cost = blob.attributes.base_metabolic_cost + blob.attributes.action_costs[action.type]
    if action.type == ActionType.REPRODUCE:
        cost += blob.attributes.offspring_energy  # transfer only happens on success
    return blob.status.energy >= cost
```

If unaffordable → downgrade to IDLE. **One-shot**: downgrade fires once. IDLE is NOT re-checked for affordability. IDLE is always taken. If `blob.energy < base_metabolic_cost`, the blob proceeds with IDLE and energy goes transiently negative in CHARGE (see INV-5). Metabolism is unconditional — it is a tax on existence, not an action cost.

**Distinction from CHARGE**: The CHARGE phase deducts `base_metabolic + action_costs[type]` only. The `offspring_energy` is deducted separately in EXECUTE, only for successful reproductions. Failed reproductions pay `action_costs[REPRODUCE]` (attempt cost) but NOT `offspring_energy` (transfer cost).

---

## 9. Simulation Engine

### Step Execution (10 phases)

```
Step t → t+1:

Phase 1: OBSERVE
  For each alive blob: build Observation from grid state + neighborhood.
  CellView fields populated according to observer's sensing_level.
  Observation is a frozen snapshot — no post-decision info leaks.

Phase 2: DECIDE
  For each alive blob: action = blob.rules.decide(observation, blob.rng)

Phase 3: VALIDATE
  For each blob:
    Contract enforcement (rules engine bugs → crash with descriptive exception):
    1. if action.type not in blob.attributes.allowed_actions → raise RulesEngineError
    2. if MOVE and action.direction not in blob.attributes.allowed_directions → raise RulesEngineError
    Game mechanic (legitimate state → graceful downgrade):
    3. Affordability: if not can_afford(blob, action) → Action(IDLE)
  Affordability downgrade is one-shot: IDLE is NOT re-checked for affordability (see §8).

Phase 4: CLAIM
  Convert actions to cell claims (engine handles all coordinate translation):
  - MOVE:       target = ((pos[0]+direction[0]) % grid_size, (pos[1]+direction[1]) % grid_size)
                Direction already validated in Phase 3.
                → Claim(blob_id, target, action)
  - REPRODUCE:  Candidate cells = empty cells at offsets in blob.attributes.allowed_directions.
                Pick one uniformly at random (blob.rng) → Claim(blob_id, target, action).
                If no candidate: raise RulesEngineError("REPRODUCE with no empty neighbor").
                The rules engine has full neighborhood visibility (§6) — outputting REPRODUCE
                with no valid target is a contract violation, not an edge case.
  - IDLE:       No claim (implicit stay at current cell)

  Output of Phase 4:
    claims_by_cell: dict[tuple[int,int], list[Claim]]   # from MOVE and REPRODUCE
    # IDLE blobs produce no claim — handled in Phase 5.
    # REPRODUCE with no candidate crashes (RulesEngineError), so no early_failures dict.

  occupied_cells = {blob.status.position for blob in alive_blobs}
  All current positions are locked during resolution (conservative vacancy model, see §10).

Phase 5: RESOLVE
  resolved = resolve_all(claims_by_cell, occupied_cells, resolver, engine_rng)
  # resolve_all only covers blobs with claims; see §10 for pseudocode.

  Synthesize ResolvedActions for IDLE blobs (no claim):
    for blob in alive_blobs:
      if blob.id not in resolved:
        resolved[blob.id] = ResolvedAction(blob.id, Action(IDLE), target=None, succeeded=True)
  After this, EVERY alive (pre-EXECUTE) blob has exactly one ResolvedAction.

Phase 6: EXECUTE
  For each ResolvedAction where succeeded=True:
  - MOVE: update blob.status.position
  - REPRODUCE:
      parent.energy -= offspring_energy
      child = Blob(id=_next_blob_id++, energy=offspring_energy, position=target,
                    attributes=parent.attributes,
                    rules=species_configs[parent.attributes.species].rules_factory(),
                    rng=make_blob_rng(blob_pool_entropy, child.id))
  - IDLE: no-op
  Occupancy update (MUST be atomic/batch):
    1. Collect all moves: list of (blob_id, old_pos, new_pos)
    2. Collect all births: list of (child_id, pos)
    3. Apply ALL removals from old positions, THEN all placements to new positions.
    No per-blob incremental updates — prevents transient INV-3 violations.
  The engine maintains a dict[str, SpeciesConfig] (species name → config) for factory lookup.

Phase 7: CHARGE
  Per-blob iteration order is irrelevant (one blob per cell, no cross-blob dependencies).
  Newborns identified by: blob.id not in resolved_actions dict (created in Phase 6, after RESOLVE).
  For each alive blob (including newborns):
    if blob.id not in resolved_actions:       # newborn
      cost = base_metabolic_cost
    else:
      cost = base_metabolic_cost + action_costs[resolved_actions[blob.id].action.type]
    blob.energy -= cost                       # MAY go transiently negative (INV-5)
    ledger.add_dissipated(cost)               # always full cost, no clamping
  Note: failed actions (RESOLVE or CLAIM) pay their attempted action cost, not IDLE cost.
  Actions downgraded to IDLE in Phase 3 carry ActionType.IDLE → pay action_costs[IDLE].
  Actions that failed in Phase 5 carry their original type → pay original action cost.

Phase 8: ABSORB
  Per-blob iteration order is irrelevant (one blob per cell, each absorbs from its own cell).
  For each alive blob:
    headroom = blob.attributes.max_energy - blob.energy
    if headroom < 0:
        raise InvariantError(f"blob {blob.id} energy {blob.energy} > max_energy {blob.attributes.max_energy}")
    absorbed = min(grid.energy[pos], headroom)
    blob.energy += absorbed
    grid.energy[pos] -= absorbed

Phase 9: DEATH CHECK
  DEATH_THRESHOLD = 1e-10  (distinct from conservation tolerance, same value but different purpose)
  For each blob where energy ≤ DEATH_THRESHOLD:
    ledger.add_dissipated(blob.energy)  # remainder (may be negative if metabolism > absorption)
    blob.status.alive = False
    grid.remove_blob(pos)
    # Do NOT zero out blob.status.energy — preserve for BlobRecord observability.

Phase 10: ENVIRONMENT
  injected = environment.replenish(grid, step, env_rng)
  ledger.add_injected(injected)

POST-STEP: RECORD + VALIDATE
  For each alive blob: blob.status.age += 1   # includes newborns (age 0 → 1)
  observability.snapshot(step, blobs, grid, ledger, step_data)
  check_conservation(blobs, grid, ledger)     # raises ConservationError, not assert
  check_one_per_cell(grid)                    # raises InvariantError
```

### Phase Ordering Rationale
| Order | Why |
|-------|-----|
| OBSERVE → DECIDE | Rules engine needs current world state; frozen snapshot prevents info leaks |
| VALIDATE after DECIDE | Enforces rules engine contract (crash) + affordability downgrade |
| CLAIM after VALIDATE | Target resolution; REPRODUCE feasibility enforced (crash on violation) |
| RESOLVE after CLAIM | Simultaneous conflict resolution; `occupied_cells` = all current positions |
| EXECUTE before CHARGE | Reproduction creates child before costs applied |
| CHARGE before ABSORB | Metabolism before feeding (can't eat out of starvation mid-step) |
| ABSORB before DEATH | Feeding can prevent death this step |
| DEATH before ENVIRONMENT | Dead blobs don't benefit from replenishment |
| ENVIRONMENT last | Replenishment applies to end-of-step grid |

### Engine Validation

### Error Handling Philosophy: Crash on Contract Violation

The engine distinguishes two categories:
- **Contract violations** → `RulesEngineError` (crash with descriptive message including blob_id, action, reason)
  - Invalid action type (Phase 3)
  - Invalid MOVE direction (Phase 3)
  - REPRODUCE with no empty neighbor (Phase 4)
- **Legitimate game states** → graceful downgrade to IDLE
  - Unaffordable action (Phase 3): blob has info to check this but isn't required to

Rationale: the rules engine receives full Observation (§6) with all info needed to avoid contract violations. If a violation occurs, it's a bug in the rules engine implementation — crash immediately with a diagnostic message so the developer can fix it. No silent degradation, no weak error handling.

### Newborn Handling
- Created in EXECUTE (Phase 6); immediately alive and on the grid
- Identified in CHARGE by absence from `resolved_actions` dict
- Charged `base_metabolic_cost` only in CHARGE (no action cost)
- DO absorb energy in ABSORB
- Do NOT act until step t+1 (no OBSERVE/DECIDE this step)
- Age = 0 during birth step; incremented to 1 in POST-STEP
- `BlobRecord` for newborns: `action_taken=ActionType.IDLE, action_succeeded=True, action_cost=0.0`

---

## 10. Conflict Resolution

### Claim Model
```python
@dataclass(frozen=True)
class Claim:
    blob_id: int
    target: tuple[int, int]
    action: Action
    priority: float = 0.0    # v2: enables priority-based resolution (energy, strength, etc.)
```

### Conservative Vacancy Model (MVP Resolver Policy)

The conservative vacancy model has two parts with different abstraction boundaries:

1. **Lock set construction** (engine-level, Phase 4): `occupied_cells = {blob.status.position for blob in alive_blobs}` — ALL current positions, movers included. This is built by the engine, not the resolver.
2. **Resolution logic** (resolver-level, Phase 5): claims on occupied cells fail; ties broken randomly. This is in the pluggable `ConflictResolver`.

MVP behavior: no swaps, no chains. Claims on empty cells with single claimant succeed; multiple claimants → random winner.

**V2 note**: A `PredationResolver` would require changes to BOTH the engine (Phase 4 lock set construction must allow claims on occupied cells) AND the resolver (attack resolution logic). See Appendix A.

### Resolver ABC
```python
class ConflictResolver(ABC):
    @abstractmethod
    def resolve(self, claims: list[Claim], rng: np.random.Generator) -> Claim:
        """Pick one winner from competing claims on a single cell."""
        ...

class RandomResolver(ConflictResolver):
    def resolve(self, claims, rng):
        return claims[rng.integers(len(claims))]
```

### Resolution Algorithm
```python
def resolve_all(claims_by_cell, occupied_cells, resolver, rng):
    results: dict[int, ResolvedAction] = {}

    for cell, claims in sorted(claims_by_cell.items()):  # sorted for deterministic iteration
        if cell in occupied_cells:
            for c in claims:
                results[c.blob_id] = ResolvedAction(c.blob_id, c.action, cell, succeeded=False)
        elif len(claims) == 1:
            c = claims[0]
            results[c.blob_id] = ResolvedAction(c.blob_id, c.action, cell, succeeded=True)
        else:
            winner = resolver.resolve(claims, rng)
            for c in claims:
                succeeded = (c.blob_id == winner.blob_id)
                results[c.blob_id] = ResolvedAction(c.blob_id, c.action, cell, succeeded=succeeded)

    return results
```

Note: `claims_by_cell` iterated in sorted order for deterministic `engine_rng` consumption (INV-2).

---

## 11. Energy Accounting

### Ledger (compensated summation)
```python
class EnergyLedger:
    initial_total: float              # E_blobs(0) + E_grid(0), set once
    _dissipated_deltas: list[float]   # append per-step; sum via math.fsum
    _injected_deltas: list[float]     # append per-step; sum via math.fsum

    @property
    def dissipated(self) -> float:
        return math.fsum(self._dissipated_deltas)

    @property
    def injected(self) -> float:
        return math.fsum(self._injected_deltas)

    def add_dissipated(self, amount: float): self._dissipated_deltas.append(amount)
    def add_injected(self, amount: float): self._injected_deltas.append(amount)
```

Using `math.fsum` on a list of deltas provides exact summation (no accumulation drift), maintaining `1e-10` tolerance even over 100k+ steps.

### Per-Step Accounting Trace (derived view — authoritative source: §9 Phases 6-9)
```
EXECUTE (reproduce, success only):
  parent.energy -= offspring_energy
  child.energy  += offspring_energy
  net ΔE_blobs = 0                   ← INV-4
  (failed reproduce: no energy transfer. offspring_energy stays with parent.)

CHARGE:
  each blob: energy -= (base_metabolic + action_cost)
  ledger.add_dissipated(sum_of_costs)
  ΔE_blobs = -sum(costs)
  ΔE_dissipated = +sum(costs)        ← zero-sum
  Energy may go transiently negative (INV-5). No clamping.

  REPRODUCE cost breakdown:
    Success: EXECUTE transfers offspring_energy (zero-sum).
             CHARGE deducts base_metabolic + action_costs[REPRODUCE].
    Failure: No transfer. CHARGE still deducts base_metabolic + action_costs[REPRODUCE].
    Attempt cost (action_costs[REPRODUCE]) always paid. Transfer cost (offspring_energy) on success only.

ABSORB:
  each blob: energy += absorbed
  grid.energy[pos] -= absorbed
  ΔE_blobs = +total_absorbed
  ΔE_grid = -total_absorbed          ← zero-sum

DEATH:
  ledger.add_dissipated(remainder)
  ΔE_blobs = -remainder
  ΔE_dissipated = +remainder         ← zero-sum

ENVIRONMENT:
  grid.energy += additions
  ledger.add_injected(total_added)
  ΔE_grid = +total_added
  ΔE_injected = +total_added         ← accounted
```

### Conservation Check
```python
class ConservationError(RuntimeError): ...
class InvariantError(RuntimeError): ...      # INV-3 (one-per-cell), INV-5 (energy > max_energy)
class RulesEngineError(RuntimeError): ...    # contract violations: invalid action, direction, infeasible REPRODUCE

def check_conservation(blobs, grid, ledger, tolerance=1e-10):
    e_blobs = math.fsum(b.status.energy for b in blobs if b.status.alive)
    e_grid = float(grid.energy.sum())
    lhs = e_blobs + e_grid + ledger.dissipated - ledger.injected
    delta = lhs - ledger.initial_total
    if abs(delta) > tolerance:
        raise ConservationError(
            f"Conservation violated: delta={delta}, "
            f"E_blobs={e_blobs}, E_grid={e_grid}, "
            f"dissipated={ledger.dissipated}, injected={ledger.injected}"
        )
```

---

## 12. Observability Module

### StateRecorder
Captures full simulation state per step. Dual storage: structured snapshots + columnar export.

`snapshot()` receives: all blobs that were alive at the start of the step (pre-EXECUTE) PLUS newborns created in Phase 6. This includes blobs that died in Phase 9 (recorded with `alive=False`). Does NOT include blobs that died in prior steps.

```python
class StateRecorder:
    blob_records: list[list[BlobRecord]]    # [step][blob_index]
    grid_records: list[np.ndarray]          # [step] → (n,n) energy snapshot (copy)
    ledger_records: list[LedgerSnapshot]    # [step]
    record_interval: int                    # store grid snapshot every N steps (default: 1)
```

`record_interval` controls **grid snapshot frequency only**. Blob records and ledger snapshots are captured every step regardless. Conservation checks also run every step. Grid energy snapshots (expensive: full `(n,n)` array copy) are stored every `record_interval` steps to manage memory on long runs.

### Per-Blob Step Data Flow

```python
@dataclass
class StepData:
    action_taken: ActionType     # from ResolvedAction (IDLE for newborns)
    action_succeeded: bool       # from ResolvedAction (True for newborns)
    action_cost: float           # action_costs[type] only, excludes base_metabolic (0.0 for newborns)
    energy_absorbed: float       # amount absorbed in ABSORB phase
```

The engine accumulates `dict[int, StepData]` during phases 5-8:
- Phase 5 (RESOLVE): `action_taken`, `action_succeeded` from ResolvedAction
- Phase 7 (CHARGE): `action_cost` = `action_costs[resolved_action.action.type]` (0.0 for newborns)
- Phase 8 (ABSORB): `energy_absorbed`

This dict is passed to `StateRecorder.snapshot()` in POST-STEP along with blob state, grid state, and ledger. The recorder builds `BlobRecord` from this data. Dead blobs (Phase 9) are included with `alive=False`.

Memory estimate: 100x100 grid, 500 steps, 200 blobs avg → ~70 MB total (acceptable).

```python
@dataclass(frozen=True)
class BlobRecord:
    step: int
    blob_id: int
    species: str
    position: tuple[int, int]       # position after EXECUTE (final position this step)
    energy: float                    # energy after ABSORB (for living) or at death (for dying)
    age: int                         # age after POST-STEP increment
    action_taken: ActionType         # from ResolvedAction (IDLE for newborns)
    action_succeeded: bool           # from ResolvedAction (True for newborns)
    action_cost: float               # action_costs[type] ONLY (excludes base_metabolic; 0.0 for newborns)
    energy_absorbed: float           # amount absorbed in ABSORB phase
    alive: bool                      # status AFTER DEATH CHECK
    # Total charge = base_metabolic_cost + action_cost (derivable from species attributes + this field)

@dataclass(frozen=True)
class LedgerSnapshot:
    step: int
    e_blobs: float
    e_grid: float
    e_dissipated: float
    e_injected: float
    n_alive: int
    n_births: int
    n_deaths: int
```

### Columnar Export
```python
def to_dataframes(self) -> dict[str, pd.DataFrame]:
    return {
        "blobs": pd.DataFrame([...]),     # one row per (step, blob_id)
        "grid":  pd.DataFrame([...]),     # one row per step (recorded steps only)
        "ledger": pd.DataFrame([...]),    # one row per step, all ledger fields
    }
```

### Query Helpers
```python
def energy_over_time(self) -> np.ndarray:       # (T,) total energy per step
def population_over_time(self) -> np.ndarray:    # (T,) alive count per step
def mean_lifespan(self) -> float:                # avg steps alive at death
def species_population(self) -> dict[str, np.ndarray]:  # per-species pop curves
def conservation_residuals(self) -> np.ndarray:  # (T,) should be ~0
def birth_rate(self) -> np.ndarray:              # (T,) births per step / alive
def death_rate(self) -> np.ndarray:              # (T,) deaths per step / alive
```

---

## 13. RNG Architecture

### Hierarchy
```python
root_seed: int                          # user-provided
root_ss = np.random.SeedSequence(root_seed)

engine_ss, env_ss, blob_pool_ss = root_ss.spawn(3)

engine_rng = np.random.default_rng(engine_ss)    # conflict resolution (deterministic iteration order)
env_rng = np.random.default_rng(env_ss)          # stochastic environments
```

### Per-Blob Independence
```python
def make_blob_rng(blob_pool_entropy: int, blob_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=blob_pool_entropy, spawn_key=(blob_id,))
    return np.random.default_rng(ss)
```

Uses `spawn_key` parameter (not entropy unpacking) to derive per-blob sequences. `blob_pool_entropy = blob_pool_ss.entropy` (an `int` for root-spawned sequences).

Guarantees:
- Adding/removing blob X does not change blob Y's RNG stream
- Same `root_seed` + same `blob_id` → identical sequence regardless of other blobs
- Full reproducibility: same seed → bit-identical simulation

### Deterministic Engine RNG

The `engine_rng` stream depends on the order of `resolve_all` iteration. To ensure INV-2, `claims_by_cell` is iterated in **sorted cell order** (see §10). This makes engine RNG consumption deterministic regardless of dict insertion order.

---

## 14. User API

```python
@dataclass
class SpeciesConfig:
    rules_factory: Callable[[], RulesEngine]   # creates rules engine instance per blob
    attributes: BlobAttributes                  # shared attributes for species
    count: int                                  # initial population
    initial_energy: float                       # energy each blob starts with at step 0

@dataclass
class SimulationConfig:
    grid_size: int
    environment: EnvironmentFn
    species: list[SpeciesConfig]
    seed: int
    conflict_resolver: ConflictResolver = field(default_factory=RandomResolver)
    record_interval: int = 1                    # grid snapshot frequency
```

Config validation runs in `Simulation.__init__()` (see Appendix B).

### Initial Blob Placement
Initial blobs are placed at random unique positions selected using `engine_rng`. Positions are sampled without replacement from all grid cells. Species are placed in `config.species` list order; within each species, blobs are placed in order of ascending `blob_id`. Each initial blob receives `energy=species_config.initial_energy`.

### WorldState
```python
@dataclass(frozen=True)
class WorldState:
    step: int
    n_alive: int
    n_births: int                   # births this step
    n_deaths: int                   # deaths this step
    e_blobs: float                  # total blob energy
    e_grid: float                   # total grid energy
    e_dissipated: float             # cumulative dissipated
    e_injected: float               # cumulative injected
```

### SimulationResult
```python
@dataclass(frozen=True)
class SimulationResult:
    recorder: StateRecorder
    config: SimulationConfig
    final_step: int
```

### Simulation
```python
class Simulation:
    """Single-use. Calling run() or iterate() a second time raises RuntimeError.
    Runs all requested steps even if population reaches 0 (no early termination).
    Phases are no-ops on empty blob collections; environment still replenishes."""
    def __init__(self, config: SimulationConfig): ...

    def run(self, steps: int) -> SimulationResult:
        """Run all steps, return result with full observability data."""
        for _ in self.iterate(steps):
            pass
        return self.result()

    def iterate(self, steps: int) -> Generator[WorldState, None, None]:
        """Yield WorldState after each step. Generator interface."""
        for step in range(steps):
            self._step()
            yield self._world_state()

    def result(self) -> SimulationResult:
        """Return observability data + final state."""
        ...
```

### Usage Example
```python
from blobsim import (
    Simulation, SimulationConfig, SpeciesConfig,
    BlobAttributes, ActionType, SensingLevel, MOORE,
)
from blobsim.species import RandomWalkerRules
from blobsim.environments import RegeneratingEnvironment

config = SimulationConfig(
    grid_size=30,
    environment=RegeneratingEnvironment(rate=0.05, capacity=2.0, initial_cell_energy=1.0),
    species=[
        SpeciesConfig(
            rules_factory=RandomWalkerRules,
            attributes=BlobAttributes.create(
                species="walker",
                max_energy=10.0,
                offspring_energy=3.0,
                reproduction_threshold=7.0,
                base_metabolic_cost=0.1,
                move_cost=0.2,
                reproduce_cost=0.5,
                observation_radius=1,             # sees 8 neighbors (Moore r=1)
                sensing_level=SensingLevel.BASIC,  # sees occupied + species only
                allowed_actions=frozenset(ActionType),
                allowed_directions=MOORE,          # 8-connected movement + reproduction
            ),
            count=50,
            initial_energy=5.0,                    # starting energy per blob
        ),
    ],
    seed=42,
)

sim = Simulation(config)

# Option A: full run
result = sim.run(steps=500)
print(result.recorder.mean_lifespan())
residuals = result.recorder.conservation_residuals()
assert residuals.max() < 1e-10

# Option B: iterate with generator
sim2 = Simulation(config)
for state in sim2.iterate(steps=500):
    if state.step % 100 == 0:
        print(f"Step {state.step}: {state.n_alive} alive")
```

Note: `BlobAttributes.create()` maps keyword args to internal types: `move_cost → action_costs[MOVE]`, `reproduce_cost → action_costs[REPRODUCE]`, `idle_cost → action_costs[IDLE]` (default 0.0). Wraps dicts into `MappingProxyType`, validates all fields including `IDLE in allowed_actions` and `offspring_energy > base_metabolic_cost`.

---

## 15. MVP Scope

### In scope
| Component | What |
|-----------|------|
| Core engine | 10-phase step loop, energy conservation |
| Grid | Toroidal nxn, generic layer system, occupancy tracking |
| Environment ABC | + `DecayEnvironment`, `RegeneratingEnvironment` |
| Blob | `BlobAttributes`, `BlobStatus`, per-blob RNG, sensing levels |
| Rules Engine ABC | + `RandomWalkerRules`, `GrazerRules` |
| Actions | MOVE (8-dir via direction tuple), IDLE, REPRODUCE |
| Conflict resolution | Claims-based with priority field, `RandomResolver` |
| Observability | Full state capture, `to_dataframe()`, query helpers, configurable interval |
| User API | `Simulation(config).run()` / `.iterate()` |
| Tests | Conservation, reproducibility, one-per-cell, decay, ledger decomposition, reproduction |
| Config validation | Preconditions checked at init (see Appendix B) |

### Out of scope (v2+)
- RL-based rules engines (ABC supports it; no implementation)
- Predation / blob-blob energy transfer (architecture supports it; see Appendix A)
- Multi-action turns
- Visualization / animation
- Mutation / evolution / trait inheritance
- GPU/JAX acceleration

---

## 16. Module Structure

```
main.py                          # end-to-end usage example (SC-1)
blobsim/
├── __init__.py              # public API re-exports
├── types.py                 # ActionType, Action, ResolvedAction, Claim, SensingLevel,
│                            #   CellView, Observation, BlobRecord, LedgerSnapshot,
│                            #   VON_NEUMANN, MOORE direction presets
├── grid.py                  # Grid class (toroidal, generic layers, occupancy)
├── environment.py           # EnvironmentFn ABC + DecayEnvironment, RegeneratingEnvironment
├── blob.py                  # Blob, BlobAttributes (with .create() factory), BlobStatus
├── rules.py                 # RulesEngine ABC
├── conflict.py              # ConflictResolver ABC + RandomResolver + resolve_all()
├── engine.py                # SimulationEngine (10-phase step loop, conservation check)
│                            #   owns: _next_blob_id, species_configs dict (built from SimulationConfig.species)
├── ledger.py                # EnergyLedger (compensated summation), ConservationError, InvariantError, RulesEngineError
├── observability.py         # StateRecorder, query helpers, to_dataframe()
├── config.py                # SimulationConfig, SpeciesConfig, validation
├── simulation.py            # Simulation (user-facing: run, iterate, result)
├── rng.py                   # RNG hierarchy: make_blob_rng, seed management
└── species/
    ├── __init__.py
    ├── random_walker.py     # RandomWalkerRules
    └── grazer.py            # GrazerRules
```

Estimated: ~700-1000 LOC core engine, ~100-150 LOC per species.

---

## 17. Verification

### Test 1: Energy Conservation (INV-1)
Run 500 steps, 30x30 grid, 50 walkers + `RegeneratingEnvironment`. Assert `conservation_residuals().max() < 1e-10`.

### Test 2: Reproducibility (INV-2)
Run identical config twice with same seed. Assert state identical at every step (blob positions, energies, grid state).

### Test 3: One-Per-Cell (INV-3)
After every step, assert no cell contains >1 blob.

### Test 4: Eventual Decay
Run with `DecayEnvironment`, `base_metabolic_cost > 0`. Assert population reaches 0.

### Test 5: Energy Ledger Decomposition
Verify `E_dissipated == sum(all metabolism) + sum(all action costs) + sum(all death remainders)` by cross-checking observability blob records field by field.

### Test 6: Reproduction Energy Transfer
Single blob reproduces. Assert `parent.energy_before == parent.energy_after + child.energy + action_cost + base_metabolic`.

### Validation Protocol
After implementation, run all 6 tests. Conservation test is the primary gate.

---

## Success Criteria

Implementation is complete when ALL of the following hold:

### SC-1: `main.py` — End-to-End Simulation
A runnable `main.py` at the project root that:
1. Configures a simulation with both species (RandomWalker + Grazer) on a 30×30 grid with `RegeneratingEnvironment`
2. Runs 200 steps via `sim.run(steps=200)`
3. Prints per-step summary every 50 steps (population, total energy)
4. Asserts conservation residuals < 1e-10
5. Prints final stats: mean lifespan, total births, total deaths, final population by species
6. Exits cleanly with return code 0

This file serves as the **canonical usage example** of the User API (§14). It must use ONLY the public API — no internal imports.

```
python main.py
```

### SC-2: All 6 Verification Tests Pass (§17 Tests 1-6)
Run as a test suite. Zero failures.

### SC-3: All Per-Phase Validation Checks Pass (Phases A-G)
The smoke tests defined in the Implementation Order section. These are the developer's self-check during implementation.

### SC-4: Clean Imports
```
python -c "import blobsim"
```
No import errors. All public API symbols accessible from `blobsim` top-level.

### SC-5: No Warnings Under `python -W error`
```
python -W error main.py
```
No deprecation warnings, no runtime warnings.

### SC-6: Conservation Holds Under `-O`
```
python -O main.py
```
Conservation check uses `if/raise` (not `assert`), so `-O` must not silently disable it.

---

## Implementation Order

### Phase A: Foundation (types.py, rng.py, ledger.py)

1. `types.py` — ActionType, Action, ResolvedAction, Claim, SensingLevel, CellView, Observation, BlobRecord, LedgerSnapshot
2. `rng.py` — seed hierarchy, `make_blob_rng`
3. `ledger.py` — EnergyLedger with compensated summation, ConservationError

**Validation — Phase A:**
| # | Check | What it catches |
|---|-------|-----------------|
| A1 | `make_blob_rng(entropy, 5)` called twice → first 10 values identical | spawn_key determinism broken (caching, statefulness) |
| A2 | `make_blob_rng(entropy, 5)` vs `make_blob_rng(entropy, 6)` → first value differs | IDs not actually seeding distinct streams |
| A3 | Delete blob id=3 from simulation, re-run → blob id=7's stream unchanged | Spawn independence violated (sequential spawn vs spawn_key) |
| A4 | Ledger: alternate adding `+1e15` and `-1e15` 1000x, then add `1.0` → `fsum == 1.0`; naive running sum ≠ 1.0 | Using `+=` accumulator instead of `math.fsum` on delta list |
| A5 | Construct `MappingProxyType(d)`, then mutate source dict `d` → proxy reflects mutation (known; verify `.create()` copies first) | Shallow wrap without copy — source dict mutation leaks through |
| A6 | `Action(MOVE, direction=(0,0))` — confirm system either rejects or handles as no-op | Silent identity move bypasses direction validation |

### Phase B: Grid & Environment (grid.py, environment.py)

4. `grid.py` — toroidal grid with generic layers + occupancy
5. `environment.py` — ABC + two implementations

**Validation — Phase B:**
| # | Check | What it catches |
|---|-------|-----------------|
| B1 | Grid size=10: move blob from `(0,0)` by `(-1,-1)` → position `(9,9)` | Negative modulo wrong (C-style `%` gives `-1`, Python gives `9`, but numpy int may differ) |
| B2 | `grid.remove_blob(pos)` then `grid.occupancy[pos] == -1` (not `0`, not `None`) | Sentinel value wrong — `0` is a valid blob_id |
| B3 | Place blob at `(3,3)`, `grid.move_blob(id, (3,3), (3,3))` → no crash, occupancy intact | Self-move clears then re-sets; intermediate `= -1` could leave cell empty if code short-circuits |
| B4 | RegeneratingEnvironment: set one cell to `capacity + 1.0`, call replenish → that cell unchanged, return excludes it | `np.maximum(space, 0.0)` missing → energy created from nothing |
| B5 | RegeneratingEnvironment: mark cell occupied, call replenish → that cell untouched | `unoccupied` mask wrong or not applied → occupied cells gain free energy |
| B6 | RegeneratingEnvironment: compare return value against `float((energy_after - energy_before).sum())` | Return value computed separately from mutation → desync breaks conservation |
| B7 | `grid.energy` property returns the actual layer array (not a copy) — mutate via property, verify `grid.layers["energy"]` changed | If it returns a copy, ABSORB writes to a throwaway array |

### Phase C: Blob & Rules (blob.py, rules.py)

6. `blob.py` — Blob, BlobAttributes with `.create()` factory, BlobStatus
7. `rules.py` — RulesEngine ABC

**Validation — Phase C:**
| # | Check | What it catches |
|---|-------|-----------------|
| C1 | `BlobAttributes.create(offspring_energy=0.05, base_metabolic_cost=0.1)` → raises | Missing validation: child born with less energy than first step's metabolism |
| C2 | `BlobAttributes.create(allowed_actions=frozenset({ActionType.MOVE}))` (no IDLE) → raises | Missing IDLE-mandatory check → downgrade fallback crashes |
| C3 | Pass `extra={"k": [1,2,3]}`, construct, mutate `extra["k"].append(4)` → `attributes.extra["k"]` still `[1,2,3]` | `MappingProxyType` prevents key add/delete but NOT value mutation of mutable contents |
| C4 | `Observation.self_status` — mutate it, verify blob's actual status unchanged | Aliased status object → rules engine can cheat by mutating observation |
| C5 | CellView for blob with `BASIC` sensing: `occupant_energy is None` even when neighbor has energy=5.0 | Sensing level filter not applied — all fields always populated |

### Phase D: Conflict Resolution (conflict.py)

8. `conflict.py` — ConflictResolver ABC + RandomResolver + `resolve_all()`

**Validation — Phase D:**
| # | Check | What it catches |
|---|-------|-----------------|
| D1 | 1 MOVE claim on an occupied cell → `succeeded=False` in result (not silently dropped) | Occupied-cell rejection drops blob from results → missing from CHARGE → energy leak |
| D2 | 3 claims on same empty cell, fixed seed, run 100x → same winner every time | RNG consumption not deterministic (dict iteration order, unsorted claims) |
| D3 | Claim on occupied `(5,3)` + claim on unoccupied `(5,4)` → first fails, second succeeds | Cell-level resolution leaking across cells |
| D4 | 0 claims → returns empty dict, no crash | Edge case: all blobs IDLE on first step |
| D5 | Blob A at `(2,2)` MOVEs to `(2,3)`, Blob B at `(5,5)` MOVEs to `(2,2)` → B **fails** | `occupied_cells` must include `(2,2)` (A's pre-move position) under conservative vacancy |
| D6 | Count result dict entries == count of input claim blob_ids (no drops, no dupes) | Missing/duplicate entries → CHARGE skips or double-charges blobs |

### Phase E: Engine Core (engine.py)

9. `engine.py` — 10-phase step loop + conservation/invariant checks

**Validation — Phase E:**
| # | Check | What it catches |
|---|-------|-----------------|
| E1 | 1 blob, IDLE only, DecayEnvironment, 1 step. Hand-compute: `energy_after = initial - base_metabolic - action_costs[IDLE] + min(grid_cell, max_energy - energy_post_charge)`. Compare exact float. | Phase ordering wrong (ABSORB before CHARGE), or IDLE action cost not applied |
| E2 | Successful REPRODUCE: `parent_before == parent_after + child.energy + base_metabolic + action_costs[REPRODUCE]`. No grid energy change from parent. | offspring_energy deducted in wrong phase, or action cost skipped |
| E3 | Failed REPRODUCE (no empty neighbor): blob pays `base_metabolic + action_costs[REPRODUCE]`, does **NOT** pay `offspring_energy` | Failed reproduce charged offspring_energy → over-dissipation |
| E4 | Blob with `energy=0.05, base_metabolic=0.1`, grid cell energy=1.0 → after CHARGE: `-0.05`, after ABSORB: recovers, survives DEATH CHECK | Clamping energy to 0 in CHARGE breaks conservation. Or DEATH CHECK before ABSORB kills recoverable blob. |
| E5 | Newborn: charged `base_metabolic` only (action_cost=0), DOES absorb from grid, does NOT appear in DECIDE phase output | Newborn charged full action cost, or skipped in ABSORB, or included in DECIDE |
| E6 | Blob MOVE invalidated in VALIDATE (bad direction) → downgraded to IDLE. Charged `base_metabolic + action_costs[IDLE]`, NOT `base_metabolic + action_costs[MOVE]`. | Downgraded action retains original type's cost |
| E7 | Blob MOVE loses in RESOLVE (conflict). Charged `base_metabolic + action_costs[MOVE]` (original cost, NOT IDLE). | Failed-in-RESOLVE confused with downgraded-in-VALIDATE |
| E8 | Run conservation check under `python -O`. Verify it still raises on violation. | `assert` stripped by `-O` → check silently disabled |

### Phase F: Observability & API (observability.py, config.py, simulation.py)

10. `observability.py` — StateRecorder + queries + `to_dataframe()`
11. `config.py` — SimulationConfig, SpeciesConfig, validation preconditions
12. `simulation.py` — Simulation (user-facing: run, iterate, result)

**Validation — Phase F:**
| # | Check | What it catches |
|---|-------|-----------------|
| F1 | `record_interval=5`: `blob_records` exist for step 3, `grid_records` do NOT exist for step 3 | Interval applied to wrong record type |
| F2 | `conservation_residuals()[0] == 0.0` exactly | Initial total computed wrong → non-zero residual from step 0 |
| F3 | `to_dataframes()["blobs"]` contains rows where `alive=False` | Dead blobs filtered out → lifespan/death queries break |
| F4 | Blob dies at step 5. Its `BlobRecord` at step 5: `alive=False`, `energy ≤ DEATH_THRESHOLD` | Death-step record missing or has stale pre-death energy |
| F5 | `Simulation.run(steps=0)` → returns result with 0 steps, no crash | Off-by-one in range loop or uninitialized recorder |
| F6 | Config with `sum(counts) == grid_size²` → valid. `sum(counts) == grid_size² + 1` → raises. | Off-by-one in `<=` vs `<` boundary |
| F7 | `iterate()` first yield: `WorldState.step == 0` (not `1`) | Step counter incremented before yield |

### Phase G: Species Implementations (species/)

13. `species/random_walker.py` — first species
14. `species/grazer.py` — second species

**Validation — Phase G:**
| # | Check | What it catches |
|---|-------|-----------------|
| G1 | RandomWalker: all 8 neighbors occupied, energy above reproduction threshold → returns IDLE | Rules engine doesn't check for empty neighbors → constant wasted REPRODUCE attempts |
| G2 | GrazerRules: two visible cells with energy 5.0 vs 1.0 → moves toward 5.0 deterministically | Grazer using random selection instead of argmax |
| G3 | GrazerRules: highest-energy cell is occupied → does NOT target it; picks next-best or IDLE | Grazer ignores occupancy → constant conflict failures |
| G4 | Both species on 3×3 grid, 9 blobs, all below reproduction threshold → no crash, all IDLE | Degenerate: no empty cells, no valid moves |
| G5 | RandomWalker with `allowed_directions=VON_NEUMANN` → never outputs diagonal `(1,1)` | Rules engine ignoring `allowed_directions`, using hardcoded MOORE |

### Phase H: Integration Tests

15. Tests (conservation, reproducibility, one-per-cell, decay, ledger, reproduction)

---

## Appendix A: Blob-Blob Interactions (v2 Design Notes)

Blob-blob interactions (predation, cooperation, mating) are implemented as **actions through the existing claim system**, not as a separate phase.

### Design

New action types (e.g., `ATTACK`) target occupied cells. The rules engine chooses ATTACK when it senses an adjacent prey (requires `sensing_level >= BASIC`). The engine generates a Claim on the target cell. The resolver sees:

- Attacker's claim (with `priority` set by engine based on blob attributes)
- Defender's implicit stay-claim (with its own `priority`)

The resolver determines the outcome. If attacker wins:
1. EXECUTE: prey dies, `predator.energy += prey.energy × efficiency`
2. `E_dissipated += prey.energy × (1 - efficiency)`
3. Predator optionally moves into the vacated cell

Energy accounting remains zero-sum: `ΔE_blobs = +efficiency×E_prey - E_prey = -(1-efficiency)×E_prey`, and `ΔE_dissipated = +(1-efficiency)×E_prey`.

### What's already in place for this
- `Claim.priority` field (default 0.0)
- `ConflictResolver` ABC (pluggable resolution policy)
- `SensingLevel.ENERGY` (predators can assess prey strength)
- Conservative vacancy model is a resolver policy, not hardcoded

### What v2 would add
- New `ActionType` values (ATTACK, SHARE)
- A `PredationResolver` that allows claims on occupied cells
- Defender stay-claim synthesis: engine generates an implicit `Claim` for the defender on its own cell, with priority derived from defender's attributes, so the resolver can compare attacker vs. defender
- Interaction energy transfer logic in EXECUTE
- Interaction efficiency parameter in `BlobAttributes.extra`

---

## Appendix B: Config Validation Preconditions

Checked in `Simulation.__init__()`. All raise `ConfigError` with descriptive message.

| Precondition | Rationale |
|---|---|
| `sum(s.count for s in species) <= grid_size²` | Prevents INV-3 violation at step 0 |
| `ActionType.IDLE in s.attributes.allowed_actions` for all species | IDLE is mandatory fallback (§8) |
| `s.attributes.offspring_energy <= s.attributes.max_energy` for all species | Child not born above energy cap |
| `s.attributes.offspring_energy > s.attributes.base_metabolic_cost` | Child survives at least one step (INV-5) |
| `s.initial_energy > s.attributes.base_metabolic_cost` | Initial blobs survive at least one step |
| `s.initial_energy <= s.attributes.max_energy` | Initial blobs not born above energy cap |
| `s.attributes.base_metabolic_cost >= 0` | Non-negative metabolism |
| `s.attributes.reproduction_threshold >= offspring_energy + base_metabolic + action_costs[REPRODUCE]` | Parent can afford reproduction |
| `action_costs` has entry for every type in `allowed_actions` | Prevents KeyError during affordability check |
| `s.attributes.action_costs[IDLE] == 0` | IDLE costs nothing beyond metabolism |
| `s.attributes.observation_radius >= 1` | Blob must see at least immediate neighbors |
| `s.attributes.allowed_directions` is non-empty (if MOVE in allowed_actions) | MOVE requires at least one legal direction |
| All `(dx,dy)` in `allowed_directions`: `abs(dx) <= 1` and `abs(dy) <= 1` | MVP restricts to adjacent moves; v2 may allow longer-range |
| `"energy"` layer in `environment.initial_grid()` | Required layer for absorption |
| `grid_size >= 1` | Non-degenerate grid |
| `seed` is `int` | Type check for reproducibility |
| Unique species names across all `SpeciesConfig` | Engine uses species name for factory lookup |
