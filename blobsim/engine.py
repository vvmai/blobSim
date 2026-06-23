"""10-phase simulation engine for the blob simulation.

Spec reference: tech_spec_mvp.md sections 8 (Action System),
9 (Simulation Engine), 10 (Conflict Resolution), 11 (Energy Accounting).
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from blobsim.blob import Blob, BlobStatus
from blobsim.conflict import resolve_all
from blobsim.grid import Grid
from blobsim.ledger import (
    ConfigError,
    EnergyLedger,
    InvariantError,
    RulesEngineError,
    check_conservation,
    check_one_per_cell,
)
from blobsim.rng import make_blob_rng
from blobsim.types import (
    Action,
    ActionType,
    CellView,
    Claim,
    Observation,
    ResolvedAction,
    SensingLevel,
    StepData,
)

if TYPE_CHECKING:
    from blobsim.config import SpeciesConfig
    from blobsim.conflict import ConflictResolver
    from blobsim.environment import EnvironmentFn
    from blobsim.observability import StateRecorder

# Blobs at or below this energy are dead. Distinct from conservation
# tolerance (same value, different purpose). See spec Phase 9.
DEATH_THRESHOLD: float = 1e-10


class SimulationEngine:
    """Core simulation loop executing the 10-phase step cycle.

    Owns the mutable blob list, delegates to grid/ledger/resolver/recorder.
    """

    def __init__(
        self,
        grid: Grid,
        blobs: list[Blob],
        environment: EnvironmentFn,
        conflict_resolver: ConflictResolver,
        ledger: EnergyLedger,
        engine_rng: np.random.Generator,
        env_rng: np.random.Generator,
        blob_pool_entropy: int,
        species_configs: dict[str, SpeciesConfig],
        recorder: StateRecorder,
    ) -> None:
        self.grid = grid
        self.blobs = blobs
        self.environment = environment
        self.resolver = conflict_resolver
        self.ledger = ledger
        self.engine_rng = engine_rng
        self.env_rng = env_rng
        self.blob_pool_entropy = blob_pool_entropy
        self.species_configs = species_configs
        self.recorder = recorder
        self._next_blob_id: int = (
            max(b.id for b in blobs) + 1 if blobs else 0
        )
        self._blob_index: dict[int, Blob] = {b.id: b for b in blobs}
        self._step: int = 0

        # FLAG-7: Validate that every prey name in preys_on is a known species
        known_species = set(species_configs.keys())
        for blob in blobs:
            for prey_name in blob.attributes.preys_on:
                if prey_name not in known_species:
                    raise ConfigError(
                        f"Species '{blob.attributes.species}' preys_on "
                        f"'{prey_name}' which is not a known species. "
                        f"Known: {sorted(known_species)}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> tuple[int, int]:
        """Execute one full simulation step.

        Returns:
            (n_births, n_deaths) for this step.
            n_deaths includes both predation deaths and starvation deaths.
        """
        # Phase 1: OBSERVE
        observations = self._phase_observe()

        # Phase 2: DECIDE
        actions = self._phase_decide(observations)

        # Phase 3: VALIDATE
        actions = self._phase_validate(actions)

        # Phase 4: CLAIM
        claims_by_cell, attack_claims_by_prey, occupied_cells = (
            self._phase_claim(actions)
        )

        # Phase 5: RESOLVE
        resolved, step_data = self._phase_resolve(
            claims_by_cell, attack_claims_by_prey, occupied_cells,
        )

        # Capture pre-EXECUTE snapshot before ATTACK kills remove blobs
        # (FLAG-1: predation deaths must appear in all_step_blobs for recorder)
        pre_execute_blobs = list(self.blobs)

        # Phase 6: EXECUTE — ATTACK-first ordering
        newborns, n_predation_deaths = self._phase_execute(resolved, step_data)
        n_births = len(newborns)

        # Phase 7: CHARGE
        self._phase_charge(resolved)

        # Phase 8: ABSORB
        self._phase_absorb(step_data)

        # Build full step blob list: pre-EXECUTE blobs + newborns.
        # Includes predation-killed blobs (alive=False set in EXECUTE)
        # and blobs about to be removed by DEATH_CHECK.
        all_step_blobs = pre_execute_blobs + newborns

        # Phase 9: DEATH CHECK
        n_starvation_deaths = self._phase_death_check()
        n_deaths = n_predation_deaths + n_starvation_deaths

        # Phase 10: ENVIRONMENT
        self._phase_environment()

        # POST-STEP: RECORD + VALIDATE
        self._post_step(all_step_blobs, step_data, n_births, n_deaths)

        return n_births, n_deaths

    # ------------------------------------------------------------------
    # Phase 1: OBSERVE
    # ------------------------------------------------------------------

    def _phase_observe(self) -> dict[int, Observation]:
        """Build frozen Observation for each alive blob."""
        observations: dict[int, Observation] = {}
        for blob in self.blobs:
            observations[blob.id] = self._build_observation(blob)
        return observations

    def _build_observation(self, blob: Blob) -> Observation:
        """Construct Observation for a single blob."""
        pos = blob.status.position
        radius = blob.attributes.observation_radius
        sensing = blob.attributes.sensing_level

        # Current cell (offset=(0,0), occupied=True, self as occupant)
        current_cell = self._make_cell_view(
            pos, (0, 0), sensing, self_species=blob.attributes.species,
        )

        # Neighborhood: all cells within Chebyshev distance <= radius, excl self
        neighbors: list[CellView] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                abs_pos = self.grid.wrap(pos[0] + dx, pos[1] + dy)
                cell_view = self._make_cell_view(abs_pos, (dx, dy), sensing)
                neighbors.append(cell_view)

        return Observation(
            self_status=dataclasses.replace(blob.status),
            self_attributes=blob.attributes,
            neighborhood=tuple(neighbors),
            current_cell=current_cell,
            step=self._step,
        )

    def _make_cell_view(
        self,
        abs_pos: tuple[int, int],
        offset: tuple[int, int],
        sensing: SensingLevel,
        self_species: str | None = None,
    ) -> CellView:
        """Build a CellView for a cell at abs_pos with given offset and sensing.

        Args:
            abs_pos: Absolute grid coordinate.
            offset: Relative offset from observer.
            sensing: Observer's sensing level.
            self_species: If set, treat as self-cell (occupied=True, this species).
        """
        layers = MappingProxyType({
            name: float(layer[abs_pos[0], abs_pos[1]])
            for name, layer in self.grid.layers.items()
        })

        occupant_id = int(self.grid.occupancy[abs_pos[0], abs_pos[1]])

        # Determine occupancy and species
        if self_species is not None:
            occupied = True
            occupant_species: str | None = self_species
        elif occupant_id >= 0:
            occupied = True
            occupant_blob = self._find_blob(occupant_id)
            occupant_species = (
                occupant_blob.attributes.species if occupant_blob else None
            )
        else:
            occupied = False
            occupant_species = None

        # Populate energy/age fields based on sensing level
        occupant_energy: float | None = None
        occupant_age: int | None = None

        if occupied and occupant_id >= 0:
            target_blob = self._find_blob(occupant_id)
            if target_blob is not None:
                if sensing in (SensingLevel.ENERGY, SensingLevel.FULL):
                    occupant_energy = target_blob.status.energy
                if sensing == SensingLevel.FULL:
                    occupant_age = target_blob.status.age

        return CellView(
            offset=offset,
            layers=layers,
            occupied=occupied,
            occupant_species=occupant_species,
            occupant_energy=occupant_energy,
            occupant_age=occupant_age,
        )

    def _find_blob(self, blob_id: int) -> Blob | None:
        """Look up blob by id. O(1) via index dict."""
        return self._blob_index.get(blob_id)

    # ------------------------------------------------------------------
    # Phase 2: DECIDE
    # ------------------------------------------------------------------

    def _phase_decide(
        self, observations: dict[int, Observation],
    ) -> dict[int, Action]:
        """Each blob decides its action from its observation."""
        actions: dict[int, Action] = {}
        for blob in self.blobs:
            actions[blob.id] = blob.rules.decide(
                observations[blob.id], blob.rng,
            )
        return actions

    # ------------------------------------------------------------------
    # Phase 3: VALIDATE
    # ------------------------------------------------------------------

    def _phase_validate(self, actions: dict[int, Action]) -> dict[int, Action]:
        """Enforce rules engine contract + affordability downgrade."""
        validated: dict[int, Action] = {}
        for blob in self.blobs:
            action = actions[blob.id]

            # Contract check 1: action type must be allowed
            if action.type not in blob.attributes.allowed_actions:
                raise RulesEngineError(
                    f"Blob {blob.id}: action {action.type} "
                    f"not in allowed_actions"
                )

            # Contract check 2: MOVE direction must be allowed
            if (
                action.type == ActionType.MOVE
                and action.direction not in blob.attributes.allowed_directions
            ):
                raise RulesEngineError(
                    f"Blob {blob.id}: direction {action.direction} "
                    f"not in allowed_directions"
                )

            # Contract check 3: ATTACK requires a direction in allowed_directions
            if action.type == ActionType.ATTACK and action.direction is None:
                raise RulesEngineError(
                    f"Blob {blob.id}: ATTACK requires a direction"
                )
            if (
                action.type == ActionType.ATTACK
                and action.direction not in blob.attributes.allowed_directions
            ):
                raise RulesEngineError(
                    f"Blob {blob.id}: ATTACK direction {action.direction} "
                    f"not in allowed_directions"
                )

            # Game mechanic: affordability gate (one-shot downgrade)
            if not self._can_afford(blob, action):
                action = Action(ActionType.IDLE)

            validated[blob.id] = action
        return validated

    def _can_afford(self, blob: Blob, action: Action) -> bool:
        """Gate check: can blob afford this action if it succeeds?

        Worst-case cost includes offspring_energy for REPRODUCE.
        """
        cost = (
            blob.attributes.base_metabolic_cost
            + blob.attributes.action_costs[action.type]
        )
        if action.type == ActionType.REPRODUCE:
            cost += blob.attributes.offspring_energy
        return blob.status.energy >= cost

    # ------------------------------------------------------------------
    # Phase 4: CLAIM
    # ------------------------------------------------------------------

    def _phase_claim(
        self, actions: dict[int, Action],
    ) -> tuple[
        dict[tuple[int, int], list[Claim]],
        dict[tuple[int, int], list[Claim]],
        set[tuple[int, int]],
    ]:
        """Convert validated actions to cell claims.

        ATTACK claims are collected into a separate channel
        (attack_claims_by_prey) to keep them out of resolve_all,
        which rejects all occupied-cell claims (FLAG-5).

        Returns:
            (claims_by_cell, attack_claims_by_prey, occupied_cells)
        """
        claims_by_cell: dict[tuple[int, int], list[Claim]] = defaultdict(list)
        attack_claims_by_prey: dict[tuple[int, int], list[Claim]] = defaultdict(list)

        for blob in self.blobs:
            action = actions[blob.id]
            pos = blob.status.position

            if action.type == ActionType.MOVE:
                dx, dy = action.direction  # type: ignore[misc]
                target = self.grid.wrap(pos[0] + dx, pos[1] + dy)
                claims_by_cell[target].append(
                    Claim(blob.id, target, action),
                )

            elif action.type == ActionType.REPRODUCE:
                target = self._pick_reproduce_target(blob)
                claims_by_cell[target].append(
                    Claim(blob.id, target, action),
                )

            elif action.type == ActionType.ATTACK:
                dx, dy = action.direction  # type: ignore[misc]
                target = self.grid.wrap(pos[0] + dx, pos[1] + dy)
                target_id = int(self.grid.occupancy[target[0], target[1]])
                if target_id < 0:
                    # Prey has vacated (defensive guard — should not occur
                    # within a single step since grid is immutable between
                    # OBSERVE and CLAIM). Treat as failed: emit no claim.
                    pass
                else:
                    target_blob = self._find_blob(target_id)
                    if (
                        target_blob is None
                        or target_blob.attributes.species
                        not in blob.attributes.preys_on
                    ):
                        raise RulesEngineError(
                            f"Blob {blob.id}: ATTACK targets non-prey "
                            f"species at {target}"
                        )
                    attack_claims_by_prey[target].append(
                        Claim(blob.id, target, action),
                    )
            # IDLE: no claim

        occupied_cells = {b.status.position for b in self.blobs}
        return dict(claims_by_cell), dict(attack_claims_by_prey), occupied_cells

    def _pick_reproduce_target(self, blob: Blob) -> tuple[int, int]:
        """Pick random empty neighbor cell for reproduction.

        Raises:
            RulesEngineError: If no empty neighbor exists in allowed_directions.
        """
        pos = blob.status.position
        candidates: list[tuple[int, int]] = []

        for dx, dy in sorted(blob.attributes.allowed_directions):
            abs_pos = self.grid.wrap(pos[0] + dx, pos[1] + dy)
            if self.grid.occupancy[abs_pos[0], abs_pos[1]] < 0:
                candidates.append(abs_pos)

        if not candidates:
            raise RulesEngineError(
                f"Blob {blob.id}: REPRODUCE with no empty neighbor"
            )

        idx = blob.rng.integers(len(candidates))
        return candidates[idx]

    # ------------------------------------------------------------------
    # Phase 5: RESOLVE
    # ------------------------------------------------------------------

    def _phase_resolve(
        self,
        claims_by_cell: dict[tuple[int, int], list[Claim]],
        attack_claims_by_prey: dict[tuple[int, int], list[Claim]],
        occupied_cells: set[tuple[int, int]],
    ) -> tuple[dict[int, ResolvedAction], dict[int, StepData]]:
        """Resolve all claims; synthesize IDLE for non-claiming blobs.

        MOVE/REPRODUCE claims go through resolve_all (vacant-cell model).
        ATTACK claims are resolved separately via _resolve_attack_claims,
        after resolve_all, so engine_rng is consumed in a deterministic
        sorted order: MOVE/REPRODUCE cells first, then ATTACK cells (INV-2).

        Returns:
            (resolved_actions, step_data) where step_data is initialized
            with action_taken/action_succeeded/action_cost from resolution.
        """
        resolved = resolve_all(
            claims_by_cell, occupied_cells, self.resolver, self.engine_rng,
        )

        # Resolve ATTACK claims after MOVE/REPRODUCE for determinism (INV-2).
        # ATTACK targets are occupied cells — disjoint from resolve_all targets.
        attack_resolved = self._resolve_attack_claims(attack_claims_by_prey)
        resolved.update(attack_resolved)

        # Synthesize IDLE ResolvedAction for blobs not in results
        for blob in self.blobs:
            if blob.id not in resolved:
                resolved[blob.id] = ResolvedAction(
                    blob.id, Action(ActionType.IDLE),
                    target=None, succeeded=True,
                )

        # Initialize step_data from resolution results
        step_data: dict[int, StepData] = {}
        for blob in self.blobs:
            ra = resolved[blob.id]
            step_data[blob.id] = StepData(
                action_taken=ra.action.type,
                action_succeeded=ra.succeeded,
                action_cost=blob.attributes.action_costs[ra.action.type],
                energy_absorbed=0.0,
            )

        return resolved, step_data

    def _resolve_attack_claims(
        self,
        attack_claims_by_prey: dict[tuple[int, int], list[Claim]],
    ) -> dict[int, ResolvedAction]:
        """Resolve competing ATTACK claims on each prey cell.

        Iterates sorted prey positions for determinism (INV-2).
        Uses same resolver + engine_rng as resolve_all, consumed after it.

        Args:
            attack_claims_by_prey: Mapping from prey cell to list of
                ATTACK claims targeting that cell.

        Returns:
            Mapping from blob_id to ResolvedAction for every attacker.
        """
        attack_resolved: dict[int, ResolvedAction] = {}

        for prey_pos in sorted(attack_claims_by_prey.keys()):
            claims = attack_claims_by_prey[prey_pos]
            if len(claims) == 1:
                winner = claims[0]
            else:
                winner = self.resolver.resolve(claims, self.engine_rng)

            for c in claims:
                succeeded = c.blob_id == winner.blob_id
                attack_resolved[c.blob_id] = ResolvedAction(
                    c.blob_id, c.action, prey_pos, succeeded=succeeded,
                )

        return attack_resolved

    # ------------------------------------------------------------------
    # Phase 6: EXECUTE
    # ------------------------------------------------------------------

    def _phase_execute(
        self,
        resolved: dict[int, ResolvedAction],
        step_data: dict[int, StepData],
    ) -> tuple[list[Blob], int]:
        """Execute resolved actions. Returns (newborns, n_predation_deaths).

        Ordering: ATTACK kills first (so prey cannot dodge), then
        MOVE/REPRODUCE for surviving blobs only (§8.2 Option A).
        """
        # --- ATTACK-first: kill prey before MOVE reshuffles the grid ---
        # Iterate in sorted prey-cell order (matching _resolve_attack_claims
        # determinism, INV-2) so that mutual-predation (A↔B) resolves
        # consistently: the predator whose prey cell sorts smallest fires first,
        # removing that prey from the grid; the other predator then finds its
        # prey's cell empty and skips (§8.2, §12.2).
        blob_index = {b.id: b for b in self.blobs}
        winning_attacks: list[tuple[tuple[int, int], int]] = []  # (prey_pos, blob_id)
        for blob in self.blobs:
            ra = resolved.get(blob.id)
            if ra is not None and ra.succeeded and ra.action.type == ActionType.ATTACK:
                winning_attacks.append((ra.target, blob.id))  # type: ignore[arg-type]

        # Sort by prey position for determinism (INV-2)
        winning_attacks.sort(key=lambda x: x[0])

        killed_ids: set[int] = set()
        n_predation_deaths = 0

        for prey_pos, attacker_id in winning_attacks:
            # Skip if this attacker was itself killed earlier in this loop
            # (mutual-predation: A kills B, then B's attack on A is skipped)
            if attacker_id in killed_ids:
                continue

            blob = blob_index.get(attacker_id)
            if blob is None:
                continue

            # re-read occupancy: prey may already be gone (mutual-attack case)
            prey_id = int(self.grid.occupancy[prey_pos[0], prey_pos[1]])
            if prey_id < 0:
                # Prey already killed by an earlier attacker (sorted-cell order)
                continue
            prey = self._find_blob(prey_id)
            if prey is None:
                continue

            # Energy transfer: predator gains min(prey.energy, headroom)
            headroom = blob.attributes.max_energy - blob.status.energy
            transfer = min(prey.status.energy, max(0.0, headroom))
            dissipate = prey.status.energy - transfer

            # Assert conservation before zeroing prey energy (§9)
            assert abs(transfer + dissipate - prey.status.energy) < 1e-10, (
                f"Attack energy accounting error: transfer={transfer}, "
                f"dissipate={dissipate}, prey.energy={prey.status.energy}"
            )

            blob.status.energy += transfer
            self.ledger.add_dissipated(dissipate)

            if attacker_id in step_data:
                step_data[attacker_id].energy_gained_from_predation += transfer

            # Kill prey immediately (grid.remove_blob so subsequent mutual
            # attacks find the cell empty)
            prey.status.energy = 0.0
            prey.status.alive = False
            self.grid.remove_blob(prey.status.position)
            killed_ids.add(prey_id)
            n_predation_deaths += 1

        # Remove killed prey from blobs list and index
        if killed_ids:
            self.blobs = [b for b in self.blobs if b.id not in killed_ids]
            for kid in killed_ids:
                del self._blob_index[kid]

        # --- MOVE and REPRODUCE for surviving blobs ---
        moves: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        births: list[tuple[int, tuple[int, int]]] = []
        newborns: list[Blob] = []

        for blob in self.blobs:
            ra = resolved[blob.id]
            if not ra.succeeded:
                continue

            if ra.action.type == ActionType.MOVE:
                old_pos = blob.status.position
                blob.status.position = ra.target  # type: ignore[assignment]
                moves.append((blob.id, old_pos, ra.target))  # type: ignore[arg-type]

            elif ra.action.type == ActionType.REPRODUCE:
                # Deduct offspring_energy from parent (EXECUTE, not CHARGE)
                blob.status.energy -= blob.attributes.offspring_energy

                child_id = self._next_blob_id
                self._next_blob_id += 1

                child = Blob(
                    blob_id=child_id,
                    attributes=blob.attributes,
                    status=BlobStatus(
                        energy=blob.attributes.offspring_energy,
                        age=0,
                        position=ra.target,  # type: ignore[arg-type]
                    ),
                    rules=self.species_configs[
                        blob.attributes.species
                    ].rules_factory(),
                    rng=make_blob_rng(self.blob_pool_entropy, child_id),
                )
                newborns.append(child)
                births.append((child_id, ra.target))  # type: ignore[arg-type]

                # StepData for newborn: IDLE, succeeded, 0 cost
                step_data[child_id] = StepData(
                    action_taken=ActionType.IDLE,
                    action_succeeded=True,
                    action_cost=0.0,
                    energy_absorbed=0.0,
                )

        # Add newborns to blob list and index
        self.blobs.extend(newborns)
        for nb in newborns:
            self._blob_index[nb.id] = nb

        # Batch occupancy update: ALL removals first, THEN all placements
        for _, old_pos, _ in moves:
            self.grid.remove_blob(old_pos)
        for blob_id, _, new_pos in moves:
            self.grid.place_blob(blob_id, new_pos)
        for child_id, pos in births:
            self.grid.place_blob(child_id, pos)

        return newborns, n_predation_deaths

    # ------------------------------------------------------------------
    # Phase 7: CHARGE
    # ------------------------------------------------------------------

    def _phase_charge(
        self,
        resolved: dict[int, ResolvedAction],
    ) -> None:
        """Deduct metabolism + action costs from all alive blobs."""
        for blob in self.blobs:
            if blob.id not in resolved:
                # Newborn: base metabolic only
                cost = blob.attributes.base_metabolic_cost
            else:
                ra = resolved[blob.id]
                cost = (
                    blob.attributes.base_metabolic_cost
                    + blob.attributes.action_costs[ra.action.type]
                )
            blob.status.energy -= cost
            self.ledger.add_dissipated(cost)

    # ------------------------------------------------------------------
    # Phase 8: ABSORB
    # ------------------------------------------------------------------

    def _phase_absorb(self, step_data: dict[int, StepData]) -> None:
        """Each blob absorbs energy from its cell."""
        for blob in self.blobs:
            pos = blob.status.position
            headroom = blob.attributes.max_energy - blob.status.energy
            if headroom < 0:
                raise InvariantError(
                    f"blob {blob.id} energy {blob.status.energy} "
                    f"> max_energy {blob.attributes.max_energy}"
                )
            absorbed = min(
                float(self.grid.energy[pos[0], pos[1]]),
                headroom,
            )
            blob.status.energy += absorbed
            self.grid.energy[pos[0], pos[1]] -= absorbed
            step_data[blob.id].energy_absorbed = absorbed

    # ------------------------------------------------------------------
    # Phase 9: DEATH CHECK
    # ------------------------------------------------------------------

    def _phase_death_check(self) -> int:
        """Remove blobs at or below DEATH_THRESHOLD. Returns death count."""
        dead_blobs: list[Blob] = []
        for blob in self.blobs:
            if blob.status.energy <= DEATH_THRESHOLD:
                self.ledger.add_dissipated(blob.status.energy)
                blob.status.alive = False
                self.grid.remove_blob(blob.status.position)
                dead_blobs.append(blob)

        self.blobs = [b for b in self.blobs if b.status.alive]
        for db in dead_blobs:
            del self._blob_index[db.id]
        return len(dead_blobs)

    # ------------------------------------------------------------------
    # Phase 10: ENVIRONMENT
    # ------------------------------------------------------------------

    def _phase_environment(self) -> None:
        """Replenish grid energy via environment function."""
        injected = self.environment.replenish(
            self.grid, self._step, self.env_rng,
        )
        self.ledger.add_injected(injected)

    # ------------------------------------------------------------------
    # POST-STEP: RECORD + VALIDATE
    # ------------------------------------------------------------------

    def _post_step(
        self,
        all_step_blobs: list[Blob],
        step_data: dict[int, StepData],
        n_births: int,
        n_deaths: int,
    ) -> None:
        """Increment ages, record state, check invariants.

        Args:
            all_step_blobs: All blobs from this step (incl. newly dead
                with alive=False), captured before Phase 9 filtering.
            step_data: Per-blob accumulated data from phases 5-8.
            n_births: Number of newborns this step.
            n_deaths: Number of deaths this step.
        """
        # Age increment applies to surviving blobs only
        for blob in self.blobs:
            blob.status.age += 1

        # Recorder receives ALL blobs that participated this step
        self.recorder.snapshot(
            self._step, all_step_blobs, self.grid, self.ledger,
            step_data, n_births, n_deaths,
        )

        # Conservation check uses only alive blobs (self.blobs)
        check_conservation(self.blobs, self.grid, self.ledger)
        check_one_per_cell(self.grid)
        self._step += 1
