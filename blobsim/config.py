"""Simulation configuration and validation.

Spec reference: tech_spec_mvp.md section 14 (User API),
Appendix B (Config Validation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from blobsim.conflict import RandomResolver
from blobsim.ledger import ConfigError
from blobsim.types import ActionType

if TYPE_CHECKING:
    from blobsim.blob import BlobAttributes
    from blobsim.conflict import ConflictResolver
    from blobsim.environment import EnvironmentFn
    from blobsim.rules import RulesEngine


@dataclass
class SpeciesConfig:
    """Per-species initialization parameters."""

    rules_factory: Callable[[], RulesEngine]
    attributes: BlobAttributes
    count: int
    initial_energy: float


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    grid_size: int
    environment: EnvironmentFn
    species: list[SpeciesConfig]
    seed: int
    conflict_resolver: ConflictResolver = field(default_factory=RandomResolver)
    record_interval: int = 1


def validate_config(config: SimulationConfig) -> None:
    """Check all Appendix B preconditions. Raises ConfigError on failure.

    Args:
        config: SimulationConfig to validate.

    Raises:
        ConfigError: If any precondition is violated.
    """
    # B15: grid_size >= 1
    if config.grid_size < 1:
        raise ConfigError(
            f"grid_size must be >= 1, got {config.grid_size}"
        )

    # B16: seed is int
    if not isinstance(config.seed, int):
        raise ConfigError(
            f"seed must be int, got {type(config.seed).__name__}"
        )

    # B1: total blob count <= grid_size^2
    total_blobs = sum(s.count for s in config.species)
    max_cells = config.grid_size ** 2
    if total_blobs > max_cells:
        raise ConfigError(
            f"Total blob count ({total_blobs}) exceeds "
            f"grid capacity ({max_cells})"
        )

    # B14: "energy" layer in environment.initial_grid()
    layers = config.environment.initial_grid(config.grid_size)
    if "energy" not in layers:
        raise ConfigError(
            "environment.initial_grid() must include 'energy' layer"
        )

    # B17: unique species names
    species_names = [s.attributes.species for s in config.species]
    if len(species_names) != len(set(species_names)):
        seen: set[str] = set()
        dupes: list[str] = []
        for name in species_names:
            if name in seen:
                dupes.append(name)
            seen.add(name)
        raise ConfigError(
            f"Duplicate species names: {dupes}"
        )

    # Per-species validations
    for sc in config.species:
        attrs = sc.attributes
        sp = attrs.species

        # B2: IDLE in allowed_actions
        if ActionType.IDLE not in attrs.allowed_actions:
            raise ConfigError(
                f"Species '{sp}': ActionType.IDLE must be in allowed_actions"
            )

        # B3: offspring_energy <= max_energy
        if attrs.offspring_energy > attrs.max_energy:
            raise ConfigError(
                f"Species '{sp}': offspring_energy ({attrs.offspring_energy}) "
                f"> max_energy ({attrs.max_energy})"
            )

        # B4: offspring_energy > base_metabolic_cost
        if attrs.offspring_energy <= attrs.base_metabolic_cost:
            raise ConfigError(
                f"Species '{sp}': offspring_energy "
                f"({attrs.offspring_energy}) must exceed "
                f"base_metabolic_cost ({attrs.base_metabolic_cost})"
            )

        # B5: initial_energy > base_metabolic_cost
        if sc.initial_energy <= attrs.base_metabolic_cost:
            raise ConfigError(
                f"Species '{sp}': initial_energy "
                f"({sc.initial_energy}) must exceed "
                f"base_metabolic_cost ({attrs.base_metabolic_cost})"
            )

        # B6: initial_energy <= max_energy
        if sc.initial_energy > attrs.max_energy:
            raise ConfigError(
                f"Species '{sp}': initial_energy ({sc.initial_energy}) "
                f"> max_energy ({attrs.max_energy})"
            )

        # B7: base_metabolic_cost >= 0
        if attrs.base_metabolic_cost < 0:
            raise ConfigError(
                f"Species '{sp}': base_metabolic_cost "
                f"({attrs.base_metabolic_cost}) must be >= 0"
            )

        # B8: repro threshold >= offspring + metabolic + cost
        if ActionType.REPRODUCE in attrs.allowed_actions:
            min_threshold = (
                attrs.offspring_energy
                + attrs.base_metabolic_cost
                + attrs.action_costs[ActionType.REPRODUCE]
            )
            if attrs.reproduction_threshold < min_threshold:
                raise ConfigError(
                    f"Species '{sp}': reproduction_threshold "
                    f"({attrs.reproduction_threshold}) must be >= "
                    f"offspring_energy + base_metabolic + reproduce_cost "
                    f"({min_threshold})"
                )

        # B9: action_costs has entry for every type in allowed_actions
        for action_type in attrs.allowed_actions:
            if action_type not in attrs.action_costs:
                raise ConfigError(
                    f"Species '{sp}': action_costs missing entry "
                    f"for {action_type}"
                )

        # B10: IDLE costs nothing
        if attrs.action_costs[ActionType.IDLE] != 0:
            raise ConfigError(
                f"Species '{sp}': action_costs[IDLE] must be 0, "
                f"got {attrs.action_costs[ActionType.IDLE]}"
            )

        # B11: observation_radius >= 1
        if attrs.observation_radius < 1:
            raise ConfigError(
                f"Species '{sp}': observation_radius must be >= 1, "
                f"got {attrs.observation_radius}"
            )

        # B12: allowed_directions non-empty (if MOVE in allowed_actions)
        if ActionType.MOVE in attrs.allowed_actions:
            if not attrs.allowed_directions:
                raise ConfigError(
                    f"Species '{sp}': allowed_directions must be "
                    f"non-empty when MOVE is allowed"
                )

        # B13: all directions within abs(dx)<=1, abs(dy)<=1
        for dx, dy in attrs.allowed_directions:
            if abs(dx) > 1 or abs(dy) > 1:
                raise ConfigError(
                    f"Species '{sp}': direction ({dx}, {dy}) exceeds "
                    f"adjacent range (abs(dx) <= 1, abs(dy) <= 1)"
                )
