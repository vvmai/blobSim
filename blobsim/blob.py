"""Blob entity and supporting data structures.

Spec reference: tech_spec_mvp.md sections 3 (Data Model), 5 (Blob Architecture).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from blobsim.ledger import ConfigError
from blobsim.types import ActionType, SensingLevel

if TYPE_CHECKING:
    from blobsim.rules import RulesEngine


# ---------------------------------------------------------------------------
# BlobAttributes (frozen, identity-compared)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BlobAttributes:
    """Immutable species configuration set at creation, never changes.

    Uses identity comparison (``is``) because ``MappingProxyType`` fields
    are not hashable.  Construct via the ``.create()`` classmethod only.
    """

    species: str
    max_energy: float
    offspring_energy: float
    reproduction_threshold: float
    base_metabolic_cost: float
    action_costs: MappingProxyType  # ActionType -> float
    observation_radius: int
    sensing_level: SensingLevel
    allowed_actions: frozenset  # frozenset[ActionType]
    allowed_directions: frozenset  # frozenset[tuple[int, int]]
    extra: MappingProxyType  # str -> Any

    # -- Identity-based equality / hash (MappingProxyType not hashable) -----

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    # -- Factory (sole construction path) -----------------------------------

    @classmethod
    def create(
        cls,
        species: str,
        max_energy: float,
        offspring_energy: float,
        reproduction_threshold: float,
        base_metabolic_cost: float,
        move_cost: float,
        reproduce_cost: float,
        observation_radius: int,
        allowed_actions: frozenset[ActionType],
        allowed_directions: frozenset[tuple[int, int]],
        idle_cost: float = 0.0,
        sensing_level: SensingLevel = SensingLevel.BASIC,
        extra: dict[str, Any] | None = None,
    ) -> BlobAttributes:
        """Build ``BlobAttributes`` with validation.

        Wraps raw dicts into ``MappingProxyType`` for immutability.
        Deep-copies *extra* before wrapping to prevent source mutation
        from leaking into the frozen instance.

        Raises:
            ConfigError: If ``IDLE`` not in *allowed_actions* or
                *offspring_energy* <= *base_metabolic_cost*.
        """
        if ActionType.IDLE not in allowed_actions:
            raise ConfigError(
                "ActionType.IDLE must be in allowed_actions"
            )
        if offspring_energy <= base_metabolic_cost:
            raise ConfigError(
                f"offspring_energy ({offspring_energy}) must exceed "
                f"base_metabolic_cost ({base_metabolic_cost}) so a "
                f"newborn survives its first step"
            )

        action_costs = MappingProxyType({
            ActionType.IDLE: idle_cost,
            ActionType.MOVE: move_cost,
            ActionType.REPRODUCE: reproduce_cost,
        })

        safe_extra = MappingProxyType(copy.deepcopy(extra) if extra else {})

        return cls(
            species=species,
            max_energy=max_energy,
            offspring_energy=offspring_energy,
            reproduction_threshold=reproduction_threshold,
            base_metabolic_cost=base_metabolic_cost,
            action_costs=action_costs,
            observation_radius=observation_radius,
            sensing_level=sensing_level,
            allowed_actions=allowed_actions,
            allowed_directions=allowed_directions,
            extra=safe_extra,
        )


# ---------------------------------------------------------------------------
# BlobStatus (mutable per-step state)
# ---------------------------------------------------------------------------

@dataclass
class BlobStatus:
    """Mutable per-step blob state.

    ``age`` starts at 0 on birth step, incremented once in POST-STEP.
    """

    energy: float
    age: int
    position: tuple[int, int]
    alive: bool = True


# ---------------------------------------------------------------------------
# Blob (composition root)
# ---------------------------------------------------------------------------

class Blob:
    """Concrete blob entity -- not subclassed per species.

    Species differentiation is via ``BlobAttributes`` (costs, thresholds)
    and ``RulesEngine`` implementation (decision logic).
    """

    __slots__ = ("id", "attributes", "status", "rules", "rng")

    def __init__(
        self,
        blob_id: int,
        attributes: BlobAttributes,
        status: BlobStatus,
        rules: RulesEngine,
        rng: np.random.Generator,
    ) -> None:
        self.id = blob_id
        self.attributes = attributes
        self.status = status
        self.rules = rules
        self.rng = rng
