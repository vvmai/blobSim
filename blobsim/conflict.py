"""Conflict resolution for competing claims on grid cells.

Implements the conservative vacancy model (MVP resolver policy).
See tech_spec_mvp.md section 10 (Conflict Resolution).

Resolution rules:
- Claims on occupied cells -> ALL fail (conservative vacancy model)
- Single claim on empty cell -> succeeds
- Multiple claims on empty cell -> resolver picks winner, rest fail
- Iteration order is sorted by cell coordinate for deterministic RNG (INV-2)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from blobsim.types import Claim, ResolvedAction


class ConflictResolver(ABC):
    """Pick one winner from competing claims on a single cell."""

    @abstractmethod
    def resolve(
        self, claims: list[Claim], rng: np.random.Generator,
    ) -> Claim:
        """Return winning claim using *rng* for tie-breaking."""
        ...


class RandomResolver(ConflictResolver):
    """Uniform-random conflict resolution (MVP default)."""

    def resolve(self, claims: list[Claim], rng: np.random.Generator) -> Claim:
        """Pick a winner uniformly at random."""
        return claims[rng.integers(len(claims))]


def resolve_all(
    claims_by_cell: dict[tuple[int, int], list[Claim]],
    occupied_cells: set[tuple[int, int]],
    resolver: ConflictResolver,
    rng: np.random.Generator,
) -> dict[int, ResolvedAction]:
    """Resolve all claims and return one ResolvedAction per claiming blob.

    Parameters
    ----------
    claims_by_cell:
        Mapping from target cell to list of claims on that cell.
    occupied_cells:
        Set of ALL current blob positions (built by engine Phase 4).
    resolver:
        Pluggable strategy for breaking ties on contested empty cells.
    rng:
        Engine RNG; consumed in sorted cell order for determinism (INV-2).

    Returns
    -------
    dict mapping blob_id -> ResolvedAction for every blob that filed a claim.
    IDLE blobs are NOT included; those are synthesized by the engine (Phase 5).
    """
    results: dict[int, ResolvedAction] = {}

    for cell, claims in sorted(claims_by_cell.items()):
        if cell in occupied_cells:
            # Occupied cell: all incoming claims fail
            for c in claims:
                results[c.blob_id] = ResolvedAction(
                    c.blob_id, c.action, cell, succeeded=False,
                )
        elif len(claims) == 1:
            # Uncontested empty cell: single claimant succeeds
            c = claims[0]
            results[c.blob_id] = ResolvedAction(
                c.blob_id, c.action, cell, succeeded=True,
            )
        else:
            # Contested empty cell: resolver picks one winner
            winner = resolver.resolve(claims, rng)
            for c in claims:
                succeeded = c.blob_id == winner.blob_id
                results[c.blob_id] = ResolvedAction(
                    c.blob_id, c.action, cell, succeeded=succeeded,
                )

    return results
