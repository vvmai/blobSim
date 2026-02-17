"""RNG hierarchy for deterministic, per-blob-independent random number generation.

Spec reference: tech_spec_mvp.md §13 (RNG Architecture)

Hierarchy:
    root_seed -> SeedSequence -> spawn(3) -> engine_ss, env_ss, blob_pool_ss
    - engine_rng: conflict resolution (deterministic iteration order)
    - env_rng: stochastic environments
    - blob_pool_entropy: int used to derive per-blob RNGs via spawn_key

Per-blob independence guarantee: adding/removing blob X does NOT change
blob Y's RNG stream. Same root_seed + same blob_id -> identical sequence.
"""

import numpy as np


def create_rng_hierarchy(
    seed: int,
) -> tuple[np.random.Generator, np.random.Generator, int]:
    """Create the three-stream RNG hierarchy from a single root seed.

    Args:
        seed: User-provided root seed for full reproducibility.

    Returns:
        Tuple of (engine_rng, env_rng, blob_pool_entropy) where:
        - engine_rng: Generator for conflict resolution.
        - env_rng: Generator for stochastic environments.
        - blob_pool_entropy: int entropy for per-blob RNG creation.
    """
    root_ss = np.random.SeedSequence(seed)
    engine_ss, env_ss, blob_pool_ss = root_ss.spawn(3)

    engine_rng = np.random.default_rng(engine_ss)
    env_rng = np.random.default_rng(env_ss)
    entropy = blob_pool_ss.entropy
    if not isinstance(entropy, int):
        raise TypeError(f"Expected int entropy from SeedSequence, got {type(entropy)}")
    blob_pool_entropy: int = entropy

    return engine_rng, env_rng, blob_pool_entropy


def make_blob_rng(blob_pool_entropy: int, blob_id: int) -> np.random.Generator:
    """Create a deterministic RNG for a specific blob.

    Uses spawn_key (not entropy unpacking) so each blob's stream is
    independent: adding/removing other blobs has no effect.

    Args:
        blob_pool_entropy: Entropy from the blob pool SeedSequence.
        blob_id: Unique blob identifier.

    Returns:
        Generator with a sequence determined solely by blob_pool_entropy
        and blob_id.
    """
    ss = np.random.SeedSequence(entropy=blob_pool_entropy, spawn_key=(blob_id,))
    return np.random.default_rng(ss)
