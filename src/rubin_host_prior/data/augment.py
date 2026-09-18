"""Augmentations that leave the noise alone.

The eight symmetries of the square are exact re-indexings of the pixel grid:
they do not interpolate, do not blur, and do not change the per-pixel noise
distribution or its independence between pixels.  They are therefore safe for a
model whose whole purpose is to represent a distribution with correct noise
properties.  Adding noise or blur would not be -- and is not offered here.
"""

from __future__ import annotations

import numpy as np

N_DIHEDRAL = 8


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """Apply element ``k`` of the dihedral group D4 to the trailing two axes.

    ``k`` in 0-3 are rotations by ``90 * k`` degrees; 4-7 are the same rotations
    composed with a horizontal flip.
    """
    if not 0 <= k < N_DIHEDRAL:
        raise ValueError(f"k must be in [0, {N_DIHEDRAL}), got {k}")
    out = np.rot90(a, k % 4, axes=(-2, -1))
    if k >= 4:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def random_dihedral(
    a: np.ndarray, rng: np.random.Generator, per_example: bool = True
) -> np.ndarray:
    """Random D4 element, independently per example if ``a`` is batched."""
    if not per_example or a.ndim < 3:
        return dihedral(a, int(rng.integers(N_DIHEDRAL)))
    ks = rng.integers(N_DIHEDRAL, size=a.shape[0])
    return np.stack([dihedral(a[i], int(k)) for i, k in enumerate(ks)])
