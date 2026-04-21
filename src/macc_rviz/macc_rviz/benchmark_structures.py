"""Benchmark structures for MILP+CBS hybrid planner characterization.

Six hand-designed structures spanning a complexity axis from tiny to
large. Each is a ``numpy.ndarray`` shaped ``(Z, Y, X)`` with ``1`` for
filled and ``0`` for empty. All satisfy Definition 1 (no floating
blocks: every cell at ``z>0`` has a filled cell directly below).

These are NOT a reproduction of any paper's benchmark set. They are
designed to exercise the MILP scale boundary and the CBS fallback path
in our hybrid planner.
"""

import numpy as np


def _from_heightmap(hm: np.ndarray, z_max: int) -> np.ndarray:
    """Build an ``(Z, Y, X)`` voxel array from a per-column heightmap.

    Each column is filled contiguously from ``z=0`` up to ``hm[y, x]-1``.
    """
    Y, X = hm.shape
    out = np.zeros((z_max, Y, X), dtype=int)
    for y in range(Y):
        for x in range(X):
            for z in range(int(hm[y, x])):
                out[z, y, x] = 1
    return out


def case1_tiny() -> np.ndarray:
    """5 blocks, 5x5x3. Two well-separated low towers."""
    hm = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ])
    return _from_heightmap(hm, z_max=3)


def case2_small() -> np.ndarray:
    """13 blocks, 5x5x3. 3x3 base with 4 corner h=2 towers and a
    single h=1 center cell. Decomposes into 5 small subs (one per
    corner tower + center), all parallelizable."""
    hm = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 1, 2, 0],
        [0, 1, 1, 1, 0],
        [0, 2, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    return _from_heightmap(hm, z_max=3)


def case3_medium_low() -> np.ndarray:
    """20 blocks, 6x6x3. Flat 4x4 base with two corner towers at h=3.

    Designed to give a small handful of subs with parallelizable corners.
    """
    hm = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 3, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 3, 0],
        [0, 0, 0, 0, 0, 0],
    ])
    return _from_heightmap(hm, z_max=3)


def case4_medium_high() -> np.ndarray:
    """27 blocks, 5x5x3. A 3x3x3 solid cube — dense, near MILP boundary."""
    hm = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0],
    ])
    return _from_heightmap(hm, z_max=3)


def case5_large_flat() -> np.ndarray:
    """40 blocks, 8x8x2. Sparse pattern with 4 corner + 4 inner h=2
    towers and h=1 fill. Designed to give a moderate number of small
    subs (rather than dozens of singletons), all parallelizable."""
    hm = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 1, 1, 2, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 2, 2, 1, 1, 0],
        [0, 1, 1, 2, 2, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 2, 1, 1, 1, 1, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    return _from_heightmap(hm, z_max=2)


def case6_large_dense() -> np.ndarray:
    """44 blocks, 7x7x4. Cross-shaped layout with central h=4 tower,
    intermediate h=2 ring, corner h=1 satellites. Expected MILP failure
    domain at our budgets — primary fallback validation."""
    hm = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [1, 2, 1, 0, 1, 2, 1],
        [0, 1, 2, 1, 2, 1, 0],
        [0, 0, 1, 4, 1, 0, 0],
        [0, 1, 2, 1, 2, 1, 0],
        [1, 2, 1, 0, 1, 2, 1],
        [1, 1, 0, 0, 0, 1, 1],
    ])
    return _from_heightmap(hm, z_max=4)


ALL_CASES = [
    ('case1_tiny', case1_tiny),
    ('case2_small', case2_small),
    ('case3_medium_low', case3_medium_low),
    ('case4_medium_high', case4_medium_high),
    ('case5_large_flat', case5_large_flat),
    ('case6_large_dense', case6_large_dense),
]
