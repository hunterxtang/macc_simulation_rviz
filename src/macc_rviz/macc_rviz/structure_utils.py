import numpy as np
import random as _random


def create_small_random_structure(seed=None):
    """
    Generates a valid random structure that fits in a 5x5x3 voxel grid with
    5-12 blocks. Satisfies Definition 1: every block at z>0 has a block
    directly beneath it (no floating blocks). Seed is random by default.
    """
    rng = _random.Random(seed)
    X, Y, Z = 5, 5, 3

    while True:
        # Build column-by-column; each column is a stack of height 0..Z
        structure = np.zeros((Z, Y, X), dtype=int)
        for y in range(Y):
            for x in range(X):
                h = rng.randint(0, Z)
                for z in range(h):
                    structure[z, y, x] = 1
        count = int(structure.sum())
        if 5 <= count <= 12:
            return structure


def create_example_structure():
    """
    Returns a simple 5x5x3 voxel structure:
    - Layer 0 (ground): 3x3 square
    - Layer 1: a central vertical tower
    - Layer 2: one top cube at the tower’s peak
    """
    structure = np.array([
        [[0,0,0,0,0],
         [0,1,1,1,0],
         [0,1,1,1,0],
         [0,1,1,1,0],
         [0,0,0,0,0]],

        [[0,0,0,0,0],
         [0,0,1,0,0],
         [0,0,1,0,0],
         [0,0,1,0,0],
         [0,0,0,0,0]],

        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,1,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]]
    ])
    return structure


def create_random_structure(x=7, y=7, z=4, density=0.3, seed=None):
    """
    Creates a random structure (for testing parallel behavior).
    - density controls how filled the grid is (0–1)
    - automatically ensures structural validity (no floating blocks)
    - seed: explicit int for reproducibility, or None for OS-entropy randomness.
      Uses an isolated RNG instance so global numpy state is never touched.
    """
    rng = np.random.default_rng(seed)
    structure = np.zeros((z, y, x), dtype=int)
    for i in range(y):
        for j in range(x):
            height = int(rng.binomial(z, density))
            for k in range(height):
                structure[k, i, j] = 1
    return structure


def topological_sort(num_nodes, dependencies):
    """
    Performs topological sort given a list of edges (a,b)
    meaning a -> b (b depends on a).
    Returns a valid build order list.
    """
    indeg = {i: 0 for i in range(num_nodes)}
    for a, b in dependencies:
        indeg[b] += 1
    order, queue = [], [i for i in indeg if indeg[i] == 0]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for a, b in dependencies:
            if a == cur:
                indeg[b] -= 1
                if indeg[b] == 0:
                    queue.append(b)
    return order
