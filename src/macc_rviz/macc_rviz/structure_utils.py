import numpy as np

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
    """
    if seed is not None:
        np.random.seed(seed)
    structure = np.zeros((z, y, x), dtype=int)
    for i in range(y):
        for j in range(x):
            height = np.random.binomial(z, density)
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
