import numpy as np
from macc_rviz.structure_utils import topological_sort

def decompose_structure(structure):
    # (same as before)
    H, Y, X = structure.shape
    heights = np.sum(structure, axis=0)
    substructures = []
    assigned = np.zeros_like(structure)
    towers = sorted(
        [(x, y, heights[y, x]) for y in range(Y) for x in range(X) if heights[y, x] > 0],
        key=lambda t: -t[2]
    )
    for x, y, h in towers:
        if assigned[:, y, x].any():
            continue
        shadow = []
        for dx in range(-h+1, h):
            for dy in range(-h+1, h):
                if abs(dx) + abs(dy) < h:
                    sx, sy = x+dx, y+dy
                    if 0 <= sx < X and 0 <= sy < Y:
                        shadow.append((sx, sy))
        sub = np.zeros_like(structure)
        for sx, sy in shadow:
            for z in range(H):
                if structure[z, sy, sx] and not assigned[z, sy, sx]:
                    sub[z, sy, sx] = 1
                    assigned[z, sy, sx] = 1
        if sub.any():
            substructures.append(sub)
    return substructures

def compute_dependencies(substructures):
    dependencies = []
    for i, s1 in enumerate(substructures):
        for j, s2 in enumerate(substructures):
            if i == j: continue
            if np.any((np.roll(s2, 1, axis=0) & s1)):
                dependencies.append((j, i))
    return dependencies

def order_substructures(substructures):
    deps = compute_dependencies(substructures)
    order = topological_sort(len(substructures), deps)
    return order, deps
