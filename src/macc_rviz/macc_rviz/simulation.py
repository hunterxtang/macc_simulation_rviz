import numpy as np
import time

def simulate_construction(structure, substructures, order, delay=0.05):
    world = np.zeros_like(structure)
    step = 0
    print("\n=== Sequential Simulation Start ===")
    for idx in order:
        s = substructures[idx]
        coords = np.argwhere(s == 1)
        print(f"\n-- Building Substructure {idx+1} ({len(coords)} blocks) --")
        for z, y, x in coords:
            if world[z, y, x] == 0:
                world[z, y, x] = 1
                step += 1
                print(f"Step {step:03d}: placed block at (x={x}, y={y}, z={z})")
                time.sleep(delay)
    print("\n=== Construction Complete ===")
    return world


def simulate_parallel_construction(structure, substructures, parallel_groups, delay=0.05):
    """
    Simulates parallel construction layer by layer.
    Each group = substructures that can be built simultaneously.
    """
    world = np.zeros_like(structure)
    step = 0
    print("\n=== Parallel Simulation Start ===")
    for g, group in enumerate(parallel_groups):
        print(f"\n== Parallel Layer {g+1}: {len(group)} substructures ==")
        active_blocks = []
        # gather all blocks from this layer
        for idx in group:
            s = substructures[idx]
            coords = np.argwhere(s == 1)
            active_blocks.extend(coords)
        for (z, y, x) in active_blocks:
            if world[z, y, x] == 0:
                world[z, y, x] = 1
                step += 1
                print(f"Step {step:03d}: placed block at (x={x}, y={y}, z={z})")
                time.sleep(delay)
    print("\n=== Parallel Construction Complete ===")
    return world
