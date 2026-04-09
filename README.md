# Multi-Agent Collective Construction (MACC) – RViz Simulation

A Python + ROS 2 (RViz) simulation of multi-agent collective construction inspired by the paper:

**Multi-agent Collective Construction using 3D Decomposition**
Akshaya Kesarimangalam Srinivasan et al., 2023 — https://arxiv.org/abs/2309.00985

The simulation visualizes how multiple simple cubic robots can collaboratively build a randomly generated 3D voxel structure by decomposing it into independent substructures and executing a parallelizable construction schedule, mirroring Fig. 1 of the paper.

## Overview

The system runs as a 4-stage animated pipeline in RViz:

1. **Stage 1 — Input Structure.** A randomly generated, valid 3D voxel structure is shown as solid cubes, matching Fig. 1a of the paper.
2. **Stage 2 — Structural Decomposition.** The structure is recolored by substructure using Algorithm 1 (shadow-region decomposition). Substructures are revealed one at a time so the decomposition is visually traceable. Mirrors Fig. 1b.
3. **Stage 3 — Construction Order.** Each substructure is labeled with its index in the feasible construction order produced by Algorithm 3 (substructure ordering and merging). Mirrors Fig. 1c.
4. **Stage 4 — Construction.** Cubic robots build the structure substructure-by-substructure, following the permissible actions defined in Section II of the paper: move N/S/E/W, climb up/down one block, pick up, place, and wait. Robots and blocks are rendered as same-sized voxel cubes; carried blocks are visibly stacked on top of the robot and colored by destination substructure. Mirrors Fig. 1d.

A new random structure is generated on every launch. The grid dimensions (default 7×7×4) stay constant; the block placements vary.

## Technologies Used

- Python 3
- ROS 2 (Jazzy)
- RViz2
- NumPy
- Matplotlib (earlier non-ROS visualization path, still available)

## Key Features

- **Random structure generator** — produces a new valid (no floating blocks, per Definition 1 of the paper) 3D voxel structure on every run, with optional `--seed` for reproducibility.
- **Structural decomposition** via shadow regions (Algorithm 1).
- **Bottom-up ordering and merging** to compute a feasible construction sequence (Algorithm 3).
- **Parallel construction grouping** — substructures with no mutual dependencies are scheduled to be built simultaneously.
- **4-stage animated visualization** mapping directly to Fig. 1 (a–d) of the paper, with configurable timing for each stage.
- **Boundary block supply** — per the paper, blocks come from the grid boundary and robots start and finish outside the grid world.
- **Cubic, block-sized robots** — robots are rendered as white cubes the same size as structure blocks, visually distinct from every substructure color, with their carried block stacked on top and colored by its destination substructure.
- **Configurable timing** — per-action duration, stage pauses, and substructure reveal speed are all exposed as constants at the top of the publisher node.

## Running

In separate sourced terminals:

```bash
# Terminal 1: RViz
rviz2

# Terminal 2: launch the simulation
ros2 launch macc_rviz macc.launch.py
```

In RViz, set the **Fixed Frame** to `world` and add MarkerArray displays for the `/macc/*` topics (`blocks`, `robots`, `ghost`, `intro`, `stage_label`, `subtext`). Save the config so you don't have to redo this on each launch.

## Scope

This project focuses on visualization and system-level behavior. It does not implement the MILP optimization layer from the paper — the action sequences for each substructure are generated heuristically rather than via Gurobi. The decomposition, ordering, and parallelization logic follow the paper directly.
