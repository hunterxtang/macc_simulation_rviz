# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

ROS2 Python package (`ament_python`). All commands from the workspace root (`~/ros2_ws`).

```bash
# Build
colcon build --packages-select macc_rviz

# Source after building
source ~/ros2_ws/install/setup.bash

# Run via launch file (recommended — sets all parameters)
ros2 launch macc_rviz macc.launch.py

# Run node directly with parameter overrides
ros2 run macc_rviz macc_rviz_sim --ros-args -p seed:=42 -p num_robots:=6

# Standalone matplotlib mode (no ROS required)
python3 src/macc_rviz/macc_rviz/main.py [--seed 42]
```

## Tests

Style/lint only — no functional unit tests.

```bash
colcon test --packages-select macc_rviz
colcon test-result --verbose

# Single check
python3 -m pytest src/macc_rviz/test/test_flake8.py -v
```

## Architecture

Simulates **Multi-Agent Collective Construction (MACC)**: robots build a 3D voxel target collaboratively, following the decomposition + parallel-group planning pipeline from the paper.

### Pipeline (shared by both execution modes)

1. **`structure_utils.py`** — Generates the 3D voxel target as `numpy` array `(Z, Y, X)` where `1` = occupied. Columns are always contiguous from ground up (no floating blocks, Definition 1). Two generators:
   - `create_random_structure(x=7, y=7, z=4, density=0.4, seed=...)` — used by the RViz node.
   - `create_small_random_structure(seed=...)` — used by the matplotlib path; generates a 5×5×3 structure with 5–12 blocks.

2. **`decomposition.py`** — Splits the target into substructures via a shadow-based heuristic (tall towers claim nearby columns). Computes inter-substructure dependencies via layer overlap (`compute_dependencies`), then topologically sorts them (`order_substructures`).

3. **`parallel.py`** — `find_parallel_groups` runs Kahn's algorithm on the dependency DAG to produce groups of substructures that can be built concurrently.

4. **Execution** — two independent modes (see below).

### `cbs_planner.py` — Space-time A* and CBS-lite (not yet integrated)

Fully implemented (570 lines) but not imported by either execution path. Contains:
- `space_time_astar` — plans a single robot's path in (x, y, t) space-time, respecting vertex and edge constraints.
- `cbs_plan` — joint CBS-lite solver: plans all robots' full trajectories (depot → pickup → placement per block), detects vertex/swap conflicts, branches by adding forbidden-state constraints, and replans until conflict-free or branch limit is hit.
- Constraint format: vertex `(x, y, t)` or edge `(x1, y1, x2, y2, t)` tuples in a `frozenset`.

Currently, both execution modes use simple BFS navigation without inter-robot conflict resolution. Integrating `cbs_planner` would replace that BFS with conflict-free multi-agent paths.

### RViz mode (`macc_rviz_sim.py`)

`MACCRvizSim` is a ROS2 node with a discrete-time simulation driven by a timer at `STEP_DURATION_SEC` (currently `0.4` s/tick). It runs a 5-stage intro sequence before construction:

- **Stage 1** — full target, uniform color
- **Stage 2** — progressive decomposition reveal (one substructure per `SUBSTRUCTURE_REVEAL_SEC`)
- **Stage 3** — build-order labels added
- **Stage 4** — actual construction (robots move, pick up, place)
- **Stage 5** — robots return to nearest boundary cell (`bfs_to_boundary`) after build completes

Robot navigation uses BFS on a heightmap (`heightmap()` sums `world` along Z). Robots can climb ±1 block per step. All robots share a single fixed depot at `(DEPOT_X, DEPOT_Y) = (0, 0)` for block pickups during Stage 4.

RViz topics (all `MarkerArray`):
- `/macc/blocks` — placed blocks colored by substructure (tab10 palette)
- `/macc/robots` — robot cubes (pure white)
- `/macc/ghost` — latched semi-transparent target overlay
- `/macc/subtext` — latched build-order labels per substructure
- `/macc/intro` — intro-stage voxels (cleared when Stage 4 begins)
- `/macc/stage_label` — latched main stage text

ROS parameters (set in `macc.launch.py` or via `--ros-args`):
| Parameter | Default | Notes |
|---|---|---|
| `num_robots` | 4 | |
| `step_duration_sec` | 0.4 | overrides `STEP_DURATION_SEC` constant |
| `block_scale` | 1.0 | marker cube side length in metres (declared in node, not in launch file) |
| `use_example_structure` | false | hardcoded 5×5×3 structure |
| `seed` | -1 | `-1` = fresh random each run; any non-negative value fixes the layout |

Startup log reports the resolved seed, block count, and an 8-character MD5 fingerprint so a specific run can be reproduced.

### Matplotlib mode (`main.py` + `visualization.py`)

Standalone — no ROS dependency. `show_construction_process` chains three matplotlib windows: target preview → decomposition coloring → animated robot simulation (`animate_build`). Uses the same BFS/heightmap logic as the RViz node but re-implemented locally in `visualization.py`. Robots navigate to `(0, 0)` for pickups (mirroring the RViz depot).

### Key conventions

- Voxel array indexing is `[z, y, x]` throughout (Z = height axis).
- The heightmap is recomputed each tick from the current `world` state.
- Block color uses the tab10 palette indexed by substructure index. Parallel groups share the same palette position as their substructure; serial labels are "1", "2", … and parallel are "1a", "1b", ….
- `INTER_GROUP_PAUSE_SEC` (1.5 s) — brief highlight pause between parallel groups during construction.
- **`simulation.py`** — legacy console-print simulator, not wired into either execution path. Safe to ignore.
- **Scaffolding** — the planner does not generate temporary blocks. The rendering infrastructure (gray semi-transparent style) exists but requires planner-side changes to activate.
