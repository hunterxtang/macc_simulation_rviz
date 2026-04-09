# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See the workspace-root `~/ros2_ws/CLAUDE.md` for the full reference. This file contains the same information scoped to this package.

## Build and Run

```bash
# From workspace root
colcon build --packages-select macc_rviz
source ~/ros2_ws/install/setup.bash

ros2 launch macc_rviz macc.launch.py
ros2 run macc_rviz macc_rviz_sim --ros-args -p seed:=42 -p num_robots:=6

# Standalone (no ROS)
python3 macc_rviz/main.py [--seed 42]
```

## Tests

```bash
colcon test --packages-select macc_rviz
colcon test-result --verbose
python3 -m pytest test/test_flake8.py -v
```

Style/lint only (flake8, pep257, copyright) — no functional unit tests.

## Architecture

Pipeline: `structure_utils` → `decomposition` → `parallel` → execution.

**`structure_utils.py`** — Two generators: `create_random_structure` (7×7×4, used by RViz node) and `create_small_random_structure` (5×5×3, used by matplotlib path). Both enforce no-floating-blocks (Definition 1). Use isolated `numpy.random.default_rng(seed)` instances — global numpy RNG state is never touched.

**`decomposition.py`** — Shadow-based heuristic: tallest columns claim nearby blocks. Dependency edges computed by layer overlap (`np.roll(s2, 1, axis=0) & s1`). Topological sort via `structure_utils.topological_sort`.

**`parallel.py`** — Kahn's algorithm on the dependency DAG → list of parallel groups.

**`macc_rviz_sim.py`** — ROS2 node. 5-stage intro (preview → decomp → labels → build → return). Timer fires every `STEP_DURATION_SEC = 0.4` s. Robots use BFS on a per-tick heightmap; all share depot `(DEPOT_X, DEPOT_Y) = (0, 0)`. `seed=-1` (default) picks a fresh seed from `time.time_ns()`; startup log prints the resolved seed, block count, and MD5 fingerprint.

**`visualization.py` + `main.py`** — Standalone matplotlib path. Same BFS/heightmap logic re-implemented locally. Chains: target preview → decomp coloring → `animate_build`.

**`simulation.py`** — Legacy console-print simulator. Not used by either execution path.

### Key conventions

- Voxel arrays are `[z, y, x]`; Z = height axis.
- Block colors: tab10 palette indexed by substructure index.
- Build-order labels: serial = "1","2",…; parallel = "1a","1b",….
- `INTER_GROUP_PAUSE_SEC = 1.5` — highlight pause between parallel groups.
