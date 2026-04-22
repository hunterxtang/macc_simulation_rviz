# Multi-Agent Collective Construction (MACC) – RViz Simulation

A Python + ROS 2 (RViz) simulation of multi-agent collective construction inspired by:

> **Multi-agent Collective Construction using 3D Decomposition**
> Akshaya Kesarimangalam Srinivasan et al., IROS 2023 — https://arxiv.org/abs/2309.00985

and the exact MILP formulation from:

> **Exact Approaches to the Multi-agent Collective Construction Problem**
> Edward Lam, Peter J. Stuckey, Sven Koenig, T. K. Satish Kumar, CP 2020.

The simulation visualizes how multiple cubic robots can collaboratively build a 3D voxel structure by decomposing it into substructures and executing a parallelizable construction schedule, mirroring Fig. 1 of the decomposition paper. It implements three planners — a reactive heuristic, a CBS-based space-time planner, and a MILP planner clean-room-implemented from Lam et al. §4 Fig. 2 — plus a hybrid MILP→CBS fallback for large substructures.

## Overview

The system runs as a 4-stage animated pipeline in RViz:

1. **Stage 1 — Input Structure.** A randomly generated valid 3D voxel structure is shown as solid cubes (Fig. 1a of the paper).
2. **Stage 2 — Structural Decomposition.** The structure is recolored by substructure using Algorithm 1 (shadow-region decomposition). Substructures are revealed one at a time (Fig. 1b).
3. **Stage 3 — Construction Order.** Each substructure is labeled with its index in the feasible construction order produced by Algorithm 3 (Fig. 1c).
4. **Stage 4 — Construction.** Cubic robots build the structure substructure-by-substructure following the permissible actions from Section II of the paper: move N/S/E/W, climb up/down one block, pick up, place, wait. Robots and blocks are rendered as same-sized voxel cubes; carried blocks are visibly stacked on top and colored by destination substructure (Fig. 1d).

By default the sim runs MILP on the 13-block example structure — a fast, deterministic, ~7-second concurrent-multi-robot demo. Pass `use_example_structure:=false` to switch to random generation; grid shape defaults to 5×5×3 at density 0.25 (~19 expected blocks) and is tunable via `grid_x` / `grid_y` / `grid_z` / `density`. Random runs use a fresh seed unless `seed:=` is given.

## Planners

Three planners are selectable via the `planner` launch argument:

| Planner | Command | Speed | Quality | Notes |
|---|---|---|---|---|
| `heuristic` | `planner:=heuristic` | fastest | poor | Original reactive FSM, single-robot-ish behavior |
| `cbs` | `planner:=cbs` | fast (~ms) | medium | Conflict-Based Search over space-time A* |
| `milp` | `planner:=milp` | slow (seconds to minutes) | optimal | Lam et al. 2020 MILP via Gurobi |

On `example_structure` (13 blocks), MILP produces a makespan of ~20 vs CBS's ~84. MILP minimizes sum-of-costs exactly; CBS is serial-per-block within each substructure for correctness.

### Hybrid MILP → CBS fallback

When `planner=milp`, each substructure is attempted via MILP first. If Gurobi exceeds `milp_total_time_limit` without producing a feasible solution, that substructure falls back to CBS automatically and the sim logs the fallback explicitly. Parallel-group accounting (prior trajectories, per-tick agent caps) is preserved across the handoff. This matches the paper's own reported pattern of MILP timeouts on harder instances (>10,000 s on some of their Table III cases).

## Technologies

- Python 3.12
- ROS 2 (Jazzy)
- RViz2
- Gurobi Optimizer 13.0.1 (academic license required for the MILP planner)
- NumPy, Matplotlib

## Installation

### ROS 2 and the package
Standard ROS 2 Jazzy workspace setup. Clone into `src/` of a colcon workspace and build:

```bash
cd ~/ros2_ws
colcon build --packages-select macc_rviz --symlink-install
source install/setup.bash
```

### Gurobi (only needed for MILP planner)

1. Obtain a free academic license at https://gurobi.com/academia.
2. Download Gurobi Optimizer for your architecture into `/opt`:
   - Linux x86_64: `gurobi13.0.1_linux64.tar.gz`
   - Linux ARM64 (e.g. QEMU VMs on Apple Silicon): `gurobi13.0.1_armlinux64.tar.gz`
3. Set environment variables in `~/.bashrc`:
```bash
   export GUROBI_HOME="/opt/gurobi1301/linux64"   # or armlinux64
   export PATH="${PATH}:${GUROBI_HOME}/bin"
   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```
4. Activate the license (requires connection to academic network):
```bash
   grbgetkey 
```
5. Install Python bindings into your venv:
```bash
   pip install gurobipy
```

## Running

In separate sourced terminals:

```bash
# Terminal 1: RViz
rviz2

# Terminal 2: launch the simulation (defaults to MILP on the 13-block example)
ros2 launch macc_rviz macc.launch.py
```

In RViz, set Fixed Frame to `world` and add MarkerArray displays for the `/macc/*` topics (`blocks`, `robots`, `ghost`, `intro`, `stage_label`, `subtext`). Save the config so you don't have to redo this on each launch.

### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `planner` | `milp` | `heuristic`, `cbs`, or `milp` |
| `use_example_structure` | `true` | Use the deterministic 13-block example instead of random |
| `seed` | *(none, random)* | Seed the random structure for reproducibility |
| `grid_x` | `5` | Random-structure X extent (ignored when `use_example_structure=true`) |
| `grid_y` | `5` | Random-structure Y extent |
| `grid_z` | `3` | Random-structure Z extent |
| `density` | `0.25` | Per-column height probability; expected blocks ≈ `grid_x*grid_y*grid_z*density` |
| `cbs_max_t` | `400` | CBS makespan cap before serial fallback |
| `cbs_branch_limit` | `500` | CBS node expansion limit |
| `milp_per_t_time_limit` | `60.0` | Per-T solve budget in seconds |
| `milp_total_time_limit` | `60.0` | Total budget across T-sweep before CBS fallback |
| `milp_mip_gap` | `0.0` | Gurobi MIP gap (0 = prove optimal) |
| `milp_T_max` | `40` | Maximum makespan to try before giving up |

### Examples

```bash
# Default: MILP on the 13-block example — ~7-second concurrent-multi-robot demo
ros2 launch macc_rviz macc.launch.py

# Small random structure (5×5×3 @ 0.25, ~19 expected blocks)
ros2 launch macc_rviz macc.launch.py use_example_structure:=false seed:=42

# CBS on the same small random structure (serial-per-block demo)
ros2 launch macc_rviz macc.launch.py planner:=cbs use_example_structure:=false seed:=42

# Stress-test: reproduce the old 7×7×4 @ 0.4 (~78 blocks) behavior
ros2 launch macc_rviz macc.launch.py use_example_structure:=false \
    grid_x:=7 grid_y:=7 grid_z:=4 density:=0.4

# Hybrid MILP→CBS on a larger random structure; expect fallback on big subs
ros2 launch macc_rviz macc.launch.py use_example_structure:=false \
    grid_x:=7 grid_y:=7 grid_z:=4 density:=0.4 seed:=1
```

## Features

- Random structure generator (Definition 1 valid, no floating blocks) with optional `--seed`
- Algorithm 1 shadow-region decomposition
- Algorithm 3 bottom-up ordering and merging
- Parallel-group detection — substructures with no mutual dependency can be built concurrently
- Three planners with a unified replay loop
- MILP encoding from Lam et al. 2020 §4 Fig. 2: 9-tuple action variables, height variables, two-stage solve (iterate T until feasible, then minimize sum-of-costs)
- Parallel-construction constraint injection: prior-committed trajectories are passed as (vertex/edge) blocking constraints to subsequent parallel substructures
- MILP→CBS per-substructure fallback for scale-wall substructures
- Scaffolding cleanup: pickup events clear the world voxel, so the completed structure matches the target (previously scaffolding voxels lingered)
- CBS serial-per-block replan with heightmap refresh, eliminating robot-through-block clipping
- Boundary block supply per the paper (robots enter/exit any boundary cell)
- Cubic block-sized robots rendered distinctly from structure blocks, with carried blocks stacked on top

## Benchmarks

A 6-case designed benchmark in `docs/benchmark_results.md` spans the complexity range from trivial (5 blocks) to the MILP scale wall (44 blocks). Headline numbers:

- Cases 1–5 (5–40 blocks): MILP+CBS hybrid wins against CBS-alone on makespan by 1.75× to 2.92×.
- Case 4 (27 blocks): 1 MILP sub + 1 CBS-fallback sub. Fallback path validated end-to-end.
- Case 6 (44 blocks): MILP's 24-block sub exceeds practical wall-clock budget on ARM64 VM hardware, demonstrating the scale wall the original paper also reports.

See `docs/planner_architecture.md` for the full architecture writeup including a decision log, bug history, and future work.

## Known limitations

- **Single parallel group in practice.** The shadow-based decomposition assigns each column to exactly one substructure, so inter-substructure dependency checks rarely fire on randomly generated structures. Parallelism within a group is bounded by `num_robots`, not group count.
- **MILP scale wall.** Substructures beyond ~18–24 blocks often exceed practical wall-clock budgets on modest hardware (matches the paper's own 5.7-day timeouts on their instance 5). The hybrid falls back to CBS.
- **CBS is strictly serial within a substructure.** A previous within-round parallel CBS had robots clipping through placed blocks; the current per-block serial approach eliminates this deterministically at the cost of intra-substructure parallelism.
- **`step_duration_sec` is not currently exposed as a launch arg.** Edit the source constant or the node params dict to change tick speed.

## Scope

This project implements the full decomposition + ordering + parallel-group pipeline from Srinivasan et al., plus the MILP subroutine from Lam et al. 2020 that the decomposition paper references. It is not a line-for-line reproduction of the Srinivasan et al. numerical results (Table III of their paper) — structures in that table were rendered as low-resolution thumbnails that cannot be reliably extracted. The benchmark in this repo uses 6 designed cases spanning a similar complexity range instead.

## References

- Srinivasan, A. K., Singh, S., Gutow, G., Choset, H., Vundurthy, B. "Multi-agent Collective Construction using 3D Decomposition." IROS 2023. https://arxiv.org/abs/2309.00985
- Lam, E., Stuckey, P. J., Koenig, S., Kumar, T. K. S. "Exact Approaches to the Multi-agent Collective Construction Problem." CP 2020. https://doi.org/10.1007/978-3-030-58475-7_43
- Sartoretti, G., Wu, Y., Paivine, W., Kumar, T. K. S., Koenig, S., Choset, H. "Distributed Reinforcement Learning for Multi-robot Decentralized Collective Construction." DARS 2018.
