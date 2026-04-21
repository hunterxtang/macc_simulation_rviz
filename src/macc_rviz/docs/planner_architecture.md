# Planner Architecture & Roadmap

Context for ongoing work to replace the heuristic per-tick action generator
with a paper-faithful multi-agent planner. Captures the Phase 0 survey, the
CBS-vs-MILP decision log, and the agreed phase ordering.

## Paper references

- **Decomposition paper (current codebase target):** Kesarimangalam Srinivasan
  et al., "Decomposition-based hierarchical planning for multi-agent
  collective construction" (IROS). Implements Algorithm 1 (shadow-region
  decomposition), Algorithm 3 (substructure ordering & merging), and
  parallel-group detection.
- **MILP subroutine (to integrate):** Lam, Stuckey, Koenig & Kumar,
  "Exact approaches to the multi-agent collective construction problem"
  (CP 2020). The paper above invokes this as a per-substructure subroutine.

---

## Module map

| Role | File | Symbol(s) |
|---|---|---|
| Algorithm 1 — shadow-region decomposition | `macc_rviz/decomposition.py` | `decompose_structure(structure)` → `list[np.ndarray]` |
| Algorithm 3 — ordering & dependency DAG | `macc_rviz/decomposition.py` | `compute_dependencies`, `order_substructures`; `structure_utils.topological_sort` |
| Parallel-group detector (Kahn) | `macc_rviz/parallel.py` | `find_parallel_groups(substructures, deps)` → `list[list[int]]` |
| Visualizer (RViz / ROS2) — production path | `macc_rviz/macc_rviz_sim.py` | `MACCRvizSim` node |
| Visualizer (standalone matplotlib) | `macc_rviz/visualization.py` | `show_construction_process`, `animate_build` |
| Pre-built but **unwired** space-time planner | `macc_rviz/cbs_planner.py` | `space_time_astar`, `cbs_plan`, `cbs_return`, `_serial_fallback` |
| Reusable plan primitives | `cbs_planner.py` L42–47 | `Step(action,x,y,z,t)`, `ACTION_WAIT/MOVE/PICKUP/PLACE` |

---

## Where the "heuristic action generator" actually lives

**There is no standalone function to swap out.** The heuristic is an inlined
per-tick reactive state machine in two places:

1. `macc_rviz_sim.py::MACCRvizSim.tick()` — production path (ROS).
2. `visualization.py::animate_build()` — standalone matplotlib path.

Both drive the same per-robot phase machine:
`to_pickup → pickup → to_place → place`, with BFS-on-heightmap pathfinding
(`bfs_path`). Robots pop tasks from a single FIFO (`self.task_queue`)
populated from `self.group_task_lists: list[list[(x, y, z, si)]]` (one list
per parallel group, sorted bottom-up by z). Task assignment (`assign_tasks`)
pops one task into `self.active[robot_id]`. The reactive loop then routes
each robot to the depot, picks up, routes to a neighbor of the target voxel,
and places — all via BFS, with a priority-based reservation table plus
jiggle/detour for collisions.

World state is mutated directly: `self.world[z,y,x] = 1` on `place`.
**There is no explicit "action sequence" data structure** — actions are
implicit in per-tick position deltas.

---

## Interface boundary the new planners must match

Since the heuristic never returns anything, Phase 3 has to **define** the
interface. The shape of information the heuristic actually consumes per
substructure:

**Inputs:**
- `substructure: np.ndarray(Z,Y,X) int` — blocks this substructure adds.
- `world_init: np.ndarray(Z,Y,X) int` — current built world (union of all
  previously-built substructures).
- `grid_shape: (X, Y, H)` — `H` may need augmenting since a carrying robot
  occupies z+1.
- `num_agents: int` — agent cap.
- `agent_starts: list[(x, y)]` — robot positions at substructure start.
- `depots: list[(x, y)]` or `(x, y)` — pickup cell(s). The paper permits
  any boundary cell; we open this up (see Decision D-2 below).
- `substructure_index: int` — for bookkeeping (`world_sub[z,y,x] = si + 1`).

**Output** — fits the existing `Step` namedtuple at `cbs_planner.py:42`:
```
plan: dict[int, list[Step]]          # robot_id → per-tick Steps
events_per_robot: dict[int, dict]    # robot_id → {t: ('pickup', bx, by, bz, si) | ('place', bx, by, bz, si)}
metadata: {
    "makespan": int,
    "sum_of_costs": int,
    "num_agents_used": int,
    "T_min": int,            # first feasible T (MILP)
    "T_final": int,          # T after cost minimization
    "solve_time": float,
}
```

The plan-replay loop (new) consumes `plan` + `events_per_robot` and:
- applies per-tick positions to `Robot.x/y/z`;
- on a `('pickup', bx, by, bz, si)` event sets `carrying=True,
  carrying_si=si`; if `bx >= 0` (a real grid pickup, e.g. MILP R_5
  scaffold tear-down), also clears `world[bz,by,bx]=0` and
  `world_sub[bz,by,bx]=0`.  The off-grid sentinel `(-1,-1,-1)` is
  used by CBS depot pickups and MILP entry-carrying so the replay
  knows there is no source voxel to clear;
- on a `('place', bx, by, bz, si)` event mutates `world[bz, by, bx] = 1`
  and `world_sub[bz, by, bx] = si + 1`, clears carrying.

---

## Gotchas found during the survey

- **Hardcoded single depot** at `(0, 0)` (`DEPOT_X, DEPOT_Y`). Paper allows
  any boundary cell. This is almost certainly why only one robot moves
  meaningfully today — every robot queues at a single pickup cell.
- **Starting positions differ across modes:** RViz spreads robots along
  `y=0, x=id%X`; matplotlib starts all at `(0,0)`. `agent_starts` must be
  explicit input to the planner.
- **Height dimension** — carrying adds 1 to z, so time-expanded graph needs
  `z_max = target_H + 1` at minimum.
- **Task ordering** — `group_task_lists` is one flat FIFO sorted by z. When
  all early tasks share a column, robots pile up at the same cell. Block
  assignment (one list per robot) fixes this.
- **Two visualizer paths** — only `MACCRvizSim` (RViz) receives planner
  wire-up. `animate_build` (matplotlib) stays heuristic-only. (Decision D-3)

---

## Decision log

### D-1 — CBS-first, then MILP (2026-04-20)

**Choice:** Wire up `cbs_planner.py` before writing the MILP.

**Why:**
1. CBS directly fixes the "only one robot moves" symptom: block-to-robot
   assignment upfront means each robot has its own trajectory instead of
   all robots fighting over a single FIFO queue. This is the load-bearing
   visual-goal fix.
2. CBS and MILP share all downstream plumbing — plan-replay loop,
   per-substructure orchestration, `--planner` flag, per-robot depots.
   Building it once for CBS is prerequisite work for MILP, not throwaway.
3. CBS gives a ground truth against which to validate the MILP encoding.
4. CBS effort is ~320 LOC + half-day to day of debugging; MILP is
   ~1000 LOC + multi-day encoding/tuning.

**Cost:** CBS is suboptimal (greedy one-sided branching; minimizes conflicts
resolved rather than cost). For the paper's Table III comparison, MILP is
still needed. CBS is validation infrastructure, not a replacement.

### D-2 — Pickup: any boundary cell (2026-04-20)

**Choice:** Open pickup to any boundary cell, not just `(0, 0)`.

**Why:** Single-depot is almost certainly part of why only one robot moves
today (all agents queue at `(0,0)`). Fixing this is load-bearing.

**How:** Simplest generalization is per-robot depot — each robot picks up
at its nearest boundary cell. CBS's `depot: (int, int)` parameter becomes
`depots: list[(int, int)]` (one per robot), baked into `_build_robot_plan`.

### D-3 — RViz-only planner wire-up (2026-04-20)

**Choice:** `--planner={heuristic,cbs,milp}` is wired into the RViz node
only. `animate_build` stays heuristic-only.

**Why:** Keep scope tight. RViz is the validation surface. Matplotlib path
is a legacy demo.

---

## CBS planner assessment

### Verdict

Working space-time planner. Never wired up, but the code is complete and
self-consistent. Not a stub, not broken. Real limitations exist but are
tractable.

### Quality — things that are right

- Standard, textbook A* in `(x, y, t)`. g-score dominance check;
  heap-ordered by `(f, h, g, ...)`.
- Tie-breaking explicitly designed against "prefer waiting" pathologies.
- Constraint format (3-tuple vertex, 5-tuple edge) is reusable as-is by
  an MILP layer later.
- `Step` namedtuple + action constants are clean reusable primitives.
- CBS fallback chain: `cbs_plan` → `_serial_fallback` → serialize.

### Quality — concerns (ranked by integration risk)

1. **Static heightmap.** `hm` is passed in once; never updated. A*
   plans using stale height data the moment another robot places a block.
   Mitigation: replan per-substructure; accept small-case stale-hm risk.
   **#1 thing to validate during Phase 1a/1b.**
2. **No carrying-conflict detection at z+1.** `_detect_first_conflict`
   checks only `(x, y)` collisions. Carrying robot occupies `(x, y, z+1)`;
   current heuristic handles this, CBS does not. Flag, not blocker.
3. **No agent cap.** All N robots get plans. To cap active agents, pre-
   assign empty block lists to the excess.
4. **Block assignment is a required upstream input.** Needs a task-to-
   robot assignment step (~20 LOC, round-robin bottom-up per parallel
   group). This is net-new code.
5. **Single depot baked in.** See D-2 — generalize to per-robot depot.
6. **Zero unit tests** — likely latent bugs (off-by-one on `t`, empty
   paths, etc.). Phase 1a adds tests.
7. **`branch_limit=50`** — aggressive for hard cases; watch for frequent
   `None` returns.
8. **One-sided branching.** Weaker than full CBS; may miss solutions.
   Rarely matters for small N.

### Integration cost estimate

| Piece | LOC | Risk |
|---|---|---|
| Task→robot assignment (round-robin bottom-up per parallel group) | ~30 | Low |
| Plan-replay loop in `MACCRvizSim.tick()` | ~80 | Medium |
| Per-substructure orchestration | ~50 | Medium |
| Per-robot depot | ~10 | Low |
| CLI + ROS parameter plumbing | ~20 | Low |
| `_serial_fallback` glue | ~10 | Low |
| `cbs_return` integration | ~40 | Medium |
| Minimal test file | ~80 | Low |

**Total: ~320 LOC + half-day to day of debugging.**

---

## Phase roadmap (current)

- **Phase 0** ✅ — Repo survey (this doc).
- **Phase 1a** — CBS unit tests, targeting stale-hm and empty-path edge
  cases identified above.
- **Phase 1b** — Wire `--planner=cbs`: plan-replay loop, task assignment,
  per-robot depots. Validate multiple robots move concurrently in RViz.
  **Decision point after Phase 1b:** continue to MILP now, or defer?
- **Phase 2** — Gurobi smoke test.
- **Phase 3** — MILP formulation (Lam et al. 2020). Pure Python.
- **Phase 4** — Parallel-construction constraint injection (MACC §IV-C).
- **Phase 5** — `--planner={heuristic,cbs,milp}` A/B comparison against
  Table III.

Per-substructure logging (all planners): `T_min`, `T_final`, `solve_time`,
`num_agents_used`. `num_agents_used == 1` should be an immediately visible
log line for future debugging.
