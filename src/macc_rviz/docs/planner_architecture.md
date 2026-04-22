# Planner Architecture

Canonical handoff doc for this repo's MILP+CBS hybrid planner. If you
are reading the codebase cold, this is the single artifact that
explains what exists, how it fits together, and why each piece is
shaped the way it is.

Cross-references:
- [`docs/benchmark_results.md`](benchmark_results.md) — six-case
  benchmark with hybrid vs CBS-alone numbers.
- [`docs/benchmark_structures.md`](benchmark_structures.md) — the
  designed input set those numbers come from.

---

## 1. Executive summary

This package implements **Multi-Agent Collective Construction**: a team
of robots places voxel blocks to build a target 3D shape. The planner
follows the decomposition pipeline of Kesarimangalam Srinivasan et al.
(IROS) — split the target into substructures, order them, plan each one
independently — and uses a **hybrid MILP+CBS architecture per
substructure**: every substructure is first attempted with the exact
MILP from Lam et al. (CP 2020); on time-budget exhaustion the CBS
adapter takes over for that substructure only. The headline result is
that **MILP decisively beats CBS on makespan whenever it can solve**
(roughly 1.75–2.92× shorter schedules across the benchmark set, see
[`benchmark_results.md`](benchmark_results.md)), and **CBS catches the
MILP scale wall** (case 4: 1 MILP-solved sub + 1 CBS-fallback sub gives
T=102 vs CBS-alone T=204). CBS is therefore correctness-and-coverage
infrastructure; MILP is the optimality engine; together they degrade
gracefully across two orders of magnitude of substructure size.

---

## 2. The three planners

Numbers below are from this repo: benchmark cases (see
[`benchmark_results.md`](benchmark_results.md)) and the Fix 2
validation runs against the 13-block `create_example_structure()`
(`structure_utils.py:27`).

| | **Heuristic** | **CBS** | **MILP** |
|---|---|---|---|
| **What it does** | Inlined per-tick reactive FSM: BFS-on-heightmap navigation; per-robot phase loop `to_pickup → pickup → to_place → place` with a priority reservation table for collisions. Lives in `MACCRvizSim.tick()` and `visualization.animate_build()`. | Space-time A* per robot + serial per-block replan with heightmap refresh between blocks. Implemented in `cbs_planner.py` (`space_time_astar`, `_build_robot_plan`, `cbs_return`) and orchestrated by `planners/cbs_adapter.py` (`plan_group`). | Exact 9-tuple action-variable formulation (Lam et al. 2020 §4 Fig. 2) solved with Gurobi: iterate `T` until feasibility, then minimise sum-of-costs at that `T`. Implemented in `planners/milp_encoding.py`, `milp_planner.py`, `milp_adapter.py`. |
| **Strengths** | Effectively zero solve cost; always produces *some* motion; no external solver dep. | Fast (<1 s on every case in the set); no LP/MIP solver dependency; deterministic; serial-per-block design eliminates within-substructure clipping. | Provably optimal (or close, at the gap reported by Gurobi) on substructures it can solve; finds genuine multi-agent parallelism within a substructure. |
| **Weaknesses** | No lookahead, no parallelism guarantees, no formal correctness on conflicts. Kept in the tree as a baseline only. | Greedy: serial within a substructure (no within-sub parallelism), so makespans are roughly 2× longer than MILP on the cases where MILP completes. | Variable count and time-expanded horizon explode together — solves at the >15-block sub size start to dominate the wall budget; >24-block subs are out of reach on this hardware. |
| **Solve time (order of magnitude)** | μs–ms (no solve, just BFS) | sub-ms to ms (e.g. 0.4 ms on the 13-block example structure; 0.06 s on the 27-block case 4) | seconds to thousands of seconds (case 1 sub 0: ~4 s for 3 blocks; case 4 sub 0: ~1071 s for 18 blocks; case 6's 24-block sub did not complete) |
| **Typical makespan quality** | Whatever the FSM produces — no formal guarantee. | Across cases 1–4: T ∈ {35, 93, 181, 204} | Across cases 1–4 (hybrid): T ∈ {20, 42, 62, 102} |

**13-block `create_example_structure()` head-to-head (Fix 2 validation,
`/tmp/validate_scaffold_e2e.py` and `/tmp/validate_cbs_example_e2e.py`):**

| | T_final | MILP `obj` (sum-of-costs) | Solve time |
|---|---:|---:|---:|
| MILP   | 20 | 60.0 | ~95 s   |
| CBS    | 84 | n/a  | ~0.4 ms |

(CBS does not optimise sum-of-costs, so an apples-to-apples SoC
comparison is not meaningful — `obj` here is Gurobi's MILP objective at
the final T. The benchmark table above uses `T_final` as the consistent
comparator.)

---

## 3. Architecture

```
┌──────────────────────┐
│ target structure     │  numpy (Z, Y, X), 1=occupied
│ (Z,Y,X) np.ndarray   │  e.g. create_example_structure(),
└──────────┬───────────┘       create_random_structure(seed=...)
           │
           ▼
┌──────────────────────┐
│ decomposition.py     │  Algorithm 1 (shadow-region split)
│  decompose_structure │  → list[ndarray]  (one sub per region)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ decomposition.py     │  Algorithm 3
│  compute_dependencies│  → DAG
│  order_substructures │  → topological build order
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ parallel.py          │  Kahn's algorithm
│  find_parallel_groups│  → list[list[sub_idx]]
└──────────┬───────────┘
           │
           ▼  one parallel group at a time
┌────────────────────────────────────────────────────────┐
│ planners/milp_adapter.py::plan_group                   │
│                                                        │
│  for each substructure si in group:                    │
│    ┌──────────────────────┐                            │
│    │ milp_planner         │  iterate T until feasible, │
│    │  .plan_structure(si) │  then minimise SoC at T    │
│    └──────────┬───────────┘                            │
│               │                                        │
│        ┌──────┴──────┐                                 │
│        │ feasible?   │ no  ┌────────────────────────┐  │
│        │             ├────►│ cbs_adapter.plan_group │  │
│        │             │     │  (single sub, fallback)│  │
│        └──────┬──────┘     └──────────┬─────────────┘  │
│           yes │                       │                │
│               ▼                       ▼                │
│         ┌─────────────────────────────────────┐        │
│         │  per-trip Steps → per-robot plans   │        │
│         │  (greedy earliest-free assignment;  │        │
│         │   pad WAITs at off-grid home cell)  │        │
│         └────────────────────┬────────────────┘        │
└──────────────────────────────┼─────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│ planners/cbs_adapter.py::plan_return                     │
│   simultaneous return-to-boundary (CBS, with serial      │
│   fallback)                                              │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│ macc_rviz_sim.py::MACCRvizSim                            │
│   _tick_cbs(): advance each robot 1 Step per tick        │
│   _apply_cbs_step(): apply pickup/place events to        │
│     world / world_sub; publish RViz MarkerArrays         │
└──────────────────────────────────────────────────────────┘
```

For the `--planner=cbs` path the same diagram applies, except the MILP
node is skipped and `cbs_adapter.plan_group` is called directly per
substructure — there is no "fallback" because CBS *is* the primary.

---

## 4. The MILP encoding

Clean-room reimplementation of **Lam, Stuckey, Koenig & Kumar — "Exact
Approaches to the Multi-Agent Collective Construction Problem", CP
2020, §4 Fig. 2** (paper PDF intentionally not committed; see the
`gitignore`'d `docs/reference_paper.pdf`). Paper notation is preserved
constraint-by-constraint in `planners/milp_encoding.py`:

- **Action 9-tuple:** `i = (t, x, y, z, c, a, x', y', z')` —
  `(time, source-cell, carrying-state, action-type, dest-cell)`.
- **Action types `K = {M, P, D}`** — move/wait, pickup, deliver.
- **Off-grid sentinels** `S` (source for `R_1` agent entries) and `E`
  (destination for `R_4` agent exits). Agents enter the grid at any
  border cell carrying a block, and leave the grid at any border cell.
- **Constraint subsets `R_1..R_6`** posted in Fig. 2 order in the
  module — every constraint label in the file matches the paper.
- **Height variables `H`** track per-cell column heights with the
  invariant `|z' − z| ≤ 1` between adjacent ticks (no levitation).
- **Two-stage solve** in `milp_planner.plan_structure`:
  1. Iterate `T = T_min, T_min+1, …, T_max`. The first `T` at which
     Gurobi proves feasibility is `T_final`.
  2. At `T_final`, re-solve with the sum-of-costs objective active to
     produce the final action variable assignment.
- **Decode:** `milp_decode.py` walks the assigned 9-tuples to produce
  per-trip `(steps, events)` lists in the same shape the CBS path
  produces. The trip-to-robot greedy fuser lives in
  `milp_adapter._fuse_trips_into_plans`.

See `planners/milp_encoding.py` for the constraint-by-constraint
implementation. The module-level docstring lists the paper sections
referenced by each helper.

---

## 5. The CBS planner

### 5.1 Architecture: serial per-block with heightmap refresh

`cbs_adapter.plan_group` does not do a joint multi-agent solve. For
each block in the parallel group, in round-robin order across robots
within each `z` layer, it:

1. **Recomputes the heightmap from `cur_world`** (the current built
   state) — `assert_hm_matches_world` re-fires here as an invariant
   guard that fails loudly if the bookkeeping ever drifts.
2. Calls `cbs_planner._build_robot_plan(cur_starts[ri], [block],
   block_hm, depots[ri], cur_t, frozenset(), max_t)` — a single robot
   makes one trip (depot → pickup → placement-cell → place).
3. **Pads every other robot with `WAIT`-in-place Steps** at its
   current cell from `cur_t+1` through the active robot's last tick,
   so the replay loop (which advances each robot one Step per tick
   linearly in `MACCRvizSim._tick_cbs`) stays temporally aligned.
4. Applies the block's place event to `cur_world`, advances `cur_t`,
   updates the active robot's start cell, and moves on.

This is `~0.4 ms` for the 13-block example structure and stays
sub-second on every case in the benchmark set.

### 5.2 Why serial — the clipping bug story

CBS used to plan within-round in parallel. That was wrong, in a way
that took two iterations to fully diagnose:

**Bug 1 — stale world state on grid pickups (commit `bcc134a`,
"fix(replay): clear world on grid pickups so MILP scaffolds don't
linger").** The replay loop didn't clear `world[bz, by, bx]` when a
pickup event named a real source voxel, so MILP-emitted scaffold
tear-downs left phantom blocks behind. Fix: distinguish off-grid
sentinel pickups (`(-1, -1, -1)`, used by CBS depot pickups and MILP
agent entries) from grid pickups; only the latter mutate `world`.

**Bug 2 — within-round CBS clipping (commit `84c4ef6`, "fix(cbs):
serial per-block replan eliminates all clipping").** With Bug 1 fixed,
the example structure still showed visual clipping during CBS replay
— robots walking *through* a stack that another robot had just
placed. Root cause: `cbs_plan` solved with one shared `hm` per call,
so even an intra-round joint solve could route robot N's plan through
column C at `z=0` after robot M (M<N) had already placed at column
C earlier in the same round. A single `_replay_and_check_no_clip`
diagnostic against the central tower counted exactly 12 such intra-
round clips. The serial-per-block rewrite refreshes `hm` *between
every block*, eliminating clipping deterministically (verified to
reach exactly 0 clips on the central tower and the example
structure). The trade-off is loss of within-substructure CBS
parallelism — acceptable because the MILP path provides that
parallelism in the hybrid; CBS's role is now correctness and
fallback.

Regression coverage for both bugs lives in
`test/test_scaffold_cleanup.py` (Bug 1) and `test/test_cbs_no_clip.py`
(Bug 2).

### 5.3 Heightmap invariant

`cbs_adapter.assert_hm_matches_world(hm, world, context)` is a project-
wide load-bearing guard. Anywhere a planner consumes `hm`, the caller
must have just derived it from the world it claims to be planning
against. The serial-per-block path re-asserts this *before every
block iteration* — if it fires, a prior iteration's bookkeeping is
broken and you want a loud failure now, not a silent clip later.

---

## 6. The hybrid fallback

`milp_adapter.plan_group` runs each substructure in the group through
`milp_planner.plan_structure`. When that returns `T=None` (infeasible
at every `T` in the sweep within the time budget), the adapter calls
`cbs_adapter.plan_group([just_this_sub], ...)` and splices its
per-robot Steps and events into the accumulator at the current
`t_offset`. Subsequent substructures (and groups) keep running
against the updated `world` snapshot — the prior substructures'
placements are baked into `init_hm` for the next MILP call so heights
stay consistent.

Per-substructure metadata records `planner_used ∈ {"milp",
"cbs_fallback"}` so the benchmark runner can tabulate which solver
actually produced each sub's plan.

**Canonical example — Case 4 from the benchmark
(`docs/benchmark_results.md` §"Per-substructure breakdown"):**

| sub | blocks | planner       | T  | solve (s) |
|----:|------:|---------------|---:|----------:|
| 0   | 18    | milp          | 44 |   1071.19 |
| 1   |  9    | cbs_fallback  | 58 |      0.03 |

Sub 0's MILP solve consumed enough of the 600 s soft total budget
that sub 1's `per_t_limit` was clamped to 1.0 s, at which point Gurobi
could not find a feasible solution for sub 1 at any `T ≤ T_max=150`.
The adapter logged the fallback honestly:

```
[MILP] sub 1: MILP infeasible at all T <= T_max after 374.93s,
              falling back to CBS (blocks=9, T_max=150, per_t_limit=1.0s)
[MILP] sub 1: planner=cbs_fallback T_contribution=58 cbs_solve=0.03s
              cbs_serial_fallback=True blocks=9
```

The hybrid produced **T=102** for case 4, vs **CBS-alone T=204** —
the MILP-solved sub keeps the makespan halved even when its sibling
falls through to CBS.

---

## 7. Known limitations / residual caveats

- **Single parallel group in practice.** All six benchmark structures
  (and the example structure) decompose into exactly one parallel
  group. This is a property of the shadow-based decomposition: every
  column is assigned to a single sub, so the inter-sub dependency
  check `np.roll(s2, 1, axis=0) & s1` never fires. Within-group
  parallelism remains bounded by `num_robots=4`. Parallel-group
  constraint injection (PAR-V/PAR-E) is exercised only by Phase 4
  unit tests.
- **MILP scale wall around 18–24 blocks per sub** on this hardware
  (ARM64 VM, 600 s soft budget). Case 4's 18-block sub took 1071 s;
  case 6's 24-block sub did not complete within feasible wall time.
  This matches Lam et al.'s own published wall times — they report
  >10,000 s on three of six instances and 5.7 days on the hardest.
- **CBS is strictly serial within a substructure.** Until commit
  `84c4ef6` it tried within-round parallelism; that was buggy (see
  §5.2). Reintroducing safe within-round parallelism is future work
  (§10).
- **`step_duration_sec` is a node parameter** (defaults to 0.4 s) but
  is **not** exposed as a launch argument — overrides require
  `--ros-args -p step_duration_sec:=...`.
- **Single fixed depot at `(DEPOT_X, DEPOT_Y) = (0, 0)`** for the
  heuristic path. CBS and MILP both use per-robot nearest-boundary
  depots (`cbs_adapter.per_robot_depots`); the heuristic was not
  back-ported.
- **Two visualizer paths.** `--planner=cbs|milp` is wired into the
  RViz node only; `visualization.animate_build` (the standalone
  matplotlib path) stays heuristic-only by design.

---

## 8. How to run

```bash
# Build (from workspace root)
colcon build --packages-select macc_rviz
source install/setup.bash

# Heuristic (default)
ros2 launch macc_rviz macc.launch.py

# CBS planner
ros2 launch macc_rviz macc.launch.py planner:=cbs

# MILP planner (with CBS fallback)
ros2 launch macc_rviz macc.launch.py planner:=milp

# Use the 13-block example structure (otherwise: random per seed)
ros2 launch macc_rviz macc.launch.py planner:=milp use_example_structure:=true

# Reproducible random target
ros2 launch macc_rviz macc.launch.py planner:=milp seed:=42

# Tune solver budgets (see launch/macc.launch.py for full list)
ros2 launch macc_rviz macc.launch.py \
    planner:=milp \
    milp_per_t_time_limit:=120.0 \
    milp_total_time_limit:=900.0 \
    milp_T_max:=60 \
    cbs_max_t:=400
```

Key launch arguments (see `launch/macc.launch.py`):

| Arg | Default | Notes |
|---|---|---|
| `planner` | `heuristic` | one of `heuristic`, `cbs`, `milp` |
| `milp_per_t_time_limit` | `60.0` | Gurobi `TimeLimit` for each `T` in the sweep |
| `milp_total_time_limit` | `600.0` | soft total budget per group; shrinks `per_t` as it drains |
| `milp_T_max` | `40` | hard cap on the per-sub `T` sweep |
| `milp_mip_gap` | `0.0` | Gurobi `MIPGap` (0 = prove optimal) |
| `cbs_max_t` | `400` | A*/CBS hard time horizon per call |
| `cbs_branch_limit` | `500` | max CBS branches before serial fallback |
| `num_robots` | `4` | |
| `use_example_structure` | `false` | use the hand-built 13-block structure |
| `seed` | `-1` | `-1` = fresh random per run |

To run the full benchmark suite:

```bash
python3 scripts/run_benchmarks.py    # cases 1-5; case 6 needs raised limits
```

Outputs go to `docs/bench_full.log` / `docs/bench_rest.log`; the
write-up is `docs/benchmark_results.md`.

---

## 9. Development history

Single timeline. Commit hashes are authoritative.

| Phase | Commits | What landed |
|---|---|---|
| **Pre-history** | `beef235`, `aae3efc`, `d98efad`, `01f7fe6`, `78b0382`, `6ba584e` | Initial RViz node, random structure generator, decomposition + parallel-group code, BFS heuristic FSM, cosmetic fixes. The "only one robot moves" symptom was visible from this era — every robot queued at the single depot `(0, 0)`. |
| **Phase 0 — survey** | `5469845` | Read the heuristic FSM, audit `cbs_planner.py`, decide CBS-first then MILP. (See §A in the appendix below.) |
| **Phase 1a — CBS unit tests** | `ab2d8ad` | First test file for `cbs_planner`; targeted stale-`hm` and empty-path edge cases identified during the survey. |
| **Phase 1b — wire CBS** | `b442901`, `3db7793` | `--planner=cbs` plumbing, plan-replay loop in `MACCRvizSim`, per-robot depots, block-to-robot round-robin assignment. Multiple robots now visibly move. |
| **Phase 2 — Gurobi smoke test** | (ad-hoc; no functional commit) | Verified Gurobi import + license on the dev VM. (See §A in the appendix.) |
| **Phase 3 — MILP encoding** | `faf3add` (Phase 5 Part B-Revised step 1) | Lam et al. constraints (2)–(16) implemented in `milp_encoding.py`; trip decoder in `milp_decode.py`. |
| **Phase 5 Part B — hybrid fallback** | `6c44fbd` | `milp_adapter.plan_group` falls back to `cbs_adapter.plan_group` per substructure on MILP infeasibility / budget exhaustion. |
| **Phase 5 Part C — benchmark** | `b2e8cec`, `3373a01` | Six designed benchmark structures, hybrid-vs-CBS-alone runner, raw logs preserved. |
| **`use_example_structure` launch arg** | `e9b655e` | Lifted from node-only param to launch argument so `planner:=milp use_example_structure:=true` works end-to-end. |
| **Fix 1 — scaffold replay** | `bcc134a` | Replay clears `world[bz,by,bx]` on grid pickups; off-grid sentinel `(-1,-1,-1)` is recognised and skipped. |
| **Fix 2 — CBS serial per-block** | `84c4ef6` | `cbs_adapter.plan_group` rewritten as serial-per-block with per-block `hm` refresh; eliminates the 12 residual within-round clips on the example structure. |

---

## 10. Future work

- **Reintroduce safe within-round CBS parallelism.** Option 4 from the
  Fix 2 decision discussion: when a robot's plan would traverse a
  column at a tick after another robot's same-round place event, pre-
  inject a vertex constraint at `(cx, cy, t_place)` for `z < new_h`.
  Requires teaching `cbs_plan` to consume placement-derived
  constraints rather than just inter-robot conflict-derived ones.
  Would restore the original within-round speedup without sacrificing
  correctness.
- **Decomposition improvements to yield multiple parallel groups.**
  The shadow-based heuristic always assigns a column to one
  substructure, so PAR-V/PAR-E never gets exercised on real
  structures. A column-splitting (or layer-splitting) decomposition
  would surface the constraint-injection path on real benchmarks
  rather than only in unit tests.
- **Expose `step_duration_sec` as a launch argument.** Currently a
  node parameter only — minor plumbing fix in `launch/macc.launch.py`.
- **MILP encoding optimisations for larger substructures.** The
  variable count grows roughly with `|R| × T`; on case 6's 24-block
  sub the encoding doesn't fit in any reasonable wall budget. Symmetry
  breaking, warm starts from CBS solutions, or a column-generation
  reformulation are all worth investigating for the >20-block regime.

---

## Appendix A — original survey & decision log (Phase 0 / 1)

Preserved verbatim from the original `planner_architecture.md`. Useful
context for *why* the architecture is shaped this way.

### A.1 Module map (legacy)

| Role | File | Symbol(s) |
|---|---|---|
| Algorithm 1 — shadow-region decomposition | `macc_rviz/decomposition.py` | `decompose_structure(structure)` → `list[np.ndarray]` |
| Algorithm 3 — ordering & dependency DAG | `macc_rviz/decomposition.py` | `compute_dependencies`, `order_substructures` |
| Parallel-group detector (Kahn) | `macc_rviz/parallel.py` | `find_parallel_groups(substructures, deps)` → `list[list[int]]` |
| Visualizer (RViz / ROS2) — production path | `macc_rviz/macc_rviz_sim.py` | `MACCRvizSim` node |
| Visualizer (standalone matplotlib) | `macc_rviz/visualization.py` | `show_construction_process`, `animate_build` |
| Space-time planner | `macc_rviz/cbs_planner.py` | `space_time_astar`, `cbs_plan`, `cbs_return`, `_build_robot_plan`, `_serial_fallback` |
| Reusable plan primitives | `cbs_planner.py` L42–47 | `Step(action,x,y,z,t)`, `ACTION_WAIT/MOVE/PICKUP/PLACE` |
| CBS orchestration | `macc_rviz/planners/cbs_adapter.py` | `plan_group`, `plan_return`, `assert_hm_matches_world` |
| MILP encoding (Lam et al. CP 2020) | `macc_rviz/planners/milp_encoding.py` | constraints (2)–(16) |
| MILP solve loop + decode | `macc_rviz/planners/milp_planner.py`, `milp_decode.py` | `plan_structure` |
| MILP→CBS hybrid orchestration | `macc_rviz/planners/milp_adapter.py` | `plan_group` |

### A.2 Decision log

**D-1 — CBS-first, then MILP (2026-04-20).** Wired CBS before writing
MILP because: (1) it directly fixes the "only one robot moves" symptom
via per-robot block assignment; (2) all downstream plumbing (replay
loop, `--planner` flag, per-robot depots, per-substructure
orchestration) is shared with the eventual MILP path; (3) CBS provides
ground-truth output to validate the MILP encoding against; (4) the
effort gap (~320 LOC for CBS vs ~1000 LOC for MILP) makes CBS the
cheaper way to surface integration bugs.

**D-2 — Pickup at any boundary cell (2026-04-20).** Generalised the
hardcoded single depot at `(0, 0)` to per-robot nearest-boundary
depots in `cbs_adapter.per_robot_depots`. Single-depot was a major
contributor to the "only one robot moves" symptom — every agent queued
at one cell.

**D-3 — RViz-only planner wire-up (2026-04-20).**
`--planner={heuristic,cbs,milp}` is wired into the `MACCRvizSim` node
only. `animate_build` (matplotlib) stays heuristic-only by design.
Scope-tightening choice: RViz is the validation surface, matplotlib is
a legacy demo.

### A.3 Original CBS planner concerns (Phase 0)

The Phase 0 survey flagged eight concerns about the pre-existing
`cbs_planner.py`. Status now:

| # | Concern | Status |
|---|---|---|
| 1 | Static heightmap (stale-`hm` risk) | **Resolved by Fix 2** (`84c4ef6`); `cbs_adapter.plan_group` refreshes `hm` per block. |
| 2 | No carrying-conflict detection at `z+1` | Open. Heuristic handles it; CBS does not. Has not surfaced as a visual bug at the sizes we run. |
| 3 | No agent cap | Resolved by per-robot block assignment (excess robots get empty queues). |
| 4 | Block-to-robot assignment was an unstated upstream input | Resolved by `cbs_adapter.assign_blocks_group` (round-robin bottom-up). |
| 5 | Single depot baked in | Resolved (D-2). |
| 6 | Zero unit tests | Resolved by Phase 1a + Fix 2 regression suites (`test_cbs_planner.py`, `test_cbs_adapter.py`, `test_cbs_no_clip.py`). |
| 7 | `branch_limit=50` aggressive | Default raised to 500 in launch; serial-per-block design no longer branches at all (parameter kept for ABI). |
| 8 | One-sided branching weaker than full CBS | Moot — serial-per-block doesn't branch. |

### A.4 Phase 2 — Gurobi smoke test

Verified on the dev VM that `gurobipy` imports, the academic license
recognises the host, and a trivial 2-variable LP solves to optimality.
No functional commit — purely environmental check that the MILP path
of Phase 3 would have a working solver to call into.
