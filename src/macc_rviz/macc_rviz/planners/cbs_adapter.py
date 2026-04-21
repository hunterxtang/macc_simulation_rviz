"""
CBS adapter: bridges the decomposition pipeline to cbs_planner.

Responsibilities:
  - Compute per-robot depots (nearest boundary cell to each robot).
  - Assign blocks to robots (round-robin bottom-up per parallel group).
  - Call cbs_plan; fall back to _serial_fallback on None return.
  - Build heightmap defensively (with an invariant check against the world
    state the caller claims is current).
  - Emit per-substructure metadata for logging.

This module is pure Python — no ROS imports — so it is unit-testable
standalone and can be reused by the MILP path in later phases.
"""

import time

import numpy as np

from macc_rviz import cbs_planner
from macc_rviz.cbs_planner import ACTION_WAIT, cbs_return, Step


# ---------------------------------------------------------------------------
# Heightmap and world-state helpers
# ---------------------------------------------------------------------------

def heightmap_from_world(world):
    """Return the 2D heightmap derived from a 3D world array (Z, Y, X)."""
    return np.sum(world, axis=0).astype(int)


def assert_hm_matches_world(hm, world, context=''):
    """Fail loudly when a heightmap diverges from its backing world state.

    Guards the stale-hm invariant: the CBS plan-replay loop must recompute
    ``hm`` each time it switches substructures. If this assertion fires,
    the replay loop regressed — see docs/planner_architecture.md §D-1.
    """
    expected = heightmap_from_world(world)
    if not np.array_equal(hm, expected):
        raise AssertionError(
            f'stale heightmap detected ({context}): hm and world disagree. '
            f'The plan-replay loop MUST recompute hm per substructure. '
            f'hm.sum={int(hm.sum())}, world.sum={int(world.sum())}.'
        )


# ---------------------------------------------------------------------------
# Per-robot depots
# ---------------------------------------------------------------------------

def nearest_boundary_cell(x, y, X, Y):
    """Return the (x, y) boundary cell closest to (x, y) by Manhattan distance.

    Ties broken by iteration order: left, right, bottom, top.
    """
    options = [(0, y), (X - 1, y), (x, 0), (x, Y - 1)]
    return min(options, key=lambda c: abs(c[0] - x) + abs(c[1] - y))


def per_robot_depots(robot_starts, X, Y):
    """Return [nearest_boundary_cell for each robot_start]."""
    return [nearest_boundary_cell(x, y, X, Y) for (x, y) in robot_starts]


# ---------------------------------------------------------------------------
# Block-to-robot assignment
# ---------------------------------------------------------------------------

def blocks_in_substructure_bottom_up(sub, substructure_index):
    """Return [(x, y, z, substructure_index), ...] sorted bottom-up by z."""
    coords = []
    Z, Y, X = sub.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if sub[z, y, x]:
                    coords.append((x, y, z, substructure_index))
    return coords


def assign_blocks_round_robin(substructure, substructure_index, num_robots):
    """Assign blocks from one substructure to num_robots via round-robin.

    Blocks are ordered bottom-up first (paper's construction precondition:
    a block at z>0 requires its support at z-1 already in place). Within
    one layer, robots take turns.

    Returns: list[list[(x, y, z, si)]] of length num_robots.
    """
    all_blocks = blocks_in_substructure_bottom_up(
        substructure, substructure_index
    )
    assignments = [[] for _ in range(num_robots)]
    for i, block in enumerate(all_blocks):
        assignments[i % num_robots].append(block)
    return assignments


def assign_blocks_group(substructures, indices, num_robots):
    """Assign all blocks from a parallel group to num_robots.

    ``substructures`` is a list of Z×Y×X int arrays (one per member of
    the parallel group); ``indices`` is the list of corresponding
    substructure indices. Blocks are interleaved substructure-by-
    substructure in round-robin across robots, then sorted globally
    bottom-up within each robot's list (respecting the no-float rule).

    Returns: list[list[(x, y, z, si)]] of length num_robots.
    """
    assignments = [[] for _ in range(num_robots)]
    cursor = 0
    for sub, si in zip(substructures, indices):
        for block in blocks_in_substructure_bottom_up(sub, si):
            assignments[cursor % num_robots].append(block)
            cursor += 1
    # Within each robot, sort bottom-up so early placements respect
    # the precondition that z-1 must be filled before z.
    for i in range(num_robots):
        assignments[i].sort(key=lambda b: (b[2], b[3], b[1], b[0]))
    return assignments


# ---------------------------------------------------------------------------
# Per-substructure planning
# ---------------------------------------------------------------------------

def _pad_idle_robots_with_waits(
    plans, idle_indices, from_t, to_t, cur_starts, hm,
):
    """Append WAIT-in-place Steps from ``from_t`` to ``to_t`` (inclusive).

    Used while one robot executes its single-block trip: every other
    robot must produce a Step per tick at its current cell so the
    replay loop (``MACCRvizSim._tick_cbs``, which advances each robot
    one Step per tick linearly) keeps every robot temporally aligned
    with the active robot's plan.

    No-op when ``to_t < from_t`` (active robot's plan was empty or
    instantaneous).
    """
    if to_t < from_t:
        return
    for ri in idle_indices:
        x, y = cur_starts[ri]
        z = int(hm[y, x])
        for tt in range(from_t, to_t + 1):
            plans[ri].append(Step(ACTION_WAIT, x, y, z, tt))


def _flatten_blocks_round_robin(assignments):
    """Interleave per-robot block queues into a global per-block schedule.

    Blocks come out in the order they would be placed if every robot
    took one turn at a time, in robot-id order.  Within a single
    "round" (one slot per robot), blocks of the same z layer are
    issued before any block of the next layer — this falls out of the
    bottom-up sort that ``assign_blocks_group`` already applied to
    each robot's queue, so layer N's blocks finish before layer N+1
    starts (preserves the no-floating-block precondition).

    Returns: list of (robot_id, (bx, by, bz, si)) in placement order.
    """
    out = []
    if not assignments:
        return out
    max_len = max(len(a) for a in assignments)
    for round_idx in range(max_len):
        for ri, queue in enumerate(assignments):
            if round_idx < len(queue):
                out.append((ri, queue[round_idx]))
    return out


def plan_group(
    substructures,
    substructure_indices,
    world_init,
    robot_starts,
    num_robots,
    X,
    Y,
    t_start=0,
    max_t=400,
    branch_limit=50,
):
    """Plan a parallel group with **serial per-block** replanning.

    A "group" is either a single substructure (serial) or several
    substructures that the decomposition allows to be built concurrently.
    Blocks from all substructures in the group are merged and placed
    one at a time, in round-robin order across robots within each
    z layer.  Each block placement runs a single-robot
    ``_build_robot_plan`` against a ``hm`` freshly recomputed from
    ``cur_world`` — so the planner always sees the true height of every
    column the robot will traverse.  Other robots pad WAIT-in-place
    Steps for the duration of the active robot's trip, keeping every
    robot's plan length aligned tick-for-tick (the replay loop in
    ``MACCRvizSim._tick_cbs`` advances each robot one Step per tick).

    Why serial?  The previous attempts (single-shot ``cbs_plan`` and
    per-round CBS) suffered from stale-``hm`` clipping: when robot M
    placed a block earlier in a joint plan, robot N's path through
    the same column kept the ``z=0`` it was planned with, walking
    visually *through* the placed stack.  Truly serial per-block
    eliminates this deterministically — at the cost of within-group
    parallelism.  CBS is now the safety / correctness fallback;
    parallelism is the MILP path's responsibility.

    Parameters
    ----------
    substructures : list of ndarray (Z, Y, X)
        One per group member.
    substructure_indices : list of int
        Corresponding substructure indices (for world_sub bookkeeping).
    world_init : ndarray (Z, Y, X)
        Built world the plan will execute against. The caller promises
        this equals the true current world; the adapter asserts.
    robot_starts : list[(x, y)]
    num_robots, X, Y : int
    t_start : int
        Absolute timestep at which this group's plan begins.
    max_t : int
        Hard A* time horizon, forwarded to ``_build_robot_plan``.
    branch_limit : int
        Accepted for ABI compatibility with the old joint-CBS API; no
        longer used (no joint solve, no branching).

    Returns
    -------
    dict with keys:
      plans    : list[list[Step]] — one per robot, steps in absolute-t,
                 every robot's list ends at the same final tick.
      events   : list[dict] — one per robot, {t: event_tuple}.
      assignments : per-robot block queues (for telemetry).
      metadata : dict with T_min, T_final, solve_time, num_agents_used,
                 used_fallback, conflicts_resolved, block_count, blocks_planned.
    """
    del branch_limit  # No joint CBS solve in the serial-per-block design.

    assert_hm_matches_world(
        heightmap_from_world(world_init), world_init,
        context=f'group with substructures {substructure_indices} (pre-plan)',
    )

    assignments = assign_blocks_group(
        substructures, substructure_indices, num_robots
    )
    depots = per_robot_depots(robot_starts, X, Y)
    schedule = _flatten_blocks_round_robin(assignments)

    plans = [[] for _ in range(num_robots)]
    events = [{} for _ in range(num_robots)]
    cur_world = world_init.astype(int).copy()
    cur_starts = list(robot_starts)
    cur_t = t_start
    blocks_planned = 0
    used_fallback = False

    t0 = time.perf_counter()
    for ri, block in schedule:
        # Refresh hm from cur_world before *every* block so the active
        # robot's plan sees the latest column heights.  Re-assert the
        # invariant — if cur_world drifts from hm here, a prior
        # iteration's bookkeeping regressed and we want a loud failure.
        block_hm = heightmap_from_world(cur_world)
        assert_hm_matches_world(
            block_hm, cur_world,
            context=(
                f'group {substructure_indices} block #{blocks_planned} '
                f'(robot {ri}, target {block}) — per-block refresh'
            ),
        )

        result = cbs_planner._build_robot_plan(
            cur_starts[ri], [block], block_hm, depots[ri],
            cur_t, frozenset(), max_t,
        )
        if result is None or not result[0]:
            # Single-block plan failed (unreachable target / depot or
            # _placement_cell returned None).  Skip this block so the
            # group can finish; mark the run as degraded so milp_adapter
            # surfaces it in telemetry.
            used_fallback = True
            continue

        active_steps, active_events = result
        active_end_t = active_steps[-1].t

        # Other robots wait at their current cells for the duration of
        # this block's plan; their z is read from the freshly-computed hm.
        idle = [i for i in range(num_robots) if i != ri]
        _pad_idle_robots_with_waits(
            plans, idle, cur_t + 1, active_end_t, cur_starts, block_hm,
        )

        plans[ri].extend(active_steps)
        for tev, ev in active_events.items():
            events[ri][tev] = ev
            if ev[0] == 'place':
                _, bx, by, bz, _esi = ev
                cur_world[bz, by, bx] = 1

        cur_starts[ri] = (active_steps[-1].x, active_steps[-1].y)
        cur_t = active_end_t
        blocks_planned += 1

    solve_time = time.perf_counter() - t0

    plan_makespan = max(
        (p[-1].t for p in plans if p),
        default=t_start,
    )
    num_agents_used = sum(1 for p in plans if any(
        s.action != ACTION_WAIT for s in p
    ))
    block_count = sum(int(s.sum()) for s in substructures)

    return {
        'plans': plans,
        'events': events,
        'assignments': assignments,
        'metadata': {
            'substructure_indices': list(substructure_indices),
            'T_min': plan_makespan,           # CBS: T_min == T_final
            'T_final': plan_makespan,
            'solve_time': solve_time,
            'num_agents_used': num_agents_used,
            'used_fallback': used_fallback,
            'conflicts_resolved': 0,          # no joint CBS in serial mode
            'block_count': block_count,
            'blocks_planned': blocks_planned,
        },
    }


# ---------------------------------------------------------------------------
# Return-to-boundary
# ---------------------------------------------------------------------------

def plan_return(robot_starts, X, Y, world_state, t_start=0, max_t=200):
    """Plan simultaneous return-to-boundary for all robots.

    Returns list[list[Step]] on success, or a serialised fallback plan if
    CBS returns None. Never returns None; the caller can always replay.
    """
    hm = heightmap_from_world(world_state)
    assert_hm_matches_world(hm, world_state, context='return phase')

    goals = [nearest_boundary_cell(x, y, X, Y) for (x, y) in robot_starts]

    result = cbs_return(robot_starts, goals, hm, t_start=t_start, max_t=max_t)
    if result is not None:
        return result

    # Serial fallback: plan each robot's navigation one after another.
    plans = []
    t = t_start
    for (start, goal) in zip(robot_starts, goals):
        if start == goal:
            plans.append([])
            continue
        p = cbs_planner.space_time_astar(start, goal, hm, t_start=t, max_t=max_t)
        if p is None:
            # Truly unreachable — stay put; replay loop will notice and exit.
            plans.append([])
            continue
        plans.append(p)
        t = p[-1].t + 1 if p else t
    return plans
