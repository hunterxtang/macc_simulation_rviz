"""MILP adapter: bridges the decomposition pipeline to ``milp_planner``.

The MILP solver emits one plan *per trip* (enter → deliver → exit), not
one plan per physical robot.  A single agent can perform several trips
over the horizon, separated by R_4 exits and R_1 re-entries.  This
module translates that trip-oriented output into the per-robot Step /
event shape the simulator already knows how to replay (the CBS path).

Greedy earliest-free-robot scheduling:
  - Sort trips by entry tick.
  - For each trip, give it to the robot whose plan ends soonest.
  - Between trips, pad the robot's plan with WAITs at a fixed off-grid
    "home" cell so the replay sees one Step per tick.

Parallel-group handling: members of one ``find_parallel_groups`` group
are solved sequentially by the MILP (one ``plan_structure`` call per
substructure), and their per-substructure trips are concatenated in
order with a t-offset so the replay runs sub 0 → sub 1 → … end-to-end.
Each later MILP sees the prior substructures as already in place via
an updated ``init_hm``; no prior-trajectory collisions are needed
because the replay serialises their executions.  This trades
within-group parallelism for MILP tractability — the monolithic
group-level MILP explodes in variable count at realistic problem
sizes (82 blocks → lower-bound T ≈ 64, well beyond what Gurobi can
prove optimal in minutes).  True multi-agent parallelism still
happens *within* each substructure's own MILP solve.

MILP→CBS fallback: when ``plan_structure`` returns ``T=None`` for a
substructure (infeasible at every T in the sweep within the time
budget), the adapter calls ``cbs_adapter.plan_group`` for that single
substructure and splices its per-robot Steps into the accumulator at
the current ``t_offset``.  Subsequent substructures (and groups) keep
running.  Per-substructure ``planner_used`` ("milp" | "cbs_fallback")
in the returned ``substructure_metadata`` lets Part C tabulate which
solver actually produced each sub's plan.

Module-level ``plan_return`` delegates to ``cbs_adapter.plan_return``
so the return-to-boundary phase is planner-agnostic.
"""

import time

import numpy as np

from macc_rviz.cbs_planner import ACTION_WAIT, Step
from macc_rviz.planners import cbs_adapter
from macc_rviz.planners.milp_planner import plan_structure


# Home-row y offset for idle / pre-first-trip robot parking (off-grid).
HOME_Y_OFFSET = -2


def _heightmap(world):
    return np.sum(world, axis=0).astype(int)


def _target_sub_from_substructures(substructures, indices, shape):
    """Build a (Z, Y, X) int array tagging each voxel with its si, -1 elsewhere."""
    target_sub = np.full(shape, -1, dtype=int)
    for sub, si in zip(substructures, indices):
        target_sub[sub == 1] = si
    return target_sub


def _fuse_trips_into_plans(trips, trip_events, plans, events, next_free, homes):
    """Greedy-assign trips to per-robot plans, in place.

    Each trip is a list[Step] with absolute-t on each Step.  A trip
    occupies ticks ``[trip[0].t, trip[-1].t]``.  For each trip in
    entry-tick order, assign to the robot with the smallest
    ``next_free`` (ties → lowest id), pad that robot with home WAITs
    from its ``next_free`` up to ``t_entry``, then append the trip's
    Steps.  Updates ``next_free[chosen]`` to ``t_exit + 1``.

    Mutates ``plans``, ``events``, ``next_free`` in place.
    """
    num_robots = len(plans)
    trip_pairs = [(trip, evs) for trip, evs in zip(trips, trip_events) if trip]
    trip_pairs.sort(key=lambda p: (p[0][0].t, p[0][0].x, p[0][0].y))

    for trip, evs in trip_pairs:
        t_entry = trip[0].t
        t_exit = trip[-1].t
        chosen = min(range(num_robots), key=lambda i: (next_free[i], i))
        hx, hy = homes[chosen]

        for tt in range(next_free[chosen], t_entry):
            plans[chosen].append(Step(ACTION_WAIT, hx, hy, 0, tt))

        plans[chosen].extend(trip)
        for tev, ev in evs.items():
            events[chosen][tev] = ev

        next_free[chosen] = t_exit + 1


def _pad_to_tick(target_t, plans, events, next_free, homes):
    """Pad each robot from next_free[i] to target_t (inclusive) with home WAITs.

    A no-op for robots already past target_t.  Mutates in place.
    """
    for i in range(len(plans)):
        hx, hy = homes[i]
        for tt in range(next_free[i], target_t + 1):
            plans[i].append(Step(ACTION_WAIT, hx, hy, 0, tt))
        if target_t + 1 > next_free[i]:
            next_free[i] = target_t + 1


def _append_cbs_plans(cbs_plans, cbs_events, plans, events, next_free,
                      homes, t_offset):
    """Splice per-robot CBS Steps into the accumulator at t_offset.

    CBS plans were produced with ``t_start=t_offset`` so each robot's
    first CBS Step has ``t == t_offset + 1``.  Pad each robot up to
    ``t_offset`` with home WAITs (the robot teleports from off-grid
    home to its CBS-on-grid start at the first CBS tick — same idiom
    the MILP path uses for entry from off-grid).

    Mutates ``plans``, ``events``, ``next_free`` in place.
    """
    num_robots = len(plans)
    _pad_to_tick(t_offset, plans, events, next_free, homes)

    for i in range(num_robots):
        cbs_steps = cbs_plans[i]
        for s in cbs_steps:
            plans[i].append(s)
        for tev, ev in cbs_events[i].items():
            events[i][tev] = ev
        if cbs_steps:
            next_free[i] = cbs_steps[-1].t + 1


def _on_grid_clamp(x, y, X, Y):
    return max(0, min(X - 1, x)), max(0, min(Y - 1, y))


def plan_group(
    substructures,
    substructure_indices,
    world_init,
    robot_starts,
    num_robots,
    X,
    Y,
    t_start=0,
    per_t_time_limit=60.0,
    total_time_limit=600.0,
    mip_gap=0.0,
    T_max=None,
    cbs_max_t=400,
    cbs_branch_limit=500,
    logger=print,
):
    """Plan a parallel group end-to-end with MILP, falling back to CBS per sub.

    Substructures in the group are solved sequentially.  Each sub is
    first attempted with the MILP.  If ``plan_structure`` returns
    ``T=None`` (infeasible at every T in the sweep within the time
    budget), the adapter falls back to ``cbs_adapter.plan_group`` for
    that single sub, splices the resulting per-robot Steps into the
    accumulator, and continues with the next sub.

    Parameters
    ----------
    substructures : list of ndarray (Z, Y, X)
    substructure_indices : list of int
    world_init : ndarray (Z, Y, X)
    robot_starts : list[(x, y)]
    num_robots, X, Y : int
    t_start : int
        Group-relative t offset (ignored by MILP, which always starts at
        t=1; carried through in metadata for logging parity with CBS).
    per_t_time_limit : float
        Gurobi TimeLimit for each T in the sweep.
    total_time_limit : float
        Soft budget for this group's MILP solves.  CBS-fallback time
        is also deducted from the remaining budget.
    mip_gap : float
        Gurobi MIPGap.
    T_max : int, optional
        Hard upper bound on the MILP horizon sweep.
    cbs_max_t, cbs_branch_limit : int
        Forwarded to ``cbs_adapter.plan_group`` on fallback.
    logger : callable
        Status logger (INFO-level — fallback events MUST be visible).

    Returns
    -------
    dict with keys matching cbs_adapter.plan_group:
      plans    : list[list[Step]]
      events   : list[dict[int, tuple]]
      assignments : list — legacy parity key (empty)
      metadata : dict (substructure_indices, T_min, T_final, solve_time,
                       num_agents_used, used_fallback, conflicts_resolved,
                       block_count, substructure_metadata, per_sub)
        substructure_metadata : list[dict]
          One row per sub with planner_used ("milp" | "cbs_fallback"),
          T contribution, solve_time, and solver-specific stats.
    """
    assert len(substructures) == len(substructure_indices)
    assert world_init.ndim == 3
    Z, gY, gX = world_init.shape
    assert (gY, gX) == (Y, X), (
        f'world_init shape {world_init.shape} != (Z, {Y}, {X})'
    )

    homes = [(rx, HOME_Y_OFFSET) for (rx, _ry) in robot_starts]
    plans = [[] for _ in range(num_robots)]
    events = [dict() for _ in range(num_robots)]
    next_free = [1] * num_robots

    hm_init = _heightmap(world_init)
    block_count = sum(int(s.sum()) for s in substructures)
    current_hm = hm_init.copy()
    current_world = world_init.astype(int).copy()
    completed_subs = []  # for the reconstruction-vs-tracked assertion

    substructure_metadata = []
    used_fallback = False
    total_solve_t0 = time.perf_counter()
    time_budget_left = float(total_time_limit)
    cumulative_T_min = 0
    t_offset = 0

    for sub, si in zip(substructures, substructure_indices):
        sub_int = sub.astype(int)
        sub_heights = np.sum(sub_int, axis=0).astype(int)
        target_hm_i = current_hm + sub_heights
        target_sub_i = _target_sub_from_substructures(
            [sub_int], [si], world_init.shape,
        )

        # The paper's MILP disallows blocks on the border (x=0/X-1, y=0/Y-1):
        # the outer ring has to stay empty so agents can enter/exit there.
        # Real-world structures can have blocks at those cells, so we pad
        # the MILP's grid with a 1-cell frame of zeros, solve there, and
        # shift decoded Steps back by (-1, -1) so the sim coords match.
        padded_init = np.pad(current_hm, 1, mode='constant', constant_values=0)
        padded_target = np.pad(target_hm_i, 1, mode='constant', constant_values=0)
        padded_tsub = np.pad(
            target_sub_i, ((0, 0), (1, 1), (1, 1)),
            mode='constant', constant_values=-1,
        )

        per_t_limit = min(per_t_time_limit, max(1.0, time_budget_left))

        t0 = time.perf_counter()
        r = plan_structure(
            padded_init, padded_target,
            num_agents=num_robots,
            time_limit=per_t_limit,
            mip_gap=mip_gap,
            output_flag=0,
            T_max=T_max,
            target_sub=padded_tsub,
            logger=lambda _m: None,
        )
        solve_t = time.perf_counter() - t0
        time_budget_left = max(0.0, time_budget_left - solve_t)

        if r['T'] is not None:
            # ---------- MILP success: fuse this sub's trips ----------
            sub_trips = []
            sub_events = []
            for trip, evs in zip(r['trips'], r['events']):
                shifted_trip = [
                    Step(s.action, s.x - 1, s.y - 1, s.z, s.t + t_offset)
                    for s in trip
                ]
                shifted_evs = {}
                for tev, ev in evs.items():
                    if ev[0] == 'place':
                        _, bx, by, bz, esi = ev
                        shifted_evs[tev + t_offset] = (
                            'place', bx - 1, by - 1, bz, esi,
                        )
                    elif ev[0] == 'pickup' and len(ev) == 5:
                        _, bx, by, bz, esi = ev
                        # Off-grid sentinel (-1,-1,-1) passes through unshifted;
                        # grid pickups shift back from padded to unpadded coords.
                        if bx >= 0:
                            shifted_evs[tev + t_offset] = (
                                'pickup', bx - 1, by - 1, bz, esi,
                            )
                        else:
                            shifted_evs[tev + t_offset] = ev
                    else:
                        shifted_evs[tev + t_offset] = ev
                sub_trips.append(shifted_trip)
                sub_events.append(shifted_evs)

            _fuse_trips_into_plans(
                sub_trips, sub_events, plans, events, next_free, homes,
            )

            sub_T = r['T']
            sub_meta = {
                'si': int(si),
                'planner_used': 'milp',
                'T': r['T'],
                'obj': r['obj_val'],
                'solve_time': solve_t,
                'num_vars': r.get('num_vars'),
                'num_constrs': r.get('num_constrs'),
                'n_trips': len(r['trips']),
                'block_count': int(sub_int.sum()),
            }
            substructure_metadata.append(sub_meta)
            logger(
                f'[MILP] sub {si}: planner=milp T={r["T"]} '
                f'trips={len(r["trips"])} '
                f'vars={r.get("num_vars")} cons={r.get("num_constrs")} '
                f'solve={solve_t:.2f}s obj={r["obj_val"]} '
                f'blocks={int(sub_int.sum())}'
            )
        else:
            # ---------- MILP infeasible: CBS fallback ----------
            used_fallback = True
            logger(
                f'[MILP] sub {si}: MILP infeasible at all T <= T_max '
                f'after {solve_t:.2f}s, falling back to CBS '
                f'(blocks={int(sub_int.sum())}, T_max={T_max}, '
                f'per_t_limit={per_t_limit:.1f}s)'
            )

            # Defensive: the world we pass to CBS must equal a fresh
            # rebuild from world_init + completed sub masks.  If this
            # assertion fires, the decomposition produced overlapping
            # subs or current_world tracking has a bug.
            reconstructed = world_init.astype(int).copy()
            for completed in completed_subs:
                reconstructed = np.maximum(reconstructed, completed)
            if not np.array_equal(reconstructed, current_world):
                raise AssertionError(
                    f'[MILP-fallback sub {si}] reconstructed world '
                    f'disagrees with tracked current_world '
                    f'(reconstructed.sum={int(reconstructed.sum())}, '
                    f'tracked.sum={int(current_world.sum())}). '
                    f'Decomposition invariant likely violated — substructures '
                    f'may overlap, or current_world tracking has regressed.'
                )

            # Robot positions at fallback time: tail of each per-robot plan.
            # Robots parked at off-grid home are clamped on-grid; CBS plans
            # the on-grid trajectory and the replay teleports the robot at
            # CBS's first Step (same idiom the MILP entry-from-off-grid uses).
            current_robot_starts = []
            for i in range(num_robots):
                if plans[i]:
                    last_step = plans[i][-1]
                    cur = (last_step.x, last_step.y)
                else:
                    cur = robot_starts[i]
                current_robot_starts.append(_on_grid_clamp(cur[0], cur[1], X, Y))

            cbs_t0 = time.perf_counter()
            try:
                cbs_result = cbs_adapter.plan_group(
                    substructures=[sub_int],
                    substructure_indices=[si],
                    world_init=current_world,
                    robot_starts=current_robot_starts,
                    num_robots=num_robots,
                    X=X, Y=Y,
                    t_start=t_offset,
                    max_t=cbs_max_t,
                    branch_limit=cbs_branch_limit,
                )
            except Exception as e:
                logger(
                    f'[MILP] sub {si}: CBS FALLBACK FAILED with exception: '
                    f'{type(e).__name__}: {e}'
                )
                raise
            cbs_solve_t = time.perf_counter() - cbs_t0
            time_budget_left = max(0.0, time_budget_left - cbs_solve_t)

            if cbs_result is None or not any(
                cbs_result['plans'][i] for i in range(num_robots)
            ):
                logger(
                    f'[MILP] sub {si}: CBS FALLBACK PRODUCED EMPTY PLAN — '
                    f'aborting group.'
                )
                raise RuntimeError(
                    f'CBS fallback for sub {si} returned no usable plan '
                    f'(blocks={int(sub_int.sum())}, '
                    f'robot_starts={current_robot_starts}, '
                    f't_offset={t_offset}). '
                    f'This should be extremely rare — investigate.'
                )

            cbs_md = cbs_result['metadata']
            cbs_T_final = cbs_md['T_final']
            sub_T_contribution = max(0, cbs_T_final - t_offset)

            _append_cbs_plans(
                cbs_result['plans'], cbs_result['events'],
                plans, events, next_free, homes, t_offset,
            )

            sub_T = sub_T_contribution
            sub_meta = {
                'si': int(si),
                'planner_used': 'cbs_fallback',
                'T': sub_T_contribution,
                'obj': None,
                'solve_time': cbs_solve_t,
                'milp_solve_time': solve_t,  # how long MILP wasted before fallback
                'num_vars': None,
                'num_constrs': None,
                'n_trips': None,
                'block_count': int(sub_int.sum()),
                'cbs_used_fallback': cbs_md.get('used_fallback', False),
                'cbs_conflicts_resolved': cbs_md.get('conflicts_resolved', 0),
            }
            substructure_metadata.append(sub_meta)
            logger(
                f'[MILP] sub {si}: planner=cbs_fallback '
                f'T_contribution={sub_T_contribution} '
                f'cbs_solve={cbs_solve_t:.2f}s '
                f'cbs_serial_fallback={sub_meta["cbs_used_fallback"]} '
                f'blocks={int(sub_int.sum())}'
            )

        cumulative_T_min += sub_T
        t_offset += sub_T
        current_hm = current_hm + sub_heights
        current_world = np.maximum(current_world, sub_int)
        completed_subs.append(sub_int.copy())

    # Final pad so every robot has a Step at every tick up to T_final.
    T_final = max(next_free) - 1 if any(nf > 1 for nf in next_free) else 0
    _pad_to_tick(T_final, plans, events, next_free, homes)

    total_solve_time = time.perf_counter() - total_solve_t0

    num_agents_used = sum(
        1 for i in range(num_robots) if any(
            s.action != ACTION_WAIT or (s.x, s.y) != homes[i]
            for s in plans[i]
        )
    )

    return {
        'plans': plans,
        'events': events,
        'assignments': [],
        'metadata': {
            'substructure_indices': list(substructure_indices),
            'T_min': cumulative_T_min,
            'T_final': T_final,
            'solve_time': total_solve_time,
            'num_agents_used': num_agents_used,
            'used_fallback': used_fallback,
            'conflicts_resolved': 0,
            'block_count': block_count,
            'substructure_metadata': substructure_metadata,
            'per_sub': substructure_metadata,  # back-compat alias
        },
    }


def plan_return(robot_starts, X, Y, world_state, t_start=0, max_t=200):
    """Return-to-boundary is planner-agnostic — delegate to cbs_adapter.

    Robots were parked off-grid (y = HOME_Y_OFFSET) by the last MILP
    plan; map them onto their nearest on-grid boundary cell so the CBS
    return planner has a legal starting state.
    """
    on_grid_starts = []
    for (rx, ry) in robot_starts:
        on_grid_starts.append(_on_grid_clamp(rx, ry, X, Y))
    return cbs_adapter.plan_return(
        robot_starts=on_grid_starts, X=X, Y=Y,
        world_state=world_state, t_start=t_start, max_t=max_t,
    )
