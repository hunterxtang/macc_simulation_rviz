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


def _assign_trips_to_robots(trips, trip_events, num_robots, robot_starts):
    """Greedy earliest-free-robot scheduling of MILP trips.

    Each trip is a list[Step] with absolute-t on each Step.  A trip
    occupies ticks [trip[0].t, trip[-1].t].  Robots are padded with
    WAITs at an off-grid "home" cell between trips so the per-robot
    plan has exactly one Step per tick up to the overall plan makespan.

    Parameters
    ----------
    trips : list[list[Step]]
        Output of ``plan_structure``.
    trip_events : list[dict[int, tuple]]
        Per-trip event dicts (same length as ``trips``).
    num_robots : int
    robot_starts : list[(x, y)]
        Starting grid cell of each physical robot; used to derive each
        robot's off-grid home (same x, y = HOME_Y_OFFSET).

    Returns
    -------
    plans  : list[list[Step]]        — one per robot; plan[i][k].t == k+1
    events : list[dict[int, tuple]]  — one per robot; keyed by absolute t
    T_final : int                    — plan makespan in ticks (≥ 1)
    num_agents_used : int            — robots with at least one on-grid Step
    """
    homes = [(rx, HOME_Y_OFFSET) for (rx, _ry) in robot_starts]

    plans = [[] for _ in range(num_robots)]
    events = [dict() for _ in range(num_robots)]
    # next_free[i] is the next tick at which robot i needs a Step.
    # All robots start at tick 1 needing their first Step.
    next_free = [1] * num_robots

    trip_pairs = [(trip, evs) for trip, evs in zip(trips, trip_events) if trip]
    trip_pairs.sort(key=lambda p: (p[0][0].t, p[0][0].x, p[0][0].y))

    for trip, evs in trip_pairs:
        t_entry = trip[0].t
        t_exit = trip[-1].t
        # Pick the robot that'll be free earliest (ties → lowest id).
        chosen = min(range(num_robots), key=lambda i: (next_free[i], i))
        hx, hy = homes[chosen]

        # Pad with home WAITs from next_free up to (but not including) t_entry.
        for tt in range(next_free[chosen], t_entry):
            plans[chosen].append(Step(ACTION_WAIT, hx, hy, 0, tt))

        # Append trip Steps (t already absolute).
        plans[chosen].extend(trip)
        for tev, ev in evs.items():
            events[chosen][tev] = ev

        next_free[chosen] = t_exit + 1

    T_final = max((p[-1].t for p in plans if p), default=0)

    # Pad every robot to T_final with home WAITs so every tick has a Step.
    for i in range(num_robots):
        hx, hy = homes[i]
        for tt in range(next_free[i], T_final + 1):
            plans[i].append(Step(ACTION_WAIT, hx, hy, 0, tt))

    num_agents_used = sum(
        1 for i in range(num_robots) if any(
            s.action != ACTION_WAIT or (s.x, s.y) != homes[i]
            for s in plans[i]
        )
    )

    return plans, events, T_final, num_agents_used


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
    logger=print,
):
    """Plan a parallel group end-to-end with the MILP.

    Substructures in the group are solved sequentially, each subsequent
    solve seeing the previous ones' trips as ``prior_trajectories`` plus
    an ``agent_cap_from_priors``-derived per-t cap.  Results are fused
    into a single per-robot plan via greedy scheduling.

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
        Soft budget for this group's solve.
    mip_gap : float
        Gurobi MIPGap.
    T_max : int, optional
        Hard upper bound on the horizon sweep.
    logger : callable
        Status logger.

    Returns
    -------
    dict with keys matching cbs_adapter.plan_group:
      plans    : list[list[Step]]
      events   : list[dict[int, tuple]]
      assignments : list — legacy parity key (empty, MILP does its own assignment)
      metadata : dict (substructure_indices, T_min, T_final, solve_time,
                       num_agents_used, used_fallback, conflicts_resolved,
                       block_count)
    """
    assert len(substructures) == len(substructure_indices)
    assert world_init.ndim == 3
    Z, gY, gX = world_init.shape
    assert (gY, gX) == (Y, X)

    hm_init = _heightmap(world_init)
    block_count = sum(int(s.sum()) for s in substructures)
    current_hm = hm_init.copy()

    # Concatenate per-substructure trips in order with a t-offset so the
    # group's replay runs sub 0 → sub 1 → … end-to-end.
    all_trips = []
    all_events = []
    per_sub_rows = []
    used_fallback = False
    total_solve_t0 = time.perf_counter()
    time_budget_left = float(total_time_limit)
    cumulative_T_min = 0
    t_offset = 0  # absolute-t offset for the current substructure

    for sub, si in zip(substructures, substructure_indices):
        sub_heights = np.sum(sub, axis=0).astype(int)
        target_hm_i = current_hm + sub_heights
        target_sub_i = _target_sub_from_substructures(
            [sub], [si], world_init.shape,
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

        row = {
            'si': int(si),
            'T': r['T'],
            'obj': r['obj_val'],
            'solve_time': solve_t,
            'num_vars': r.get('num_vars'),
            'num_constrs': r.get('num_constrs'),
            'timeout': r['T'] is None,
            'n_trips': len(r['trips']),
        }
        per_sub_rows.append(row)
        logger(
            f'[MILP] sub {si}: T={row["T"]} trips={row["n_trips"]} '
            f'vars={row["num_vars"]} cons={row["num_constrs"]} '
            f'solve={solve_t:.2f}s obj={row["obj"]} '
            f'blocks={int(sub.sum())}'
        )

        if r['T'] is None:
            used_fallback = True
            logger(
                f'[MILP] sub {si} returned no plan '
                f'(time_limit={per_t_limit:.1f}s); aborting group.'
            )
            break

        # Offset this sub's trips by t_offset so they stack after prior
        # subs, and shift (x, y) back by the 1-cell pad to get sim coords.
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
                else:
                    shifted_evs[tev + t_offset] = ev
            all_trips.append(shifted_trip)
            all_events.append(shifted_evs)

        # Next sub starts after this one fully exits (last tick + 1).
        cumulative_T_min += r['T']
        t_offset += r['T']

        current_hm = current_hm + sub_heights

    total_solve_time = time.perf_counter() - total_solve_t0

    plans, events, makespan, num_agents_used = _assign_trips_to_robots(
        all_trips, all_events, num_robots, robot_starts,
    )

    return {
        'plans': plans,
        'events': events,
        'assignments': [],
        'metadata': {
            'substructure_indices': list(substructure_indices),
            'T_min': cumulative_T_min,
            'T_final': makespan,
            'solve_time': total_solve_time,
            'num_agents_used': num_agents_used,
            'used_fallback': used_fallback,
            'conflicts_resolved': 0,
            'block_count': block_count,
            'per_sub': per_sub_rows,
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
        sx = max(0, min(X - 1, rx))
        sy = max(0, min(Y - 1, ry))
        on_grid_starts.append((sx, sy))
    return cbs_adapter.plan_return(
        robot_starts=on_grid_starts, X=X, Y=Y,
        world_state=world_state, t_start=t_start, max_t=max_t,
    )
