"""Top-level MILP planner entry for MACC (Lam et al. CP 2020).

``plan_structure`` builds and solves the Fig. 2 MILP for a given
(init_hm, target_hm, num_agents) instance, sweeping the horizon T
upward from a cheap lower bound until the first feasible T is found.
For that T, the solver is asked to minimize the Fig. 2 (1) objective.

All Gurobi parameters (Threads, MIPGap, TimeLimit, Presolve) are
exposed as kwargs so callers can tune solves without editing this file.

The ``Model.write("debug.lp")`` / ``Model.write("debug.sol")`` hooks
are triggered on the first model whose var count exceeds a threshold,
so a developer can inspect the encoding byte-for-byte against Fig. 2.
"""

import os
import time

import numpy as np

from gurobipy import GRB

from macc_rviz.planners.milp_decode import extract_agent_trips_with_events
from macc_rviz.planners.milp_encoding import build_model


def lower_bound_T(init_hm, target_hm, num_agents):
    """Cheap LB on the horizon T.

    The tightest cheap bound is derived from the single-agent trip
    cadence: one trip (enter → deliver → exit) needs T=4; each extra
    trip by the same agent adds 3 timesteps (exit at t, re-enter at
    t+1, deliver at t+2, exit at t+3).  With A agents in parallel the
    number of per-agent trips is ceil(|Δ| / A).

    Any navigation / ramp overhead shows up as an additional gap
    between this LB and the first feasible T during the sweep.

    T must always be ≥ 4 so the R_1 / R_4 windows are non-empty.
    """
    init = np.asarray(init_hm, dtype=int)
    tgt = np.asarray(target_hm, dtype=int)
    delta = int(np.abs(tgt - init).sum())
    if delta == 0:
        return 4
    trips_per_agent = -(-delta // max(num_agents, 1))  # ceil div
    return max(4, 3 * trips_per_agent + 1)


def plan_structure(
    init_hm,
    target_hm,
    num_agents,
    *,
    T_min=None,
    T_max=None,
    time_limit=60.0,
    mip_gap=0.0,
    threads=0,
    presolve=-1,
    output_flag=0,
    debug_dir=None,
    logger=print,
    prior_trajectories=None,
    max_agents_at_t=None,
    target_sub=None,
):
    """Plan construction of ``target_hm`` from ``init_hm`` with A agents.

    Sweeps T from ``T_min`` (default: lower_bound_T) upward.  At each T
    the model is solved with the given TimeLimit; on OPTIMAL or feasible
    (SUBOPTIMAL / TIME_LIMIT with an incumbent) we return the per-agent
    trips.

    Parameters
    ----------
    init_hm, target_hm : ndarray (Y, X), int
    num_agents : int
    T_min, T_max : int, optional
        Bounds on the sweep. Default T_min = lower_bound_T; default
        T_max = T_min + 20.
    time_limit : float
        Per-T Gurobi time limit, seconds.
    mip_gap, threads, presolve, output_flag : see milp_encoding.build_model.
    debug_dir : str, optional
        When set, the first T that produces a model with > 500 vars writes
        ``macc_T<N>.lp`` and (on feasible) ``macc_T<N>.sol`` there.
    logger : callable(str), default print
        Per-T status logger.
    prior_trajectories : list[list[Step]], optional
        Previously-committed agent trips from parallel-group siblings
        (§IV-C).  Passed straight through to ``build_model``; at each T
        they inject vertex-/edge-collision constraints (PAR-V, PAR-E).
    max_agents_at_t : dict[t, int], optional
        Per-t tightening of constraint (11).  Combine with
        ``agent_cap_from_priors`` to enforce the global fleet cap across
        parallel substructures.
    target_sub : ndarray (Z, Y, X), optional
        Voxel-level substructure tags used to colour pickup/place events
        emitted alongside the trips.  None → events tag si=-1.

    Returns
    -------
    dict with keys:
        trips         : list[list[Step]]  — one per agent trip
        T             : int               — winning horizon
        model_status  : int               — Gurobi status code
        obj_val       : float             — objective at winning T
        solve_time    : float             — total seconds across sweep
        per_T_log     : list[dict]        — one row per T attempted
    """
    init_hm = np.asarray(init_hm, dtype=int)
    target_hm = np.asarray(target_hm, dtype=int)

    if T_min is None:
        T_min = lower_bound_T(init_hm, target_hm, num_agents)
    if T_max is None:
        T_max = T_min + 20

    per_T_log = []
    t_start = time.perf_counter()
    debug_written_at = None

    for T in range(T_min, T_max + 1):
        built = build_model(
            init_hm, target_hm, num_agents, T,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            presolve=presolve,
            output_flag=output_flag,
            prior_trajectories=prior_trajectories,
            max_agents_at_t=max_agents_at_t,
        )
        m = built['model']
        num_vars = m.NumVars
        num_cons = m.NumConstrs

        if (debug_dir is not None
                and debug_written_at is None
                and num_vars > 500):
            os.makedirs(debug_dir, exist_ok=True)
            lp_path = os.path.join(debug_dir, f'macc_T{T}.lp')
            m.write(lp_path)
            debug_written_at = (T, lp_path)

        solve_t0 = time.perf_counter()
        m.optimize()
        solve_dt = time.perf_counter() - solve_t0

        status = m.Status
        feasible = m.SolCount > 0
        obj = m.ObjVal if feasible else None

        row = {
            'T': T,
            'feasible': feasible,
            'status': status,
            'num_vars': num_vars,
            'num_constrs': num_cons,
            'solve_time': solve_dt,
            'obj_val': obj,
        }
        per_T_log.append(row)
        logger(
            f'[milp] T={T:<3d} vars={num_vars:>6d} cons={num_cons:>6d} '
            f'status={status} feasible={feasible} '
            f't={solve_dt:.2f}s obj={obj}'
        )

        if feasible:
            if (debug_dir is not None
                    and debug_written_at is not None
                    and debug_written_at[0] == T):
                sol_path = os.path.join(debug_dir, f'macc_T{T}.sol')
                m.write(sol_path)
            trips, events = extract_agent_trips_with_events(
                built['R_vars'], T, target_sub=target_sub,
            )
            return {
                'trips': trips,
                'events': events,
                'T': T,
                'model_status': status,
                'obj_val': obj,
                'solve_time': time.perf_counter() - t_start,
                'per_T_log': per_T_log,
                'num_vars': num_vars,
                'num_constrs': num_cons,
            }

        # Avoid leaking model references between iterations.
        m.dispose()

    return {
        'trips': [],
        'events': [],
        'T': None,
        'model_status': None,
        'obj_val': None,
        'solve_time': time.perf_counter() - t_start,
        'per_T_log': per_T_log,
        'num_vars': None,
        'num_constrs': None,
    }
