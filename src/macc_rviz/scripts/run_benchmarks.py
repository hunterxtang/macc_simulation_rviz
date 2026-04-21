"""Run the 6-case benchmark through the MILP+CBS hybrid and CBS-alone.

Outputs a JSON file with per-case planner_mix, T_final, solve_time,
sum-of-costs, and per-substructure breakdown for both planner modes.

Usage:
  python3 src/macc_rviz/scripts/run_benchmarks.py [--out PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_SRC = str(Path(__file__).resolve().parents[1])
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from macc_rviz.benchmark_structures import ALL_CASES  # noqa: E402
from macc_rviz.cbs_planner import ACTION_WAIT  # noqa: E402
from macc_rviz.decomposition import (  # noqa: E402
    decompose_structure, order_substructures,
)
from macc_rviz.parallel import find_parallel_groups  # noqa: E402
from macc_rviz.planners import cbs_adapter, milp_adapter  # noqa: E402


NUM_ROBOTS = 4
DEPOT_X, DEPOT_Y = 0, 0
MILP_PER_T = 60.0
MILP_TOTAL = 600.0
MILP_T_MAX = 150
CBS_MAX_T = 400
CBS_BRANCH = 500


def _logger(msg):
    print(f'    {msg}', flush=True)


def _sum_of_costs(plans, homes):
    """Per-robot SoC = last-non-trailing-home-wait tick. Group SoC = sum.

    A trailing run of home-WAIT Steps is the robot's idle pad after its
    last useful action; SoC excludes that pad to mirror the paper's
    "agent finishes when its work is done" semantics.
    """
    total = 0
    for i, plan in enumerate(plans):
        hx, hy = homes[i]
        last_active = 0
        for s in plan:
            is_home_wait = (s.action == ACTION_WAIT
                            and s.x == hx and s.y == hy)
            if not is_home_wait:
                last_active = s.t
        total += last_active
    return total


def _active_steps(plans, homes):
    """Count Steps that are not parked-at-home WAITs (cheap proxy for
    "robot is doing something this tick")."""
    n = 0
    for i, plan in enumerate(plans):
        hx, hy = homes[i]
        for s in plan:
            if not (s.action == ACTION_WAIT and s.x == hx and s.y == hy):
                n += 1
    return n


def _run_hybrid(structure, name):
    Z, Y, X = structure.shape
    subs = decompose_structure(structure)
    order, deps = order_substructures(subs)
    groups = find_parallel_groups(subs, deps)

    world_init = np.zeros_like(structure, dtype=int)
    robot_starts = [(DEPOT_X, DEPOT_Y) for _ in range(NUM_ROBOTS)]
    homes = [(rx, milp_adapter.HOME_Y_OFFSET) for (rx, _) in robot_starts]

    print(f'  [hybrid] {name}: {len(subs)} subs, {len(groups)} groups')
    t0 = time.perf_counter()
    all_plans = [[] for _ in range(NUM_ROBOTS)]
    all_events = [dict() for _ in range(NUM_ROBOTS)]
    sub_meta_concat = []
    used_fallback = False
    cumulative_T = 0
    current_world = world_init.copy()

    for gi, group in enumerate(groups):
        gsubs = [subs[i] for i in group]
        gindices = [order.index(i) if i in order else i for i in group]
        gindices = list(group)  # use raw sub index as identifier

        result = milp_adapter.plan_group(
            substructures=gsubs,
            substructure_indices=gindices,
            world_init=current_world,
            robot_starts=robot_starts,
            num_robots=NUM_ROBOTS,
            X=X, Y=Y,
            t_start=cumulative_T,
            per_t_time_limit=MILP_PER_T,
            total_time_limit=MILP_TOTAL,
            T_max=MILP_T_MAX,
            cbs_max_t=CBS_MAX_T,
            cbs_branch_limit=CBS_BRANCH,
            logger=_logger,
        )
        md = result['metadata']
        sub_meta_concat.extend(md['substructure_metadata'])
        used_fallback = used_fallback or md['used_fallback']
        for i in range(NUM_ROBOTS):
            all_plans[i].extend(result['plans'][i])
            for tev, ev in result['events'][i].items():
                all_events[i][tev] = ev
        cumulative_T = max(cumulative_T, md['T_final'])

        for s_idx in group:
            current_world = np.maximum(current_world, subs[s_idx])

    elapsed = time.perf_counter() - t0

    n_milp = sum(1 for m in sub_meta_concat if m['planner_used'] == 'milp')
    n_cbs = sum(1 for m in sub_meta_concat
                if m['planner_used'] == 'cbs_fallback')

    return {
        'mode': 'hybrid',
        'name': name,
        'num_subs': len(subs),
        'num_groups': len(groups),
        'sub_block_sizes': [int(s.sum()) for s in subs],
        'T_final': cumulative_T,
        'wall_solve_time': elapsed,
        'sum_of_costs': _sum_of_costs(all_plans, homes),
        'active_steps': _active_steps(all_plans, homes),
        'planner_mix': {'milp': n_milp, 'cbs_fallback': n_cbs},
        'used_fallback': used_fallback,
        'substructures': [
            {k: v for k, v in m.items()
             if k in ('si', 'planner_used', 'T', 'solve_time',
                      'milp_solve_time', 'block_count', 'n_trips', 'obj')}
            for m in sub_meta_concat
        ],
    }


def _run_cbs_alone(structure, name):
    Z, Y, X = structure.shape
    subs = decompose_structure(structure)
    order, deps = order_substructures(subs)
    groups = find_parallel_groups(subs, deps)

    world_init = np.zeros_like(structure, dtype=int)
    robot_starts = [(DEPOT_X, DEPOT_Y) for _ in range(NUM_ROBOTS)]
    # CBS-alone: home is on-grid depot (no off-grid parking), so "home WAIT"
    # is the depot.
    homes = [(DEPOT_X, DEPOT_Y) for _ in range(NUM_ROBOTS)]

    print(f'  [cbs ]   {name}: {len(subs)} subs, {len(groups)} groups')
    t0 = time.perf_counter()
    all_plans = [[] for _ in range(NUM_ROBOTS)]
    all_events = [dict() for _ in range(NUM_ROBOTS)]
    cumulative_T = 0
    current_world = world_init.copy()
    sub_meta = []
    starts = list(robot_starts)

    for gi, group in enumerate(groups):
        gsubs = [subs[i] for i in group]
        gindices = list(group)

        result = cbs_adapter.plan_group(
            substructures=gsubs,
            substructure_indices=gindices,
            world_init=current_world,
            robot_starts=starts,
            num_robots=NUM_ROBOTS,
            X=X, Y=Y,
            t_start=cumulative_T,
            max_t=CBS_MAX_T,
            branch_limit=CBS_BRANCH,
        )
        md = result['metadata']
        for i in range(NUM_ROBOTS):
            all_plans[i].extend(result['plans'][i])
            for tev, ev in result['events'][i].items():
                all_events[i][tev] = ev
        # Update starts to where each robot ended this group
        starts = [
            (result['plans'][i][-1].x, result['plans'][i][-1].y)
            if result['plans'][i] else starts[i]
            for i in range(NUM_ROBOTS)
        ]
        cumulative_T = max(cumulative_T, md['T_final'])
        sub_meta.append({
            'group': gi,
            'substructure_indices': md['substructure_indices'],
            'block_count': md['block_count'],
            'T_final': md['T_final'],
            'solve_time': md['solve_time'],
            'used_fallback': md['used_fallback'],
            'conflicts_resolved': md['conflicts_resolved'],
        })

        for s_idx in group:
            current_world = np.maximum(current_world, subs[s_idx])

    elapsed = time.perf_counter() - t0

    return {
        'mode': 'cbs_alone',
        'name': name,
        'num_subs': len(subs),
        'num_groups': len(groups),
        'sub_block_sizes': [int(s.sum()) for s in subs],
        'T_final': cumulative_T,
        'wall_solve_time': elapsed,
        'sum_of_costs': _sum_of_costs(all_plans, homes),
        'active_steps': _active_steps(all_plans, homes),
        'group_breakdown': sub_meta,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--out', default='src/macc_rviz/docs/benchmark_results.json',
        help='Output JSON path (relative to ros2_ws root)',
    )
    p.add_argument(
        '--cases', default='', help='Comma-separated case names (default: all)',
    )
    args = p.parse_args()

    cases = ALL_CASES
    if args.cases:
        wanted = set(args.cases.split(','))
        cases = [(n, fn) for (n, fn) in ALL_CASES if n in wanted]

    results = []
    for name, fn in cases:
        s = fn()
        print(f'==== {name}  shape={s.shape}  blocks={int(s.sum())} ====')
        try:
            r_h = _run_hybrid(s, name)
        except Exception as e:
            r_h = {'mode': 'hybrid', 'name': name, 'error': repr(e)}
            print(f'    HYBRID CRASHED: {e!r}')
        try:
            r_c = _run_cbs_alone(s, name)
        except Exception as e:
            r_c = {'mode': 'cbs_alone', 'name': name, 'error': repr(e)}
            print(f'    CBS-ALONE CRASHED: {e!r}')
        results.append({'case': name, 'hybrid': r_h, 'cbs_alone': r_c})

        # Brief summary line per case
        h_t = r_h.get('T_final', '-')
        h_s = r_h.get('wall_solve_time', float('nan'))
        h_mix = r_h.get('planner_mix', {})
        c_t = r_c.get('T_final', '-')
        c_s = r_c.get('wall_solve_time', float('nan'))
        print(
            f'    => hybrid T={h_t} solve={h_s:.2f}s mix={h_mix}   '
            f'cbs T={c_t} solve={c_s:.2f}s'
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        'budgets': {
            'num_robots': NUM_ROBOTS,
            'milp_per_t_time_limit': MILP_PER_T,
            'milp_total_time_limit': MILP_TOTAL,
            'milp_T_max': MILP_T_MAX,
            'cbs_max_t': CBS_MAX_T,
            'cbs_branch_limit': CBS_BRANCH,
        },
        'results': results,
    }, indent=2))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
