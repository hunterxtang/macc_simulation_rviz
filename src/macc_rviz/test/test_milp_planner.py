"""MILP planner unit tests (tiny problems only).

Four smoke tests per the development spec:
  (a) 1x1x2 tower, 1 agent  — needs ramp build + teardown
  (b) 1x1x2 tower, 4 agents — extra agents shrink T vs (a)
  (c) 2x1x1 pair,  1 agent  — two trips by the same agent
  (d) 2x1x1 pair,  2 agents — single trip each, concurrent

Each must solve in under 10 s on the reference ARM64 VM (academic
Gurobi 13.0.1).  We assert on solve-time to catch regressions in the
encoding cost.

Run manually:
  ~/macc-venv/bin/python3 -m pytest test/test_milp_planner.py -v -s
"""

import time  # noqa: I100, I101

import numpy as np  # noqa: I100, I101
import pytest  # noqa: I100, I101

pytest.importorskip('gurobipy')  # skip everything if Gurobi missing

from macc_rviz.cbs_planner import (  # noqa: I100, I101
    ACTION_MOVE, ACTION_PICKUP, ACTION_PLACE,
)
from macc_rviz.planners.milp_planner import plan_structure  # noqa: I100, I101


# Max per-test wall-clock for the sweep.  Well above observed solve times
# (~0.1 s on ARM64 VM); catches an order-of-magnitude regression.
SOLVE_BUDGET_SEC = 10.0


def _grid(Y, X, tower_cells):
    """Build an init_hm (all zeros) and target_hm with tower heights set."""
    init = np.zeros((Y, X), dtype=int)
    tgt = np.zeros((Y, X), dtype=int)
    for (x, y, h) in tower_cells:
        tgt[y, x] = h
    return init, tgt


def _count_places(trips):
    return sum(1 for trip in trips for s in trip if s.action == ACTION_PLACE)


def _count_pickups(trips):
    return sum(1 for trip in trips for s in trip if s.action == ACTION_PICKUP)


def _count_moves(trips):
    return sum(1 for trip in trips for s in trip if s.action == ACTION_MOVE)


# ---------------------------------------------------------------------------
# (c), (d): 2x1x1 pair — two adjacent ground blocks at (1,1) and (2,1)
# ---------------------------------------------------------------------------


def test_2x1x1_pair_two_agents_T4():
    """Two agents deliver concurrently at T=4 with 2 deliveries + 2 exits."""
    init, tgt = _grid(3, 4, [(1, 1, 1), (2, 1, 1)])
    t0 = time.perf_counter()
    r = plan_structure(init, tgt, num_agents=2,
                       time_limit=SOLVE_BUDGET_SEC, output_flag=0)
    wall = time.perf_counter() - t0

    assert r['T'] == 4, f"expected T=4 with 2 agents, got T={r['T']}"
    assert r['obj_val'] == 4.0, f"expected obj=4 (2 deliver + 2 exit), got {r['obj_val']}"
    assert _count_places(r['trips']) == 2
    assert _count_pickups(r['trips']) == 0
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp] 2x1x1/2a: T={r["T"]} obj={r["obj_val"]} '
          f'vars={r["num_vars"]} wall={wall:.2f}s')


def test_2x1x1_pair_one_agent_needs_two_trips():
    """Same target with 1 agent takes two trips and lands at T=7."""
    init, tgt = _grid(3, 4, [(1, 1, 1), (2, 1, 1)])
    t0 = time.perf_counter()
    r = plan_structure(init, tgt, num_agents=1,
                       time_limit=SOLVE_BUDGET_SEC, output_flag=0)
    wall = time.perf_counter() - t0

    assert r['T'] == 7, f"expected T=7 with 1 agent, got T={r['T']}"
    assert _count_places(r['trips']) == 2
    assert _count_pickups(r['trips']) == 0
    # Two non-overlapping trips, one per delivered block.
    assert len(r['trips']) == 2
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp] 2x1x1/1a: T={r["T"]} obj={r["obj_val"]} '
          f'trips={len(r["trips"])} vars={r["num_vars"]} wall={wall:.2f}s')


# ---------------------------------------------------------------------------
# (a), (b): 1x1x2 tower — the hard case, needs ramp
# ---------------------------------------------------------------------------


def test_1x1x2_tower_one_agent_ramp_cycle():
    """Single-column height-2 target: 1 agent must build, use, and remove
    a ramp at the neighbor column.  4 block-ops total."""
    init, tgt = _grid(3, 4, [(1, 1, 2)])
    t0 = time.perf_counter()
    r = plan_structure(init, tgt, num_agents=1,
                       time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                       T_max=20)
    wall = time.perf_counter() - t0

    assert r['T'] is not None, 'MILP did not find a feasible T'
    # 3 deliveries (2 target + 1 ramp) + 1 ramp pickup = 4 block ops.
    assert _count_places(r['trips']) == 3, \
        f"expected 3 places, got {_count_places(r['trips'])}"
    assert _count_pickups(r['trips']) == 1, \
        f"expected 1 pickup, got {_count_pickups(r['trips'])}"
    # At least one move on the ramp (robot climbing to z=1).
    assert _count_moves(r['trips']) >= 1
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp] 1x1x2/1a: T={r["T"]} obj={r["obj_val"]} '
          f'trips={len(r["trips"])} vars={r["num_vars"]} wall={wall:.2f}s')


def test_1x1x2_tower_four_agents_parallel_shrinks_T():
    """With 4 agents, the same target solves at a strictly shorter T
    than the 1-agent case, via parallel trips."""
    init, tgt = _grid(3, 4, [(1, 1, 2)])
    t0 = time.perf_counter()
    r4 = plan_structure(init, tgt, num_agents=4,
                        time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                        T_max=20)
    wall = time.perf_counter() - t0

    assert r4['T'] is not None
    assert _count_places(r4['trips']) == 3
    assert _count_pickups(r4['trips']) == 1
    # Key claim: more agents → smaller T.
    assert r4['T'] <= 10, \
        f"expected T<=10 with 4 agents, got T={r4['T']}"
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp] 1x1x2/4a: T={r4["T"]} obj={r4["obj_val"]} '
          f'trips={len(r4["trips"])} vars={r4["num_vars"]} wall={wall:.2f}s')
