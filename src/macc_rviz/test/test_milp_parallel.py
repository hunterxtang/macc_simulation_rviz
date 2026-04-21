"""MILP parallel-construction constraint tests (Phase 4).

Exercises the ``prior_trajectories`` / ``max_agents_at_t`` kwargs added
to ``milp_encoding.build_model`` for §IV-C of the decomposition paper —
vertex/edge collision and per-t agent cap injection.

Four smoke tests:
  (a) two spatially independent singletons   — no delay, cap slack
  (b) vertex blocking forces sequentialization — priors close off all
      entry routes for t=1..5, S_2 must delay entry to t=5
  (c) global agent cap binds                 — S_2's natural parallel
      plan (T=4) is forced to T=7 by a per-t cap of 1
  (d) worst case cap=1                        — S_2 must start strictly
      after S_1 ends

Each must solve in under 10 s on the reference ARM64 VM.

Run manually:
  ~/macc-venv/bin/python3 -m pytest test/test_milp_parallel.py -v -s
"""

import time  # noqa: I100, I101
from collections import Counter  # noqa: I100, I101

import numpy as np  # noqa: I100, I101
import pytest  # noqa: I100, I101

pytest.importorskip('gurobipy')  # skip everything if Gurobi missing

from macc_rviz.cbs_planner import ACTION_WAIT, Step  # noqa: I100, I101
from macc_rviz.planners.milp_encoding import (  # noqa: I100, I101
    agent_cap_from_priors,
)
from macc_rviz.planners.milp_planner import plan_structure  # noqa: I100, I101


SOLVE_BUDGET_SEC = 10.0


def _grid(Y, X, tower_cells):
    init = np.zeros((Y, X), dtype=int)
    tgt = np.zeros((Y, X), dtype=int)
    for (x, y, h) in tower_cells:
        tgt[y, x] = h
    return init, tgt


def _max_concurrent(trips):
    """Max number of agent trips active simultaneously across time."""
    counter = Counter()
    for trip in trips:
        if not trip:
            continue
        t_e = trip[0].t - 1
        t_last = trip[-1].t
        for t in range(t_e, t_last + 1):
            counter[t] += 1
    return max(counter.values()) if counter else 0


def _silent(*_a, **_k):
    pass


# ---------------------------------------------------------------------------
# (a) Two spatially independent singletons — no delay
# ---------------------------------------------------------------------------


def test_parallel_independent_substructures_no_delay():
    """S_1 and S_2 far apart on a 7x3 grid; S_2 solves at the same T
    as alone, regardless of prior_trajectories from S_1."""
    init, t1 = _grid(3, 7, [(1, 1, 1)])
    _, t2 = _grid(3, 7, [(5, 1, 1)])

    t0 = time.perf_counter()
    r1 = plan_structure(init, t1, num_agents=1,
                        time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                        logger=_silent)
    r2 = plan_structure(init, t2, num_agents=1,
                        time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                        logger=_silent,
                        prior_trajectories=r1['trips'])
    wall = time.perf_counter() - t0

    assert r1['T'] == 4, f'baseline S_1 T should be 4, got {r1["T"]}'
    assert r2['T'] == 4, (
        f'S_2 expected T=4 with independent priors, got T={r2["T"]}'
    )
    # Both start at action-time t=0 → first visible Step at t=1.
    assert r1['trips'][0][0].t == 1
    assert r2['trips'][0][0].t == 1
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp-par] independent: S_1 T={r1["T"]} S_2 T={r2["T"]} '
          f'wall={wall:.2f}s')


# ---------------------------------------------------------------------------
# (b) Vertex blocking forces sequentialization
# ---------------------------------------------------------------------------


def test_prior_vertex_blocking_forces_delayed_entry():
    """Hand-crafted ghost trips occupy all 4 borders adjacent to (1,1)
    for t=1..5, so the only place to deliver (1,1,1) is blocked.  The
    MILP must delay entry to t=5, giving T=9 (vs. T=4 alone)."""
    init, tgt = _grid(3, 3, [(1, 1, 1)])

    block_times = list(range(1, 6))
    ghost_trips = [
        [Step(ACTION_WAIT, x, y, 0, t) for t in block_times]
        for (x, y) in [(0, 1), (1, 0), (1, 2), (2, 1)]
    ]

    t0 = time.perf_counter()
    r = plan_structure(init, tgt, num_agents=1,
                       time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                       logger=_silent,
                       prior_trajectories=ghost_trips, T_max=20)
    wall = time.perf_counter() - t0

    assert r['T'] is not None, 'MILP did not find a feasible T'
    assert r['T'] == 9, f'expected T=9 with blocked entries, got T={r["T"]}'

    # First visible Step must land at or after t=6 (entry action ≥ 5).
    first_step = r['trips'][0][0]
    assert first_step.t >= 6, (
        f'S_2 landed before blocked-window end: first_step t={first_step.t}'
    )
    # Landing cell must be one of the 4 blocked cells (but only after t=5).
    assert (first_step.x, first_step.y) in {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp-par] vertex-block: T={r["T"]} first_step t={first_step.t} '
          f'cell=({first_step.x},{first_step.y}) wall={wall:.2f}s')


# ---------------------------------------------------------------------------
# (c) Global agent cap binds → S_2's parallel plan forced to serialise
# ---------------------------------------------------------------------------


def test_global_agent_cap_forces_longer_T_for_s2():
    """Global cap = 2.  S_1 uses 1 agent on a 1-block target, leaving
    cap-1 = 1 for S_2 at S_1's active times (t=0..2).  S_2's natural
    2-agent parallel plan (T=4) is forced up to T=7 because it cannot
    run both agents concurrently during S_1's window."""
    init, t1 = _grid(3, 5, [(1, 1, 1)])
    _, t2 = _grid(3, 5, [(2, 1, 1), (3, 1, 1)])

    r1 = plan_structure(init, t1, num_agents=1,
                        time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                        logger=_silent)
    assert r1['T'] == 4

    cap = agent_cap_from_priors(r1['trips'], global_cap=2)
    # S_1 is active for t=0..2 at 1 agent each; cap should be 1 there.
    assert cap == {0: 1, 1: 1, 2: 1}, f'unexpected per-t cap: {cap}'

    # Baseline without cap: 2 agents, parallel, T=4.
    t0 = time.perf_counter()
    r2_free = plan_structure(init, t2, num_agents=2,
                             time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                             logger=_silent)
    # With cap: same target, 2 agents requested, but cap=1 at t=0..2 forces
    # serial execution or post-S_1 parallel; either way T must grow.
    r2_cap = plan_structure(init, t2, num_agents=2,
                            time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                            logger=_silent,
                            prior_trajectories=r1['trips'],
                            max_agents_at_t=cap, T_max=20)
    wall = time.perf_counter() - t0

    assert r2_free['T'] == 4, (
        f'free baseline should be T=4, got {r2_free["T"]}'
    )
    assert r2_cap['T'] == 7, (
        f'capped S_2 should be T=7, got {r2_cap["T"]}'
    )
    assert r2_cap['T'] > r2_free['T'], 'cap did not bind'
    # Per-t cap respected everywhere (≤ cap[t] if set, else ≤ num_agents).
    s2_active = Counter()
    for trip in r2_cap['trips']:
        t_e = trip[0].t - 1
        t_last = trip[-1].t
        for t in range(t_e, t_last + 1):
            s2_active[t] += 1
    for t, cnt in s2_active.items():
        allowed = cap.get(t, 2)
        assert cnt <= allowed, (
            f'per-t cap violated at t={t}: S_2 active={cnt} > cap={allowed}'
        )
    # At the peak, S_2 parallelised — otherwise the cap would not be binding.
    assert _max_concurrent(r2_cap['trips']) == 2, (
        f'expected peak concurrency 2 post-S_1, got '
        f'{_max_concurrent(r2_cap["trips"])}'
    )
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp-par] cap-binding: free T={r2_free["T"]} capped '
          f'T={r2_cap["T"]} wall={wall:.2f}s')


# ---------------------------------------------------------------------------
# (d) Worst case: global cap = 1 forces S_2 to start strictly after S_1
# ---------------------------------------------------------------------------


def test_global_cap_one_forces_strict_sequentialization():
    """Global cap = 1.  S_1 owns the only agent for t=0..2.  S_2's
    first action-time must be > S_1's last action-time (t=2), i.e., ≥ 3.
    With a 1-block target, that lands us at T=7."""
    init, t1 = _grid(3, 5, [(1, 1, 1)])
    _, t2 = _grid(3, 5, [(3, 1, 1)])

    r1 = plan_structure(init, t1, num_agents=1,
                        time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                        logger=_silent)
    assert r1['T'] == 4

    cap = agent_cap_from_priors(r1['trips'], global_cap=1)
    assert cap == {0: 0, 1: 0, 2: 0}, f'unexpected cap: {cap}'

    t0 = time.perf_counter()
    r2 = plan_structure(init, t2, num_agents=1,
                        time_limit=SOLVE_BUDGET_SEC, output_flag=0,
                        logger=_silent,
                        prior_trajectories=r1['trips'],
                        max_agents_at_t=cap, T_max=20)
    wall = time.perf_counter() - t0

    assert r2['T'] == 7, f'cap=1 S_2 should be T=7, got {r2["T"]}'

    # S_1's last action time = t=2 (R_4 exit).  S_2's first action time
    # (entry) must be strictly greater.
    s1_last_action = r1['trips'][-1][-1].t
    s2_entry_action = r2['trips'][0][0].t - 1
    assert s2_entry_action > s1_last_action, (
        f'S_2 did not sequentialize: entry@{s2_entry_action} '
        f'vs S_1 exit@{s1_last_action}'
    )
    # Concurrent agents across S_1 ∪ S_2 ≤ 1 everywhere.
    combined = r1['trips'] + r2['trips']
    assert _max_concurrent(combined) == 1, (
        f'concurrent-agent cap violated: '
        f'max={_max_concurrent(combined)} > 1'
    )
    assert wall < SOLVE_BUDGET_SEC, f'wall={wall:.2f}s'
    print(f'\n[milp-par] cap=1: S_2 T={r2["T"]} entry@{s2_entry_action} '
          f's_1_exit@{s1_last_action} wall={wall:.2f}s')
