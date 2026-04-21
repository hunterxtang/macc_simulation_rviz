"""MILP→CBS fallback tests.

Verifies that when ``plan_structure`` returns ``T=None`` for a
substructure (e.g. because ``T_max`` is below the sub's ``lower_bound_T``),
``milp_adapter.plan_group`` falls back to ``cbs_adapter.plan_group``
for that single sub, splices its per-robot Steps into the accumulator
at the current ``t_offset``, and continues with the remaining subs.

Run manually:
  ~/macc-venv/bin/python3 -m pytest test/test_milp_fallback.py -v -s
"""

import numpy as np  # noqa: I100, I101
import pytest  # noqa: I100, I101

pytest.importorskip('gurobipy')  # skip everything if Gurobi missing

from macc_rviz.planners import milp_adapter  # noqa: I100, I101


def _silent(*_a, **_k):
    pass


def _count_events(events_per_robot, kind):
    return sum(
        1 for ev_dict in events_per_robot
        for ev in ev_dict.values()
        if ev[0] == kind
    )


# ---------------------------------------------------------------------------
# Mixed planner per group: small subs hit MILP, oversized sub falls back to CBS
# ---------------------------------------------------------------------------


def test_milp_fallback_to_cbs_for_oversized_sub():
    """Three subs in one group: small / oversized / small.

    On a padded 7x7 MILP grid, a 1-block sub has feasible T=6 and a
    2-block-in-the-middle sub has feasible T=11.  Setting ``T_max=10``
    therefore lets the singleton subs solve via MILP but makes the
    middle sub's MILP sweep return ``T=None``.  The fallback MUST kick
    in for that sub alone — subs 0 and 2 still solve via MILP.
    """
    Z, Y, X = 2, 5, 5
    world_init = np.zeros((Z, Y, X), dtype=int)

    # Sub 0: 1 block at (1, 2). MILP T=6 on padded 7x7.
    sub0 = np.zeros_like(world_init)
    sub0[0, 2, 1] = 1

    # Sub 1: 2 blocks at (1, 1) and (1, 3). MILP T=11 (> T_max=10) → None.
    sub1 = np.zeros_like(world_init)
    sub1[0, 1, 1] = 1
    sub1[0, 3, 1] = 1

    # Sub 2: 1 block at (3, 2). MILP T~6.
    sub2 = np.zeros_like(world_init)
    sub2[0, 2, 3] = 1

    result = milp_adapter.plan_group(
        substructures=[sub0, sub1, sub2],
        substructure_indices=[0, 1, 2],
        world_init=world_init,
        robot_starts=[(0, 0)],
        num_robots=1,
        X=X, Y=Y,
        T_max=10,  # forces sub 1 to fail (its MILP T_min for feasibility is 11)
        per_t_time_limit=10.0,
        total_time_limit=60.0,
        cbs_max_t=200,
        cbs_branch_limit=50,
        logger=_silent,
    )

    md = result['metadata']
    sub_meta = md['substructure_metadata']

    assert md['used_fallback'] is True, (
        'expected used_fallback=True with T_max forcing sub 1 to None'
    )
    assert len(sub_meta) == 3, f'expected 3 substructure metadata rows, got {len(sub_meta)}'

    # Sub 0: MILP
    assert sub_meta[0]['si'] == 0
    assert sub_meta[0]['planner_used'] == 'milp', (
        f'sub 0 expected milp, got {sub_meta[0]["planner_used"]}'
    )
    assert sub_meta[0]['n_trips'] is not None

    # Sub 1: CBS fallback
    assert sub_meta[1]['si'] == 1
    assert sub_meta[1]['planner_used'] == 'cbs_fallback'
    assert sub_meta[1]['T'] > 0, 'CBS fallback produced empty plan'
    assert 'milp_solve_time' in sub_meta[1]

    # Sub 2: MILP — proves the loop CONTINUED past the fallback.
    assert sub_meta[2]['si'] == 2
    assert sub_meta[2]['planner_used'] == 'milp', (
        f'sub 2 expected milp, got {sub_meta[2]["planner_used"]}'
    )

    # All 4 blocks (1 + 2 + 1) must be placed.
    place_events = _count_events(result['events'], 'place')
    pickup_events = _count_events(result['events'], 'pickup')
    assert place_events == 4, f'expected 4 places, got {place_events}'
    # MILP places via R-vars without an explicit pickup event (carries on entry);
    # CBS's _build_robot_plan emits one pickup per block. So expect ≥ 2.
    assert pickup_events >= 2, f'expected ≥ 2 pickups (CBS path), got {pickup_events}'

    # cumulative_T_min reflects all 3 subs.  T_final is the actual fused
    # makespan and may be < T_min (MILP's per-sub T is a budget, not the
    # exact trip length).
    assert md['T_min'] > 8, f'T_min should aggregate across 3 subs, got {md["T_min"]}'
    assert md['T_final'] > 0

    # planner_mix counters
    n_milp = sum(1 for m in sub_meta if m['planner_used'] == 'milp')
    n_cbs = sum(1 for m in sub_meta if m['planner_used'] == 'cbs_fallback')
    assert n_milp == 2 and n_cbs == 1, (
        f'expected planner mix 2 MILP / 1 CBS, got {n_milp} / {n_cbs}'
    )

    print(
        f'\n[milp-fallback] mix=milp:{n_milp}/cbs:{n_cbs} '
        f'T_min={md["T_min"]} T_final={md["T_final"]} '
        f'solve={md["solve_time"]:.2f}s'
    )


# ---------------------------------------------------------------------------
# Happy path: planner_used == "milp" for every sub when MILP doesn't time out
# ---------------------------------------------------------------------------


def test_planner_used_is_milp_on_happy_path():
    """No fallback when T_max is generous and the subs are small."""
    Z, Y, X = 2, 5, 5
    world_init = np.zeros((Z, Y, X), dtype=int)

    sub0 = np.zeros_like(world_init)
    sub0[0, 2, 1] = 1
    sub1 = np.zeros_like(world_init)
    sub1[0, 2, 3] = 1

    result = milp_adapter.plan_group(
        substructures=[sub0, sub1],
        substructure_indices=[0, 1],
        world_init=world_init,
        robot_starts=[(0, 0)],
        num_robots=1,
        X=X, Y=Y,
        T_max=10,
        per_t_time_limit=10.0,
        total_time_limit=60.0,
        logger=_silent,
    )

    md = result['metadata']
    sub_meta = md['substructure_metadata']

    assert md['used_fallback'] is False
    assert all(m['planner_used'] == 'milp' for m in sub_meta), (
        f'expected all-MILP, got {[m["planner_used"] for m in sub_meta]}'
    )
    assert len(sub_meta) == 2

    # Back-compat alias
    assert md['per_sub'] is sub_meta or md['per_sub'] == sub_meta


# ---------------------------------------------------------------------------
# Replay invariants: per-robot Step counts agree, every tick has a Step
# ---------------------------------------------------------------------------


def test_fallback_plan_has_step_per_tick_per_robot():
    """The replay loop assumes one Step per robot per tick.  After the
    fallback splice, all robots' plans must have the same length and
    Step.t values must form a contiguous 1..T_final sequence.

    Uses 1 robot — a single agent on 1 / 2 / 1 blocks with T_max=10
    forces the middle sub to fall back to CBS (its MILP T_min on the
    padded 7x7 grid is 11)."""
    Z, Y, X = 2, 5, 5
    world_init = np.zeros((Z, Y, X), dtype=int)
    sub0 = np.zeros_like(world_init); sub0[0, 2, 1] = 1
    sub1 = np.zeros_like(world_init); sub1[0, 1, 1] = 1; sub1[0, 3, 1] = 1
    sub2 = np.zeros_like(world_init); sub2[0, 2, 3] = 1

    result = milp_adapter.plan_group(
        substructures=[sub0, sub1, sub2],
        substructure_indices=[0, 1, 2],
        world_init=world_init,
        robot_starts=[(0, 0)],
        num_robots=1,
        X=X, Y=Y,
        T_max=10,
        per_t_time_limit=10.0,
        total_time_limit=60.0,
        cbs_max_t=200,
        cbs_branch_limit=50,
        logger=_silent,
    )

    plans = result['plans']
    T_final = result['metadata']['T_final']

    # Used CBS fallback for sub 1 → verify it actually fired.
    assert result['metadata']['used_fallback'] is True

    # Every robot's plan ends at T_final (after final pad).
    for i, p in enumerate(plans):
        assert len(p) == T_final, (
            f'robot {i} plan length {len(p)} != T_final {T_final}'
        )
        ts = [s.t for s in p]
        assert ts == list(range(1, T_final + 1)), (
            f'robot {i} Step.t sequence not contiguous: '
            f'{ts[:5]}...{ts[-5:]}'
        )
