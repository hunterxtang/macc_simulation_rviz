"""Regression tests for the per-round replan + heightmap-refresh fix.

Bug history: ``cbs_adapter.plan_group`` previously called ``cbs_plan``
once with the heightmap snapshot at group entry, then planned every
robot's full multi-block trajectory against that single static hm.
After robot A placed a block at column (cx, cy, 0), robot B's
already-planned path could traverse column (cx, cy) at z=0 even
though the true hm at that tick was 1+ — the robot visually
"clipped" through the placed block stack.

Fix: ``plan_group`` now plans one round at a time (one block per
robot per round), with a fresh ``heightmap_from_world`` between
rounds.  After all rounds, the per-robot Step list reflects each
robot's true z at every tick, given a replay that tracks world
state per the events.

These tests exercise the no-clip invariant on a structure that
specifically used to fail: stacks higher than 1.
"""

import numpy as np

from macc_rviz.cbs_planner import ACTION_PLACE
from macc_rviz.planners import cbs_adapter


def _replay_and_check_no_clip(result, world_init):
    """Walk every Step in absolute-t order; assert robot z agrees with hm.

    For each Step, compute hm from cur_world at that tick (i.e. after
    applying every 'place' event with t' < step.t) and assert that
    ``hm[step.y, step.x] == step.z`` — the robot stands on top of
    the column.  Pickup/place events at the same tick are applied
    *after* the position check so the robot is allowed to be at the
    cell whose top it is about to place onto.

    Returns (clip_count, total_steps, sample_clip) for diagnostics.
    """
    plans = result['plans']
    events = result['events']
    cur_world = world_init.astype(int).copy()

    # Index every (robot_id, Step) by t for ordered traversal.
    by_t = {}
    for ri, plan in enumerate(plans):
        for s in plan:
            by_t.setdefault(s.t, []).append((ri, s))

    clip_count = 0
    total = 0
    sample = None
    for t in sorted(by_t):
        # First apply any *pickup* events at this t (they happen at the
        # cell the robot already occupies; do not affect height check).
        # Defer 'place' events until after the position check.
        for ri, s in by_t[t]:
            ev = events[ri].get(t)
            if ev is not None and ev[0] == 'pickup' and len(ev) == 5:
                _, bx, by, bz, _si = ev
                if bx >= 0 and cur_world[bz, by, bx] == 1:
                    cur_world[bz, by, bx] = 0
            true_h = int(cur_world[:, s.y, s.x].sum())
            total += 1
            if s.z != true_h:
                clip_count += 1
                if sample is None:
                    sample = (t, ri, s, true_h)
        # Now apply place events at this t.
        for ri, s in by_t[t]:
            ev = events[ri].get(t)
            if ev is not None and ev[0] == 'place':
                _, bx, by, bz, _si = ev
                cur_world[bz, by, bx] = 1
    return clip_count, total, sample


def test_central_tower_no_clip_4_robots():
    """example_structure-style central 3-block tower; 4 robots.

    Pre-fix this produced ~8 clipping instances (robot at z=0 with
    true_h up to 3 — through the central tower).  After per-round
    replan with hm refresh, the invariant must hold.
    """
    Z, Y, X = 3, 5, 5
    sub = np.array([
        [[0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
    ], dtype=int)
    world_init = np.zeros((Z, Y, X), dtype=int)
    result = cbs_adapter.plan_group(
        substructures=[sub],
        substructure_indices=[0],
        world_init=world_init,
        robot_starts=[(i, 0) for i in range(4)],
        num_robots=4,
        X=X, Y=Y,
        t_start=0,
        max_t=400,
        branch_limit=200,
    )
    clips, total, sample = _replay_and_check_no_clip(result, world_init)
    assert clips == 0, (
        f'{clips}/{total} steps clipped. First: {sample}. '
        f'plan_group did not refresh heightmap mid-build.'
    )
    # Also assert all 13 blocks land in the correct cells.
    final = world_init.astype(int).copy()
    for ri, ev_dict in enumerate(result['events']):
        for tev, ev in ev_dict.items():
            if ev[0] == 'place':
                _, bx, by, bz, _ = ev
                final[bz, by, bx] = 1
    assert int(final.sum()) == 13
    assert np.array_equal(final, sub)


def test_metadata_reports_blocks_planned():
    """plan_group's metadata exposes per-block telemetry for the serial design.

    Fix 2: per-round metadata replaced by per-block — every block in the
    group runs through its own _build_robot_plan call, and the count is
    surfaced for logging.
    """
    sub = np.zeros((2, 3, 3), dtype=int)
    sub[0, 1, 1] = 1
    sub[1, 1, 1] = 1
    world = np.zeros_like(sub)
    result = cbs_adapter.plan_group(
        substructures=[sub],
        substructure_indices=[0],
        world_init=world,
        robot_starts=[(0, 0), (2, 0)],
        num_robots=2,
        X=3, Y=3,
    )
    md = result['metadata']
    assert 'blocks_planned' in md
    assert md['blocks_planned'] == 2
    assert md['conflicts_resolved'] == 0  # no joint solve in serial mode


def test_per_round_padding_keeps_steps_contiguous():
    """Concatenated per-robot plans have step.t = cur_t+1, cur_t+2, ..., T_final."""
    sub = np.zeros((1, 5, 5), dtype=int)
    sub[0, 1, 1] = 1
    sub[0, 3, 3] = 1
    world = np.zeros_like(sub)
    result = cbs_adapter.plan_group(
        substructures=[sub],
        substructure_indices=[0],
        world_init=world,
        robot_starts=[(0, 0), (4, 4)],
        num_robots=2,
        X=5, Y=5,
    )
    T_final = result['metadata']['T_final']
    for i, plan in enumerate(result['plans']):
        if not plan:
            continue
        ts = [s.t for s in plan]
        # Every robot should have a Step at every tick from its first to T_final.
        assert ts[0] >= 1
        assert ts[-1] == T_final, (
            f'robot {i} last step t={ts[-1]} != T_final {T_final}'
        )
        assert ts == sorted(ts)
        # No duplicate ticks.
        assert len(set(ts)) == len(ts)
        # No internal gaps.
        for a, b in zip(ts, ts[1:]):
            assert b == a + 1, f'robot {i} has tick gap: {a} → {b}'


def test_robot_z_tracks_climb_when_walking_over_placed_block():
    """Robot z must track the climb when walking over a placed block.

    A robot whose path crosses a column already stacked by the prior round
    should have ``Step.z`` matching the true height at that tick — i.e. the
    plan acknowledges the climb that the replay will execute.
    """
    Z, Y, X = 2, 3, 3
    # Build a single 2-block column at (1,1).  Two robots; one places (1,1,0)
    # in round 0, one places (1,1,1) in round 1 — round 1's robot must be
    # planned standing at z>=1 in some cell adjacent to (1,1) since the
    # support is now z=1.
    sub = np.zeros((Z, Y, X), dtype=int)
    sub[0, 1, 1] = 1
    sub[1, 1, 1] = 1
    world_init = np.zeros_like(sub)
    result = cbs_adapter.plan_group(
        substructures=[sub],
        substructure_indices=[0],
        world_init=world_init,
        robot_starts=[(0, 0), (2, 2)],
        num_robots=2,
        X=X, Y=Y,
        t_start=0,
        max_t=200,
        branch_limit=100,
    )
    clips, total, sample = _replay_and_check_no_clip(result, world_init)
    assert clips == 0, (
        f'{clips}/{total} steps clipped. First: {sample}.'
    )
    # Both blocks placed.
    place_count = sum(
        sum(1 for ev in evs.values() if ev[0] == ACTION_PLACE)
        for evs in result['events']
    )
    assert place_count == 2

    # And the robot that placed z=1 was standing at z>=1 the moment it placed.
    # (Sanity: any 'place' event with bz=1 implies the placing robot's Step
    # at that t had z within ±1 of 1.)
    for ri, evs in enumerate(result['events']):
        for tev, ev in evs.items():
            if ev[0] == ACTION_PLACE and ev[3] == 1:  # bz == 1
                # Find robot ri's Step at tev.
                step_at_t = next(
                    (s for s in result['plans'][ri] if s.t == tev),
                    None,
                )
                assert step_at_t is not None, (
                    f'robot {ri} place event at t={tev} but no Step at that t'
                )
                # Robot should be standing on top of an adjacent column;
                # since support height is 1 (block at (1,1,0) placed earlier),
                # robot stands at a neighbor with hm in [0, 2].  We assert
                # z is within ±1 of 1 (the placed block's height).
                assert abs(step_at_t.z - 1) <= 1, (
                    f'robot {ri} placing at z=1 was at z={step_at_t.z}'
                )
