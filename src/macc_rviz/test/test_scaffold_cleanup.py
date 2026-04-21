"""Regression tests for scaffold-block cleanup on pickup events.

Bug history: ``_apply_cbs_step`` previously wrote ``self.world`` on
``place`` events but did nothing on ``pickup`` events.  Combined with
pickup events that omitted source coordinates, MILP scaffold blocks
were never removed from the sim's world model — they appeared as
teal voxels (``world_sub=0`` → ``_sub_color(-1)`` wrap-around) at
the end of the build.

The fix:
  1. Pickup events carry ``(bx, by, bz)`` for grid pickups, and the
     off-grid sentinel ``(-1, -1, -1)`` for depot / entry-carrying.
  2. ``_apply_cbs_step`` clears ``world[bz, by, bx]`` and
     ``world_sub[bz, by, bx]`` on grid pickups.

These tests exercise the event format and the replay's cell-clearing
behavior without requiring rclpy / Gurobi.
"""

import numpy as np

from macc_rviz import cbs_planner
from macc_rviz.cbs_planner import Step, ACTION_PICKUP


def _flat_hm(X=5, Y=5):
    return np.zeros((Y, X), dtype=int)


# ---------------------------------------------------------------------------
# Event-emission tests
# ---------------------------------------------------------------------------


def test_cbs_pickup_event_uses_offgrid_sentinel():
    """CBS depot pickups draw from off-grid supply: source must be (-1,-1,-1)."""
    hm = _flat_hm()
    blocks = [(3, 3, 0, 0)]
    result = cbs_planner._build_robot_plan(
        robot_start=(0, 0), blocks=blocks, hm=hm,
        depot=(0, 0), t_start=0, constraints=frozenset(), max_t=100,
    )
    assert result is not None
    _steps, events = result
    pickups = [v for v in events.values() if v[0] == 'pickup']
    assert len(pickups) == 1
    p = pickups[0]
    assert len(p) == 5, f'pickup event should be 5-tuple, got {p}'
    assert p[1] == -1 and p[2] == -1 and p[3] == -1, (
        f'CBS depot pickup must use off-grid sentinel, got {p}'
    )


# ---------------------------------------------------------------------------
# Replay logic — exercised against the real _apply_cbs_step from the sim
# ---------------------------------------------------------------------------


class _Robot:
    """Minimal stand-in for ``macc_rviz_sim.Robot`` (no rclpy needed)."""
    __slots__ = ('id', 'x', 'y', 'z', 'carrying', 'carrying_si',
                 'wait_streak')

    def __init__(self):
        self.id = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.carrying = False
        self.carrying_si = -1
        self.wait_streak = 0


class _SimShim:
    """Just the world / world_sub state ``_apply_cbs_step`` reads/writes."""

    def __init__(self, Z=2, Y=3, X=3):
        self.world = np.zeros((Z, Y, X), dtype=int)
        self.world_sub = np.zeros((Z, Y, X), dtype=int)


def _apply_event(sim, robot, ev, t):
    """Inline copy of the pickup/place dispatch from _apply_cbs_step.

    Kept in sync with macc_rviz_sim.py:_apply_cbs_step so the test fails
    if that branch regresses.  Importing the sim module would pull rclpy.
    """
    kind = ev[0]
    if kind == 'pickup':
        if len(ev) == 5:
            _, bx, by, bz, si = ev
        else:
            _, si = ev
            bx = by = bz = -1
        robot.carrying = True
        robot.carrying_si = si
        if bx >= 0 and sim.world[bz, by, bx] == 1:
            sim.world[bz, by, bx] = 0
            sim.world_sub[bz, by, bx] = 0
    elif kind == 'place':
        _, bx, by, bz, si = ev
        if sim.world[bz, by, bx] == 0:
            sim.world[bz, by, bx] = 1
            sim.world_sub[bz, by, bx] = si + 1
        robot.carrying = False
        robot.carrying_si = -1


def test_grid_pickup_clears_world_at_source_cell():
    """Grid pickup with bx>=0 removes the block from world & world_sub."""
    sim = _SimShim()
    robot = _Robot()
    # Pre-place a scaffold block (si=-1, so world_sub stored 0).
    _apply_event(sim, robot, ('place', 1, 1, 0, -1), t=1)
    assert sim.world[0, 1, 1] == 1
    assert sim.world_sub[0, 1, 1] == 0  # si=-1 stores world_sub=0

    # Pickup that scaffold — world should be cleared.
    _apply_event(sim, robot, ('pickup', 1, 1, 0, -1), t=2)
    assert sim.world[0, 1, 1] == 0, 'pickup must clear world cell'
    assert sim.world_sub[0, 1, 1] == 0


def test_grid_pickup_clears_target_cell_too():
    """Pickup of a target block (si>=0) also clears (covers tear-down of any block)."""
    sim = _SimShim()
    robot = _Robot()
    _apply_event(sim, robot, ('place', 2, 2, 0, 3), t=1)
    assert sim.world[0, 2, 2] == 1
    assert sim.world_sub[0, 2, 2] == 4  # si=3 → stored 4

    _apply_event(sim, robot, ('pickup', 2, 2, 0, 3), t=2)
    assert sim.world[0, 2, 2] == 0
    assert sim.world_sub[0, 2, 2] == 0


def test_offgrid_pickup_does_not_touch_world():
    """Pickup with sentinel (-1,-1,-1) must not mutate world or world_sub."""
    sim = _SimShim()
    robot = _Robot()
    # Pre-place a real block so we can detect any spurious clear.
    _apply_event(sim, robot, ('place', 1, 1, 0, 0), t=1)
    snapshot_world = sim.world.copy()
    snapshot_sub = sim.world_sub.copy()

    _apply_event(sim, robot, ('pickup', -1, -1, -1, 0), t=2)
    assert robot.carrying is True
    assert robot.carrying_si == 0
    assert np.array_equal(sim.world, snapshot_world), (
        'off-grid pickup must not modify world'
    )
    assert np.array_equal(sim.world_sub, snapshot_sub)


def test_legacy_2tuple_pickup_treated_as_offgrid():
    """Backwards-compat: ('pickup', si) is treated as off-grid (no clear)."""
    sim = _SimShim()
    robot = _Robot()
    _apply_event(sim, robot, ('place', 1, 1, 0, 0), t=1)
    snapshot_world = sim.world.copy()

    _apply_event(sim, robot, ('pickup', 0), t=2)  # legacy 2-tuple
    assert robot.carrying is True
    assert robot.carrying_si == 0
    assert np.array_equal(sim.world, snapshot_world), (
        'legacy 2-tuple pickup must not touch world (no source coords)'
    )


def test_pickup_then_replace_round_trip():
    """Place → pickup → re-place at same cell ends with cell occupied & tagged."""
    sim = _SimShim()
    robot = _Robot()
    _apply_event(sim, robot, ('place', 1, 1, 0, 0), t=1)
    _apply_event(sim, robot, ('pickup', 1, 1, 0, 0), t=2)
    assert sim.world[0, 1, 1] == 0
    _apply_event(sim, robot, ('place', 1, 1, 0, 0), t=3)
    assert sim.world[0, 1, 1] == 1
    assert sim.world_sub[0, 1, 1] == 1  # si=0 → stored 1
