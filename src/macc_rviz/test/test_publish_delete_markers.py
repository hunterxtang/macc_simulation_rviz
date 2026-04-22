"""Regression tests for ``publish_blocks`` emitting DELETE markers.

Bug history: ``publish_blocks`` only emitted ``Marker.ADD`` for voxels
with ``self.world == 1`` and silently skipped empty cells.  MILP scaffold
teardown correctly cleared ``self.world`` to 0 but the previously-ADDed
marker lived on in RViz's state forever (teal, because ``world_sub``
stored 0 → ``_sub_color(-1)`` wraps to tab10 cyan), producing a visible
"scaffolding border" around the structure that CBS never showed.

The fix: track the last-published occupancy and emit ``Marker.DELETE``
for voxels that transitioned 1 → 0 since the previous publish.

These tests drive ``publish_blocks`` directly via a minimal shim, so no
rclpy / Gurobi is required.
"""

from types import SimpleNamespace

import macc_rviz.macc_rviz_sim as sim_mod

import numpy as np

from visualization_msgs.msg import Marker


class _FakePublisher:
    def __init__(self):
        self.last = None

    def publish(self, ma):
        self.last = ma


class _FakeClock:
    def __init__(self):
        self._now = SimpleNamespace(to_msg=lambda: SimpleNamespace())

    def now(self):
        return self._now


def _make_sim(Z=2, Y=3, X=3):
    """Minimal object exposing just what ``publish_blocks`` reads."""
    s = SimpleNamespace()
    s.H = Z
    s.Y = Y
    s.X = X
    s.block_scale = 1.0
    s.world = np.zeros((Z, Y, X), dtype=int)
    s.world_sub = np.zeros((Z, Y, X), dtype=int)
    s._prev_published_world = np.zeros((Z, Y, X), dtype=int)
    s.just_completed_sis = set()
    s.blocks_pub = _FakePublisher()
    s.get_clock = _FakeClock
    return s


def _publish(s):
    sim_mod.MACCRvizSim.publish_blocks(s)
    return s.blocks_pub.last.markers


def _mid(s, x, y, z):
    return int(z) * s.Y * s.X + int(y) * s.X + int(x)


def test_first_publish_emits_only_adds():
    s = _make_sim()
    s.world[0, 1, 1] = 1
    s.world_sub[0, 1, 1] = 1

    markers = _publish(s)
    assert len(markers) == 1
    assert markers[0].action == Marker.ADD
    assert markers[0].id == _mid(s, 1, 1, 0)


def test_cleared_voxel_gets_delete_marker():
    s = _make_sim()
    # Tick 1: place a scaffold cell (si=-1 → world_sub stays 0, teal).
    s.world[0, 1, 1] = 1
    s.world_sub[0, 1, 1] = 0
    markers = _publish(s)
    assert [m.action for m in markers] == [Marker.ADD]

    # Tick 2: pickup clears the cell.
    s.world[0, 1, 1] = 0
    s.world_sub[0, 1, 1] = 0
    markers = _publish(s)
    deletes = [m for m in markers if m.action == Marker.DELETE]
    adds = [m for m in markers if m.action == Marker.ADD]
    assert len(deletes) == 1
    assert deletes[0].id == _mid(s, 1, 1, 0)
    assert len(adds) == 0


def test_steady_state_no_delete_noise():
    s = _make_sim()
    s.world[0, 0, 0] = 1
    s.world_sub[0, 0, 0] = 1
    _publish(s)

    # Re-publish with no state change: must not emit DELETE for the
    # already-occupied cell.
    markers = _publish(s)
    assert all(m.action != Marker.DELETE for m in markers)
    assert len(markers) == 1
    assert markers[0].action == Marker.ADD


def test_mixed_add_and_delete_in_one_publish():
    s = _make_sim()
    # Tick 1: scaffold at (1,1,0) and target at (2,2,0).
    s.world[0, 1, 1] = 1
    s.world[0, 2, 2] = 1
    s.world_sub[0, 1, 1] = 0   # si=-1 → teal
    s.world_sub[0, 2, 2] = 1   # si=0 → blue
    _publish(s)

    # Tick 2: scaffold cleared, new target at (0,0,1).
    s.world[0, 1, 1] = 0
    s.world[0, 2, 2] = 1
    s.world[1, 0, 0] = 1
    s.world_sub[0, 1, 1] = 0
    s.world_sub[1, 0, 0] = 2
    markers = _publish(s)

    ids_by_action = {Marker.ADD: set(), Marker.DELETE: set()}
    for m in markers:
        ids_by_action[m.action].add(m.id)

    assert _mid(s, 1, 1, 0) in ids_by_action[Marker.DELETE]
    assert _mid(s, 0, 0, 1) in ids_by_action[Marker.ADD]
    # The still-occupied (2,2,0) is re-ADDed (keeps RViz display fresh)
    # but never DELETEd.
    assert _mid(s, 2, 2, 0) not in ids_by_action[Marker.DELETE]


def test_scaffold_roundtrip_leaves_no_stale_marker():
    """End-to-end: place scaffold, publish, pickup scaffold, publish.

    Final marker state in RViz (the union of every ADD since the last
    matching DELETE) must contain no live marker for the scaffold cell.
    """
    s = _make_sim()
    scaffold = (1, 1, 0)
    mid = _mid(s, *scaffold)

    live = set()

    def apply(markers):
        for m in markers:
            if m.action == Marker.ADD:
                live.add(m.id)
            elif m.action == Marker.DELETE:
                live.discard(m.id)

    # Place scaffold.
    s.world[scaffold[2], scaffold[1], scaffold[0]] = 1
    s.world_sub[scaffold[2], scaffold[1], scaffold[0]] = 0
    apply(_publish(s))
    assert mid in live

    # Pickup scaffold.
    s.world[scaffold[2], scaffold[1], scaffold[0]] = 0
    s.world_sub[scaffold[2], scaffold[1], scaffold[0]] = 0
    apply(_publish(s))
    assert mid not in live, (
        'scaffold marker must be DELETEd when world drops to 0'
    )
