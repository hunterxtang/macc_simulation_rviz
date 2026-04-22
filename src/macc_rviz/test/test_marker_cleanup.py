"""Regression tests for cross-run RViz marker cleanup.

Bug history (Bug C / D):

- On node startup the previous sim's markers linger in RViz because no
  DELETEALL is published before the new run's ADDs start streaming.
- On shutdown the node must re-publish DELETEALL so Ctrl+C followed by a
  relaunch does not show both runs overlaid.
- After Stage 5 completion the tick timer cancels itself but rclpy.spin()
  keeps idling; ``sim_done`` now signals main() to unwind cleanly.

These tests drive ``_wipe_all_topics`` / ``_cleanup_markers`` with a
minimal shim (no rclpy required), mirroring test_publish_delete_markers.
"""

from types import SimpleNamespace

import macc_rviz.macc_rviz_sim as sim_mod

from visualization_msgs.msg import Marker


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, ma):
        self.published.append(ma)


class _FakeClock:
    def now(self):
        return SimpleNamespace(to_msg=lambda: SimpleNamespace())


def _make_shim():
    """Build a stand-in for MACCRvizSim that only exposes publishers + clock."""
    s = SimpleNamespace()
    s.blocks_pub = _FakePublisher()
    s.robots_pub = _FakePublisher()
    s.ghost_pub = _FakePublisher()
    s.text_pub = _FakePublisher()
    s.intro_pub = _FakePublisher()
    s.stage_label_pub = _FakePublisher()
    s.get_clock = _FakeClock
    # Bind the real methods so internal `self.<method>(...)` calls resolve.
    s._deleteall = lambda pub, *ns: sim_mod.MACCRvizSim._deleteall(s, pub, *ns)
    s._wipe_all_topics = lambda: sim_mod.MACCRvizSim._wipe_all_topics(s)
    return s


def _all_pubs(s):
    return [
        s.blocks_pub,
        s.robots_pub,
        s.ghost_pub,
        s.text_pub,
        s.intro_pub,
        s.stage_label_pub,
    ]


def test_wipe_all_topics_emits_deleteall_on_every_publisher():
    s = _make_shim()
    sim_mod.MACCRvizSim._wipe_all_topics(s)

    for pub in _all_pubs(s):
        assert len(pub.published) == 1, 'each /macc/* topic must receive one MarkerArray'
        ma = pub.published[0]
        assert len(ma.markers) >= 1
        assert all(m.action == Marker.DELETEALL for m in ma.markers), (
            'every marker in the startup/shutdown wipe must be DELETEALL'
        )


def test_cleanup_markers_invokes_wipe_and_cancels_timer():
    s = _make_shim()

    class _Timer:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    s.timer = _Timer()
    sim_mod.MACCRvizSim._cleanup_markers(s)

    assert s.timer.cancelled, 'tick timer must be cancelled before DELETEALL publishes'
    for pub in _all_pubs(s):
        assert pub.published, f'{pub} must receive a DELETEALL on shutdown'
        assert all(
            m.action == Marker.DELETEALL
            for ma in pub.published
            for m in ma.markers
        )


def test_cleanup_markers_tolerates_missing_timer():
    """Startup failure path: _cleanup_markers may run before self.timer exists."""
    s = _make_shim()
    # Intentionally no s.timer — _cleanup_markers should not AttributeError.
    sim_mod.MACCRvizSim._cleanup_markers(s)

    for pub in _all_pubs(s):
        assert pub.published, 'wipe must still fire even when no timer is set'
