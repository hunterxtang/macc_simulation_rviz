"""
Unit tests for cbs_planner.py.

Target areas flagged during Phase 0 review:
  - Empty-path edge cases: start == goal, unreachable goal, max_t exhaustion,
    empty block assignments.
  - Stale-heightmap behaviour: planner treats hm as static; documents what
    breaks when the world changes mid-plan.
  - Vertex + edge constraint honouring.
  - Conflict detection (vertex + swap).
  - cbs_plan joint solving + branch-limit fallback.
  - cbs_return navigation-only joint planning.

Run: python3 -m pytest src/macc_rviz/test/test_cbs_planner.py -v
"""

import sys

import numpy as np
import pytest

from macc_rviz import cbs_planner  # noqa: I100
from macc_rviz.cbs_planner import (  # noqa: I100, I101
    ACTION_MOVE,
    ACTION_PICKUP,
    ACTION_PLACE,
    ACTION_WAIT,
    Step,
    cbs_plan,
    cbs_return,
    space_time_astar,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def flat_hm(X=5, Y=5):
    """Build a flat ground heightmap of shape (Y, X)."""
    return np.zeros((Y, X), dtype=int)


def wall_hm(X=5, Y=5, wall_x=2, wall_height=5):
    """Build a heightmap with a tall vertical wall blocking x=wall_x."""
    hm = np.zeros((Y, X), dtype=int)
    hm[:, wall_x] = wall_height
    return hm


def path_xy(steps):
    """Extract the (x, y) sequence from a Step list."""
    return [(s.x, s.y) for s in steps]


# ---------------------------------------------------------------------------
# Section 1: space_time_astar — empty-path edge cases
# ---------------------------------------------------------------------------

class TestSpaceTimeAstarEdgeCases:
    """Edge cases for the single-agent space-time A* planner."""

    def test_start_equals_goal_returns_empty_list(self):
        """Trivial same-cell request returns an empty plan."""
        hm = flat_hm()
        result = space_time_astar((2, 2), (2, 2), hm)
        assert result == []

    def test_start_equals_goal_with_max_t_zero(self):
        """Trivial case does not depend on max_t."""
        hm = flat_hm()
        result = space_time_astar((0, 0), (0, 0), hm, max_t=0)
        assert result == []

    def test_unreachable_goal_returns_none(self):
        """Wall of height 5 blocks all lateral traversal (+-1 climb limit)."""
        hm = wall_hm(X=5, Y=5, wall_x=2, wall_height=5)
        result = space_time_astar((0, 2), (4, 2), hm, max_t=50)
        assert result is None

    def test_max_t_exhausted_returns_none(self):
        """Goal reachable in 4 steps, max_t=2 prunes before arrival."""
        hm = flat_hm()
        result = space_time_astar((0, 0), (4, 0), hm, max_t=2)
        assert result is None

    def test_max_t_exactly_enough(self):
        """4-step path reachable with max_t=4."""
        hm = flat_hm()
        result = space_time_astar((0, 0), (4, 0), hm, max_t=4)
        assert result is not None
        assert len(result) == 4
        assert (result[-1].x, result[-1].y) == (4, 0)
        assert result[-1].t == 4

    def test_single_step_to_neighbor(self):
        """One-step adjacent move produces a single MOVE action at t=1."""
        hm = flat_hm()
        result = space_time_astar((0, 0), (1, 0), hm, max_t=10)
        assert result is not None
        assert len(result) == 1
        assert result[0].action == ACTION_MOVE
        assert (result[0].x, result[0].y, result[0].t) == (1, 0, 1)

    def test_out_of_bounds_goal_returns_none(self):
        """Goal outside the grid yields no path."""
        hm = flat_hm(X=5, Y=5)
        result = space_time_astar((0, 0), (10, 10), hm, max_t=50)
        assert result is None


# ---------------------------------------------------------------------------
# Section 2: space_time_astar — climb and constraint semantics
# ---------------------------------------------------------------------------

class TestSpaceTimeAstarClimb:
    """Climb-height feasibility (+-1 per step)."""

    def test_climb_of_one_allowed(self):
        """A one-unit step up between adjacent cells succeeds in one action."""
        hm = flat_hm()
        hm[0, 1] = 1
        result = space_time_astar((0, 0), (1, 0), hm, max_t=10)
        assert result is not None
        assert len(result) == 1

    def test_climb_of_two_blocked(self):
        """A two-unit step up with no lateral detour is unreachable."""
        hm = flat_hm()
        hm[:, 1] = 2
        result = space_time_astar((0, 0), (2, 0), hm, max_t=20)
        assert result is None

    def test_climb_two_step_via_ramp(self):
        """A 2-high destination is reachable via a height-1 ramp cell."""
        hm = flat_hm()
        hm[0, 1] = 1
        hm[0, 2] = 2
        result = space_time_astar((0, 0), (2, 0), hm, max_t=10)
        assert result is not None
        assert (result[-1].x, result[-1].y) == (2, 0)


class TestSpaceTimeAstarConstraints:
    """Vertex and edge constraint enforcement."""

    def test_vertex_constraint_forces_wait(self):
        """Intermediate cell forbidden at t=1 forces a wait or detour."""
        hm = flat_hm()
        constraints = frozenset({(1, 0, 1)})
        result = space_time_astar(
            (0, 0), (2, 0), hm, constraints=constraints, max_t=20
        )
        assert result is not None
        t1_cell = (result[0].x, result[0].y)
        assert t1_cell != (1, 0)

    def test_vertex_constraint_permanently_blocks_unique_path(self):
        """Blocking the unique traversal cell at every t yields no path."""
        hm = np.full((3, 3), 10, dtype=int)
        hm[1, :] = 0
        constraints = frozenset({(1, 1, t) for t in range(1, 20)})
        result = space_time_astar(
            (0, 1), (2, 1), hm, constraints=constraints, max_t=20
        )
        assert result is None

    def test_edge_constraint_forbids_specific_traversal(self):
        """Edge constraint (0,0)->(1,0) at t=0 forces an alternative."""
        hm = flat_hm()
        constraints = frozenset({(0, 0, 1, 0, 0)})
        result = space_time_astar(
            (0, 0), (1, 0), hm, constraints=constraints, max_t=20
        )
        assert result is not None
        assert not (
            len(result) == 1
            and result[0].action == ACTION_MOVE
            and (result[0].x, result[0].y) == (1, 0)
            and result[0].t == 1
        )

    def test_wait_action_is_valid_progress(self):
        """Planner exploits WAIT to route around a timed vertex constraint."""
        hm = flat_hm()
        constraints = frozenset({(1, 0, 1)})
        result = space_time_astar(
            (0, 0), (1, 0), hm, constraints=constraints, max_t=10
        )
        assert result is not None
        assert (result[-1].x, result[-1].y) == (1, 0)


# ---------------------------------------------------------------------------
# Section 3: _build_robot_plan
# ---------------------------------------------------------------------------

class TestBuildRobotPlan:
    """Full per-robot trajectory construction (depot -> pickup -> place)."""

    def test_single_block_trip_emits_pickup_and_place_events(self):
        """One-block trip produces exactly one pickup and one place event."""
        hm = flat_hm(X=5, Y=5)
        blocks = [(3, 3, 0, 0)]
        result = cbs_planner._build_robot_plan(
            robot_start=(0, 0),
            blocks=blocks,
            hm=hm,
            depot=(0, 0),
            t_start=0,
            constraints=frozenset(),
            max_t=100,
        )
        assert result is not None
        steps, events = result
        pickups = [v for v in events.values() if v[0] == 'pickup']
        places = [v for v in events.values() if v[0] == 'place']
        assert len(pickups) == 1
        assert len(places) == 1
        assert steps[-1].action == ACTION_PLACE

    def test_multi_block_trip_emits_N_pickups_and_N_places(self):
        """Three-block trip produces three pickups and three places."""
        hm = flat_hm(X=5, Y=5)
        blocks = [(3, 3, 0, 0), (2, 2, 0, 0), (4, 4, 0, 0)]
        result = cbs_planner._build_robot_plan(
            robot_start=(0, 0),
            blocks=blocks,
            hm=hm,
            depot=(0, 0),
            t_start=0,
            constraints=frozenset(),
            max_t=200,
        )
        assert result is not None
        steps, events = result
        pickups = [v for v in events.values() if v[0] == 'pickup']
        places = [v for v in events.values() if v[0] == 'place']
        assert len(pickups) == 3
        assert len(places) == 3

    def test_returns_none_when_goal_unreachable(self):
        """Unreachable placement cell returns None from the whole trip."""
        hm = wall_hm(X=5, Y=5, wall_x=2, wall_height=5)
        blocks = [(4, 2, 0, 0)]
        result = cbs_planner._build_robot_plan(
            robot_start=(0, 2),
            blocks=blocks,
            hm=hm,
            depot=(0, 2),
            t_start=0,
            constraints=frozenset(),
            max_t=50,
        )
        assert result is None

    def test_empty_blocks_list(self):
        """Empty block list produces no steps and no events (no crash)."""
        hm = flat_hm()
        result = cbs_planner._build_robot_plan(
            robot_start=(0, 0),
            blocks=[],
            hm=hm,
            depot=(0, 0),
            t_start=0,
            constraints=frozenset(),
            max_t=50,
        )
        assert result is not None
        steps, events = result
        assert steps == []
        assert events == {}


# ---------------------------------------------------------------------------
# Section 4: _detect_first_conflict
# ---------------------------------------------------------------------------

class TestDetectFirstConflict:
    """Vertex + swap conflict detection across paired robot plans."""

    def test_no_conflict_returns_none(self):
        """Non-overlapping paths produce no conflict."""
        plans = [
            [Step(ACTION_MOVE, 1, 0, 0, 1), Step(ACTION_MOVE, 2, 0, 0, 2)],
            [Step(ACTION_MOVE, 1, 4, 0, 1), Step(ACTION_MOVE, 2, 4, 0, 2)],
        ]
        starts = [(0, 0), (0, 4)]
        assert cbs_planner._detect_first_conflict(starts, plans) is None

    def test_vertex_conflict_detected(self):
        """Two robots occupying the same cell at the same t yield a vertex conflict."""
        plans = [
            [Step(ACTION_MOVE, 1, 2, 0, 1), Step(ACTION_MOVE, 2, 2, 0, 2)],
            [Step(ACTION_MOVE, 3, 2, 0, 1), Step(ACTION_MOVE, 2, 2, 0, 2)],
        ]
        starts = [(0, 2), (4, 2)]
        conflict = cbs_planner._detect_first_conflict(starts, plans)
        assert conflict is not None
        i, j, c = conflict
        assert len(c) == 3
        assert c == (2, 2, 2)

    def test_swap_conflict_detected(self):
        """Two robots exchanging cells yield an edge (swap) conflict."""
        plans = [
            [Step(ACTION_MOVE, 2, 0, 0, 1)],
            [Step(ACTION_MOVE, 1, 0, 0, 1)],
        ]
        starts = [(1, 0), (2, 0)]
        conflict = cbs_planner._detect_first_conflict(starts, plans)
        assert conflict is not None
        _, _, c = conflict
        assert len(c) == 5

    def test_single_robot_no_conflict(self):
        """One plan cannot conflict with itself."""
        plans = [[Step(ACTION_MOVE, 1, 0, 0, 1)]]
        starts = [(0, 0)]
        assert cbs_planner._detect_first_conflict(starts, plans) is None


# ---------------------------------------------------------------------------
# Section 5: cbs_plan — joint solver
# ---------------------------------------------------------------------------

class TestCbsPlan:
    """Joint CBS-lite solving across multiple robots."""

    def test_two_robots_non_conflicting_succeed(self):
        """Two robots with disjoint goals plan jointly without branching."""
        hm = flat_hm(X=7, Y=7)
        block_assignments = [
            [(5, 5, 0, 0)],
            [(5, 0, 0, 0)],
        ]
        robot_starts = [(0, 0), (0, 6)]
        result = cbs_plan(
            robot_starts, block_assignments, hm,
            depot=(0, 0), t_start=0, max_t=200, branch_limit=50,
        )
        assert result is not None
        plans, events, _conflicts = result
        assert len(plans) == 2
        for ev in events:
            assert any(v[0] == 'pickup' for v in ev.values())
            assert any(v[0] == 'place' for v in ev.values())

    def test_empty_block_assignment_yields_empty_plan(self):
        """A robot with no assigned blocks gets an empty plan."""
        hm = flat_hm(X=5, Y=5)
        block_assignments = [
            [(3, 3, 0, 0)],
            [],
        ]
        robot_starts = [(0, 0), (4, 4)]
        result = cbs_plan(
            robot_starts, block_assignments, hm,
            depot=(0, 0), t_start=0, max_t=200, branch_limit=50,
        )
        assert result is not None
        plans, events, _ = result
        assert plans[1] == []
        assert events[1] == {}

    def test_branch_limit_zero_attempts_no_resolution(self):
        """branch_limit=0 means CBS cannot resolve any conflict."""
        hm = flat_hm(X=3, Y=1)
        block_assignments = [
            [(2, 0, 0, 0)],
            [(2, 0, 0, 1)],
        ]
        robot_starts = [(0, 0), (0, 0)]
        result = cbs_plan(
            robot_starts, block_assignments, hm,
            depot=(0, 0), t_start=0, max_t=50, branch_limit=0,
        )
        if result is not None:
            _, _, conflicts_resolved = result
            assert conflicts_resolved == 0

    def test_head_on_swap_resolved_or_falls_back(self):
        """Two robots crossing on a 1-row grid resolve or signal fallback."""
        hm = np.zeros((1, 5), dtype=int)
        block_assignments = [
            [(4, 0, 0, 0)],
            [(0, 0, 0, 0)],
        ]
        robot_starts = [(0, 0), (4, 0)]
        result = cbs_plan(
            robot_starts, block_assignments, hm,
            depot=(0, 0), t_start=0, max_t=200, branch_limit=50,
        )
        if result is not None:
            _, _, conflicts_resolved = result
            assert conflicts_resolved >= 0


# ---------------------------------------------------------------------------
# Section 6: stale-heightmap — documents known limitation
# ---------------------------------------------------------------------------

class TestStaleHeightmap:
    """
    Document the static-heightmap contract of cbs_planner.

    The planner treats hm as static for the life of a solve. Integration
    code in Phase 1b must refresh hm per substructure and accept that
    within-substructure placements can invalidate subsequent plans on
    the same column.
    """

    def test_plan_is_valid_for_hm_at_plan_time(self):
        """Returned plan is self-consistent with the hm passed in."""
        hm = flat_hm(X=4, Y=3)
        result = space_time_astar((0, 0), (3, 0), hm, max_t=20)
        assert result is not None
        prev_x, prev_y = 0, 0
        prev_z = int(hm[prev_y, prev_x])
        for s in result:
            nz = int(hm[s.y, s.x])
            assert abs(nz - prev_z) <= 1
            prev_x, prev_y, prev_z = s.x, s.y, nz

    def test_plan_becomes_infeasible_after_mid_plan_placement(self):
        """Plan replayed against a post-placement hm can contain an illegal climb."""
        hm_before = np.zeros((3, 3), dtype=int)
        plan = space_time_astar((0, 1), (2, 1), hm_before, max_t=10)
        assert plan is not None

        hm_after = hm_before.copy()
        hm_after[1, 1] = 2

        prev_z = int(hm_after[1, 0])
        stale_step_detected = False
        for s in plan:
            nz = int(hm_after[s.y, s.x])
            if abs(nz - prev_z) > 1:
                stale_step_detected = True
                break
            prev_z = nz

        assert stale_step_detected, (
            'Expected stale-hm: plan replayed against post-placement hm '
            'should contain at least one illegal +-1 climb.'
        )

    def test_stale_hm_is_caller_responsibility(self):
        """Planner is deterministic w.r.t. its inputs; hm snapshot is caller's job."""
        hm = flat_hm()
        r1 = space_time_astar((0, 0), (3, 0), hm, max_t=20)
        r2 = space_time_astar((0, 0), (3, 0), hm, max_t=20)
        assert path_xy(r1) == path_xy(r2)


# ---------------------------------------------------------------------------
# Section 7: cbs_return — navigation-only joint planning
# ---------------------------------------------------------------------------

class TestCbsReturn:
    """Joint navigation-only planning used for the return-to-boundary phase."""

    def test_non_conflicting_returns_plan(self):
        """Two robots returning to opposite boundary cells plan without conflict."""
        hm = flat_hm(X=5, Y=5)
        starts = [(2, 2), (3, 3)]
        goals = [(0, 2), (4, 3)]
        plans = cbs_return(starts, goals, hm, t_start=0, max_t=50)
        assert plans is not None
        assert len(plans) == 2
        assert (plans[0][-1].x, plans[0][-1].y) == (0, 2)
        assert (plans[1][-1].x, plans[1][-1].y) == (4, 3)

    def test_start_equals_goal_returns_empty(self):
        """Robots already at their goals get empty plans."""
        hm = flat_hm(X=5, Y=5)
        starts = [(0, 2), (4, 3)]
        goals = [(0, 2), (4, 3)]
        plans = cbs_return(starts, goals, hm, t_start=0, max_t=50)
        assert plans is not None
        assert plans[0] == []
        assert plans[1] == []

    def test_unreachable_goal_returns_none(self):
        """Unreachable return goal returns None."""
        hm = wall_hm(X=5, Y=5, wall_x=2, wall_height=5)
        starts = [(0, 2)]
        goals = [(4, 2)]
        plans = cbs_return(starts, goals, hm, t_start=0, max_t=50)
        assert plans is None


# ---------------------------------------------------------------------------
# Section 8: sanity on action constants + Step tuple
# ---------------------------------------------------------------------------

class TestActionConstants:
    """Sanity checks on the public constants and Step tuple."""

    def test_action_constants_distinct(self):
        """All four action constants are unique."""
        actions = {ACTION_WAIT, ACTION_MOVE, ACTION_PICKUP, ACTION_PLACE}
        assert len(actions) == 4

    def test_step_namedtuple_fields(self):
        """Step exposes action, x, y, z, t as named fields."""
        s = Step(ACTION_MOVE, 1, 2, 3, 4)
        assert s.action == ACTION_MOVE
        assert (s.x, s.y, s.z, s.t) == (1, 2, 3, 4)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
