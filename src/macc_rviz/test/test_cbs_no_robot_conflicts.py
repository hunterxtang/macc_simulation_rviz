"""Regression test: CBS serial-per-block plans must be collision-free.

Bug history: ``cbs_adapter.plan_group`` called ``_build_robot_plan`` with
``constraints=frozenset()``, so the active robot's A* ignored the cells
where idle robots were WAIT-padded.  On ``example_structure`` with 4
robots this produced 14 vertex conflicts (one robot walking through
another's parked cell).  Fixed by injecting those idle-robot cells as
``(x, y, t)`` vertex constraints for the full planning window.

This test reconstructs per-tick positions from the plan output and
asserts two things:

  1. No two robots ever occupy the same (x, y) at the same tick
     (vertex conflict).
  2. No two robots swap cells between adjacent ticks (edge conflict).

The only exception is the trivial "starts" case — robots may legitimately
share the same starting cell at tick 0 if the caller passes colocated
starts.  We exclude starting cells from the vertex check.
"""

import numpy as np

from macc_rviz.planners import cbs_adapter
from macc_rviz.structure_utils import create_example_structure
from macc_rviz.decomposition import decompose_structure, order_substructures
from macc_rviz.parallel import find_parallel_groups


def _simulate_positions(plans, robot_starts, max_t):
    """Reconstruct (x, y) position per robot per tick from a plan list."""
    n = len(plans)
    positions = [[None] * (max_t + 1) for _ in range(n)]
    for i in range(n):
        for t in range(max_t + 1):
            positions[i][t] = robot_starts[i]
        for s in plans[i]:
            if 0 <= s.t <= max_t:
                positions[i][s.t] = (s.x, s.y)
        for t in range(1, max_t + 1):
            if positions[i][t] is None:
                positions[i][t] = positions[i][t - 1]
    return positions


def _find_conflicts(positions, start_cells):
    """Scan positions for vertex / swap conflicts.

    Ignores vertex collisions at ``start_cells`` (robots legitimately
    share their boot-up cell before their first trip starts).
    """
    n = len(positions)
    if n == 0:
        return [], []
    max_t = len(positions[0]) - 1
    vertex = []
    swap = []
    for t in range(max_t + 1):
        for i in range(n):
            for j in range(i + 1, n):
                pi = positions[i][t]
                pj = positions[j][t]
                if pi == pj and pi not in start_cells:
                    vertex.append((t, i, j, pi))
                if t < max_t:
                    pi2 = positions[i][t + 1]
                    pj2 = positions[j][t + 1]
                    if pi2 == pj and pj2 == pi and pi != pj:
                        swap.append((t, i, j, pi, pj))
    return vertex, swap


def _run_example_structure(num_robots):
    """Drive cbs_adapter.plan_group across all decomposition groups and
    return (flattened plans, reconstructed positions, robot_starts).
    """
    target = create_example_structure()
    substructures = decompose_structure(target)
    _order, deps = order_substructures(substructures)
    groups = find_parallel_groups(substructures, deps)

    Z, Y, X = target.shape
    world_sim = np.zeros_like(target, dtype=int)
    # Match the sim's actual boot-up layout (Robots spread along y=0).
    robot_starts = [(i % X, 0) for i in range(num_robots)]
    initial_starts = list(robot_starts)

    all_plans = [[] for _ in range(num_robots)]
    for group in groups:
        subs = [substructures[si] for si in group]
        result = cbs_adapter.plan_group(
            substructures=subs,
            substructure_indices=list(group),
            world_init=world_sim,
            robot_starts=robot_starts,
            num_robots=num_robots,
            X=X, Y=Y, t_start=0, max_t=400,
        )
        for i in range(num_robots):
            all_plans[i].extend(result['plans'][i])

        for si in group:
            world_sim[substructures[si] == 1] = 1
        robot_starts = [
            (p[-1].x, p[-1].y) if p else robot_starts[i]
            for i, p in enumerate(result['plans'])
        ]

    max_t = max((p[-1].t for p in all_plans if p), default=0)
    positions = _simulate_positions(all_plans, initial_starts, max_t)
    return all_plans, positions, initial_starts


def test_plan_group_on_example_structure_no_vertex_conflicts():
    """No two robots occupy the same (x, y) at the same tick."""
    _plans, positions, starts = _run_example_structure(num_robots=4)
    vertex, _swap = _find_conflicts(positions, start_cells=set(starts))
    assert not vertex, (
        f'plan_group produced {len(vertex)} vertex conflicts on '
        f'example_structure (first few: {vertex[:5]})'
    )


def test_plan_group_on_example_structure_no_swap_conflicts():
    """No two robots swap cells between adjacent ticks."""
    _plans, positions, starts = _run_example_structure(num_robots=4)
    _vertex, swap = _find_conflicts(positions, start_cells=set(starts))
    assert not swap, (
        f'plan_group produced {len(swap)} swap conflicts on '
        f'example_structure (first few: {swap[:5]})'
    )


def test_plan_group_scales_to_many_robots_still_conflict_free():
    """Same guarantee with more robots sharing the depot row."""
    _plans, positions, starts = _run_example_structure(num_robots=6)
    vertex, swap = _find_conflicts(positions, start_cells=set(starts))
    assert not vertex, f'{len(vertex)} vertex conflicts at 6 robots: {vertex[:5]}'
    assert not swap, f'{len(swap)} swap conflicts at 6 robots: {swap[:5]}'
