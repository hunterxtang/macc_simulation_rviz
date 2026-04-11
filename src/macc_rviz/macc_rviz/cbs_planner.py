"""Space-time A* planner and CBS-lite conflict resolver for MACC robots.

``space_time_astar`` plans one robot's path in (x, y, t) space-time.

``cbs_plan`` runs the joint CBS-lite solver: plans all robots'
full trajectories (depot → pickup → placement, repeated per block),
detects the first vertex or swap conflict, branches by adding a
forbidden-state constraint to one robot and replanning it, and repeats
until the joint plan is conflict-free or the branch limit is hit.
On branch-limit exhaustion the caller should serialise robots.

Reference: Section IV-C of the MACC paper (Lam et al. [1]).  The MILP
treats all robots as one joint flow through a time-expanded graph; CBS
matches this structure by enforcing collision-freeness at solve time
rather than as a post-hoc fix.

Constraint format (used by both phases)
---------------------------------------
Constraints are plain tuples passed as a ``frozenset`` or ``set``:

  Vertex constraint : ``(x, y, t)``
      Robot may not occupy cell (x, y) at timestep t.

  Edge constraint   : ``(x1, y1, x2, y2, t)``
      Robot may not traverse the directed edge (x1,y1) → (x2,y2)
      during tick t → t+1.  Used to prevent swap conflicts.
"""

import heapq
from collections import namedtuple


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: One step in a planned path.
#:
#: action – ACTION_* constant describing what the robot did this tick.
#: x, y, z – grid cell the robot occupies *after* this action.
#: t – absolute timestep at which the robot occupies that cell.
Step = namedtuple('Step', ['action', 'x', 'y', 'z', 't'])

ACTION_WAIT = 'wait'
ACTION_MOVE = 'move'
ACTION_PICKUP = 'pickup'
ACTION_PLACE = 'place'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reachable_neighbors(x, y, hm):
    """Return (nx, ny) cells reachable from (x, y) with a ±1 height step."""
    Y, X = hm.shape
    z = int(hm[y, x])
    result = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < X and 0 <= ny < Y:
            if abs(int(hm[ny, nx]) - z) <= 1:
                result.append((nx, ny))
    return result


def _reconstruct_path(came_from, terminal, hm):
    """Trace ``came_from`` back from ``terminal`` and return a Step list."""
    path = []
    state = terminal
    while came_from[state] is not None:
        prev_state, action = came_from[state]
        x, y, t = state
        path.append(Step(action, x, y, int(hm[y, x]), t))
        state = prev_state
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# space_time_astar — Phase 2 deliverable
# ---------------------------------------------------------------------------

def space_time_astar(
    start,
    goal,
    hm,
    t_start=0,
    constraints=frozenset(),
    max_t=200,
):
    """Plan a single robot's collision-free path in (x, y, t) space-time.

    The robot starts at ``start`` at timestep ``t_start`` and must reach
    ``goal``.  The heightmap ``hm`` is treated as static for the duration
    of the plan.  Permissible actions per timestep are: move to any
    4-compass neighbour whose height differs by at most ±1, or wait in
    place.  Pickup and place actions are inserted by the caller around
    the navigation plan; they are not generated here.

    Parameters
    ----------
    start : (int, int)
        (x, y) of the robot's current cell.
    goal : (int, int)
        (x, y) of the target cell.
    hm : numpy.ndarray
        Heightmap with shape (Y, X).  ``hm[y, x]`` is the number of
        blocks stacked at column (x, y).
    t_start : int
        Absolute timestep at which the robot is at ``start``.
    constraints : set or frozenset
        Forbidden-state constraints for *this* robot.  Each element is
        either a vertex constraint ``(x, y, t)`` or an edge constraint
        ``(x1, y1, x2, y2, t)``.  Empty by default; Phase 3 populates
        this when branching on detected conflicts.
    max_t : int
        Hard time horizon.  States at timestep >= ``max_t`` are pruned.
        If no path is found within this window, ``None`` is returned and
        the CBS-lite caller should fall back to serial execution.

    Returns
    -------
    list[Step] or None
        Sequence of Steps from ``start`` (exclusive) to ``goal``
        (inclusive).  Each Step records the action taken, the cell
        occupied, and the absolute timestep.  Returns an empty list if
        ``start == goal`` and ``None`` if no path exists within ``max_t``.

    Notes
    -----
    Heuristic: Manhattan distance to goal — admissible because each tick
    can change x or y by at most 1.  Tie-breaking on h prefers states
    spatially closer to the goal, reducing unnecessary waiting detours.
    """
    sx, sy = start
    gx, gy = goal

    if (sx, sy) == (gx, gy):
        return []

    # --- partition constraints by kind for O(1) lookup ------------------
    vertex_constraints = set()
    edge_constraints = set()
    for c in constraints:
        if len(c) == 3:
            vertex_constraints.add(c)   # (x, y, t)
        elif len(c) == 5:
            edge_constraints.add(c)     # (x1, y1, x2, y2, t)

    def _h(x, y):
        return abs(x - gx) + abs(y - gy)

    # --- A* open set: (f, h, g, x, y, t) --------------------------------
    # Secondary key on h breaks f-ties in favour of spatial progress;
    # tertiary key on g prevents identical-h ties being broken by t (which
    # would bias towards waiting).
    start_state = (sx, sy, t_start)
    open_set = [(_h(sx, sy), _h(sx, sy), 0, sx, sy, t_start)]
    came_from = {start_state: None}      # state → (prev_state, action)
    g_score = {start_state: 0}

    while open_set:
        _f, _hval, g, x, y, t = heapq.heappop(open_set)

        # --- goal check --------------------------------------------------
        if (x, y) == (gx, gy):
            return _reconstruct_path(came_from, (x, y, t), hm)

        # --- prune stale entries and time horizon ------------------------
        if t >= max_t:
            continue
        if g > g_score.get((x, y, t), float('inf')):
            continue

        nt = t + 1

        # --- generate successors: wait + 4-compass moves -----------------
        successors = [((x, y), ACTION_WAIT)]
        for nx, ny in _reachable_neighbors(x, y, hm):
            successors.append(((nx, ny), ACTION_MOVE))

        for (nx, ny), action in successors:
            # Vertex constraint: robot cannot be at (nx, ny) at nt
            if (nx, ny, nt) in vertex_constraints:
                continue
            # Edge constraint: robot cannot traverse (x,y)→(nx,ny) at t
            if (x, y, nx, ny, t) in edge_constraints:
                continue

            ns = (nx, ny, nt)
            ng = g + 1
            if ng < g_score.get(ns, float('inf')):
                g_score[ns] = ng
                came_from[ns] = ((x, y, t), action)
                nh = _h(nx, ny)
                heapq.heappush(
                    open_set, (ng + nh, nh, ng, nx, ny, nt)
                )

    return None  # no path found within max_t


# ---------------------------------------------------------------------------
# Phase 3: CBS-lite helpers
# ---------------------------------------------------------------------------

def _cell_at_time(start_xy, steps, t):
    """Return (x, y) occupied by a robot at absolute timestep t.

    Before the first step the robot is at ``start_xy``.  After the last
    step the robot stays at the final step's cell (implicit wait-at-goal).
    """
    if not steps or t < steps[0].t:
        return start_xy
    result = start_xy
    for s in steps:
        if s.t <= t:
            result = (s.x, s.y)
        else:
            break
    return result


def _placement_cell(bx, by, bz, robot_xy, hm):
    """Nearest valid 4-neighbour of (bx, by) from which to place at z=bz.

    Valid means in-bounds and reachable (height within ±1 of bz).
    Falls back to any in-bounds neighbour if none meet the height test.
    Returns None if (bx, by) is on the grid boundary with no neighbours.
    """
    Y, X = hm.shape
    rx, ry = robot_xy
    strict, relaxed = [], []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = bx + dx, by + dy
        if not (0 <= nx < X and 0 <= ny < Y):
            continue
        relaxed.append((nx, ny))
        if abs(int(hm[ny, nx]) - bz) <= 1:
            strict.append((nx, ny))
    pool = strict if strict else relaxed
    if not pool:
        return None
    return min(pool, key=lambda c: abs(c[0] - rx) + abs(c[1] - ry))


def _build_robot_plan(robot_start, blocks, hm, depot, t_start,
                      constraints, max_t):
    """Plan a single robot's full trajectory through all its assigned blocks.

    For each block the robot navigates to the depot, picks up, navigates
    to a placement cell, and places.  All navigation uses
    ``space_time_astar`` with the supplied constraints.

    Parameters
    ----------
    robot_start : (int, int)  —  starting (x, y).
    blocks      : list of (bx, by, bz, si) in placement order.
    hm          : static heightmap (Y, X).
    depot       : (int, int)  —  pickup location.
    t_start     : absolute timestep at which robot is at robot_start.
    constraints : set/frozenset of vertex/edge constraint tuples.
    max_t       : hard time horizon for each A* call.

    Returns
    -------
    (steps, events) or None.
      steps  – list[Step] covering all phases.
      events – dict {t: ('pickup', si)} or {t: ('place', bx, by, bz, si)}
               for the semantic actions embedded in the plan.
    """
    steps = []
    events = {}
    cur = tuple(robot_start)
    t = t_start

    for (bx, by, bz, si) in blocks:
        # 1. Navigate to depot (skip if already there)
        if cur != depot:
            nav = space_time_astar(cur, depot, hm, t, constraints, max_t)
            if nav is None:
                return None
            steps.extend(nav)
            t = nav[-1].t if nav else t
        cur = depot

        # 2. Pickup (one tick at depot)
        t += 1
        dz = int(hm[depot[1], depot[0]])
        steps.append(Step(ACTION_PICKUP, depot[0], depot[1], dz, t))
        events[t] = ('pickup', si)

        # 3. Navigate to placement cell
        pcell = _placement_cell(bx, by, bz, cur, hm)
        if pcell is None:
            return None
        if cur != pcell:
            nav = space_time_astar(cur, pcell, hm, t, constraints, max_t)
            if nav is None:
                return None
            steps.extend(nav)
            t = nav[-1].t if nav else t
        cur = pcell

        # 4. Place (one tick)
        t += 1
        pz = int(hm[cur[1], cur[0]])
        steps.append(Step(ACTION_PLACE, cur[0], cur[1], pz, t))
        events[t] = ('place', bx, by, bz, si)

    return steps, events


def _detect_first_conflict(robot_starts, plans):
    """Find the first vertex or swap conflict across all robot pairs.

    Returns ``(i, j, constraint)`` where constraint is a tuple to add to
    robot j's forbidden set:
      vertex  → (x, y, t)
      edge    → (x1, y1, x2, y2, t)   blocks the move (x1,y1)→(x2,y2) at t

    Returns None when the joint plan is conflict-free.
    """
    n = len(plans)
    if n < 2:
        return None

    t_max = max(
        (s[-1].t for s in plans if s),
        default=0,
    )

    for t in range(t_max + 2):   # +2 to catch post-plan swap at final t
        positions = [
            _cell_at_time(start, plan, t)
            for start, plan in zip(robot_starts, plans)
        ]
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = positions[i]
                xj, yj = positions[j]

                # Vertex conflict
                if (xi, yi) == (xj, yj):
                    return (i, j, (xj, yj, t))

                # Swap conflict: robots exchange cells at t → t+1
                xi2, yi2 = _cell_at_time(robot_starts[i], plans[i], t + 1)
                xj2, yj2 = _cell_at_time(robot_starts[j], plans[j], t + 1)
                if (xi2, yi2) == (xj, yj) and (xj2, yj2) == (xi, yi):
                    # Constrain j: forbid traversal (xj,yj)→(xi,yi) at t
                    return (i, j, (xj, yj, xi, yi, t))

    return None


def _serial_fallback(robot_starts, block_assignments, hm, depot, t_start,
                     max_t):
    """Plan robots one at a time (guaranteed conflict-free).

    Robot 0 plans from t_start; each subsequent robot starts immediately
    after the previous one finishes, so no two robots are active at once.
    Returns (plans, events_per_robot).
    """
    plans = []
    events_per_robot = []
    t = t_start

    for start, blocks in zip(robot_starts, block_assignments):
        if not blocks:
            plans.append([])
            events_per_robot.append({})
            continue
        result = _build_robot_plan(
            start, blocks, hm, depot, t, frozenset(), max_t,
        )
        if result is None:
            # Unreachable goal — treat as empty plan
            plans.append([])
            events_per_robot.append({})
            continue
        robot_steps, robot_events = result
        plans.append(robot_steps)
        events_per_robot.append(robot_events)
        # Next robot starts one tick after this one finishes
        t = (robot_steps[-1].t + 1) if robot_steps else t

    return plans, events_per_robot


# ---------------------------------------------------------------------------
# Phase 3: cbs_plan — joint CBS-lite planner
# ---------------------------------------------------------------------------

def cbs_plan(robot_starts, block_assignments, hm, depot,
             t_start=0, max_t=400, branch_limit=50):
    """Joint CBS-lite planner for N robots.

    Plans all robots' full block-delivery trajectories jointly, resolving
    vertex and swap conflicts by adding forbidden-state constraints to
    individual robots and replanning them (Section IV-C).

    Parameters
    ----------
    robot_starts      : list of (x, y)  — one per robot.
    block_assignments : list of list of (bx, by, bz, si)  — one per robot.
    hm                : numpy.ndarray (Y, X), treated as static.
    depot             : (int, int)  — block pickup location.
    t_start           : int  — absolute timestep of robot starts.
    max_t             : int  — hard A* time horizon.
    branch_limit      : int  — max constraint additions before giving up.

    Returns
    -------
    (plans, events_per_robot, conflicts_resolved) or None.
      plans             – list[list[Step]], one per robot.
      events_per_robot  – list[dict], one per robot;
                          each dict maps t → ('pickup', si) or
                          ('place', bx, by, bz, si).
      conflicts_resolved – int, how many conflicts CBS resolved.
    Returns None when no conflict-free plan is found within branch_limit,
    signalling the caller to use serial execution as fallback.
    """
    n = len(robot_starts)
    constraints = [frozenset() for _ in range(n)]

    # --- initial independent plans ----------------------------------------
    plans = []
    events_per_robot = []
    for i in range(n):
        if not block_assignments[i]:
            plans.append([])
            events_per_robot.append({})
            continue
        result = _build_robot_plan(
            robot_starts[i], block_assignments[i], hm, depot,
            t_start, constraints[i], max_t,
        )
        if result is None:
            return None   # goal unreachable even without constraints
        robot_steps, robot_events = result
        plans.append(robot_steps)
        events_per_robot.append(robot_events)

    # --- CBS branching loop -----------------------------------------------
    branches = 0
    conflicts_resolved = 0

    while branches < branch_limit:
        conflict = _detect_first_conflict(robot_starts, plans)
        if conflict is None:
            return plans, events_per_robot, conflicts_resolved

        ri, rj, cj_tuple = conflict

        # Try constraining the higher-index robot (rj) first
        new_cj = constraints[rj] | {cj_tuple}
        result_j = _build_robot_plan(
            robot_starts[rj], block_assignments[rj], hm, depot,
            t_start, new_cj, max_t,
        )
        branches += 1

        if result_j is not None:
            constraints[rj] = new_cj
            plans[rj], events_per_robot[rj] = result_j
            conflicts_resolved += 1
            continue

        # Fall back to constraining the lower-index robot (ri)
        if len(cj_tuple) == 3:
            ci_tuple = cj_tuple          # same vertex constraint
        else:
            x1, y1, x2, y2, ct = cj_tuple
            ci_tuple = (x2, y2, x1, y1, ct)   # reverse the edge

        new_ci = constraints[ri] | {ci_tuple}
        result_i = _build_robot_plan(
            robot_starts[ri], block_assignments[ri], hm, depot,
            t_start, new_ci, max_t,
        )
        branches += 1

        if result_i is not None:
            constraints[ri] = new_ci
            plans[ri], events_per_robot[ri] = result_i
            conflicts_resolved += 1
            continue

        # Neither robot can be rerouted — give up
        break

    return None   # branch_limit hit; caller falls back to serial


# ---------------------------------------------------------------------------
# Return-to-boundary: CBS-lite over navigation-only plans
# ---------------------------------------------------------------------------

def cbs_return(robot_starts, goals, hm, t_start=0, max_t=200,
               branch_limit=50):
    """CBS-lite for simultaneous return-to-boundary navigation.

    Parameters
    ----------
    robot_starts : list of (x, y)
    goals        : list of (x, y)  — one boundary goal per robot.
    hm           : numpy.ndarray (Y, X).
    t_start, max_t, branch_limit : same as ``cbs_plan``.

    Returns
    -------
    list[list[Step]] or None.
      One navigation Step list per robot (no pickup/place events).
      None triggers a serial fallback in the caller.
    """
    n = len(robot_starts)
    constraints = [frozenset() for _ in range(n)]

    # Initial independent plans
    plans = []
    for i in range(n):
        if robot_starts[i] == goals[i]:
            plans.append([])
            continue
        p = space_time_astar(
            robot_starts[i], goals[i], hm, t_start, constraints[i], max_t,
        )
        if p is None:
            return None
        plans.append(p)

    branches = 0
    while branches < branch_limit:
        conflict = _detect_first_conflict(robot_starts, plans)
        if conflict is None:
            return plans

        ri, rj, cj_tuple = conflict

        new_cj = constraints[rj] | {cj_tuple}
        new_plan_j = space_time_astar(
            robot_starts[rj], goals[rj], hm, t_start, new_cj, max_t,
        )
        branches += 1
        if new_plan_j is not None:
            constraints[rj] = new_cj
            plans[rj] = new_plan_j
            continue

        if len(cj_tuple) == 3:
            ci_tuple = cj_tuple
        else:
            x1, y1, x2, y2, ct = cj_tuple
            ci_tuple = (x2, y2, x1, y1, ct)

        new_ci = constraints[ri] | {ci_tuple}
        new_plan_i = space_time_astar(
            robot_starts[ri], goals[ri], hm, t_start, new_ci, max_t,
        )
        branches += 1
        if new_plan_i is not None:
            constraints[ri] = new_ci
            plans[ri] = new_plan_i
            continue

        break

    return None
