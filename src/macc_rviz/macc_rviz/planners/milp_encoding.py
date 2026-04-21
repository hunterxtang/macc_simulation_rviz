"""MILP encoding of the MACC problem from Lam et al. (CP 2020).

Reference
---------
  E. Lam, P. J. Stuckey, S. Koenig, T. K. S. Kumar.
  Exact Approaches to the Multi-Agent Collective Construction Problem.
  Principles and Practice of Constraint Programming (CP), 2020. §4, Fig. 2.

Paper notation is preserved so readers can diff this module against the
paper constraint-by-constraint:

  - Action 9-tuple: i = (t, x, y, z, c, a, x', y', z')
  - Off-grid sentinels: S (source) for entries R_1, E (dest) for exits R_4
  - Action types K = {M, P, D} — move/wait, pickup, deliver
  - R = R_1 ∪ ... ∪ R_6 (subsets defined on pp. 6-7)
  - H = height transitions (t, x, y, z, z') with |z'-z| ≤ 1
  - P = all grid (x,y) positions; B = border cells; C = interior cells
  - A = num_agents (constraint 11)

Constraints (2)-(16) are posted in Fig. 2 order with inline comments
naming each label.
"""

import numpy as np

import gurobipy as gp
from gurobipy import GRB


# Off-grid sentinels (paper §4).
S_SENT = 'S'
E_SENT = 'E'

# Action types K = {M, P, D}.
M = 'M'
P = 'P'
D = 'D'


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _neighbors(x, y, X, Y):
    """4-connected neighbors of (x,y), clipped to the grid."""
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < X and 0 <= ny < Y:
            out.append((nx, ny))
    return out


def _is_border(x, y, X, Y):
    return x == 0 or y == 0 or x == X - 1 or y == Y - 1


def border_cells(X, Y):
    return [(x, y) for y in range(Y) for x in range(X)
            if _is_border(x, y, X, Y)]


def interior_cells(X, Y):
    return [(x, y) for y in range(Y) for x in range(X)
            if not _is_border(x, y, X, Y)]


def _all_positions(X, Y):
    return [(x, y) for y in range(Y) for x in range(X)]


def _valid_robot_cells(X, Y, z_max):
    """Return all (x,y,z) a robot can physically occupy: interior ×
    {0..z_max} ∪ border × {0}.  Used to widen constraints (7), (8), (12)."""
    out = []
    for (x, y) in _all_positions(X, Y):
        if _is_border(x, y, X, Y):
            out.append((x, y, 0))
        else:
            for z in range(z_max + 1):
                out.append((x, y, z))
    return out


# ---------------------------------------------------------------------------
# Parallel-substructure prior trajectories — §IV-C of the decomposition paper
# ---------------------------------------------------------------------------

def _derive_prior_state(prior_trajectories):
    """Flatten prior-agent trips into occupancy / edges / active-count.

    Parameters
    ----------
    prior_trajectories : list[list[Step]] | None
        One list per prior agent's trip (as returned by
        ``extract_agent_trips``).  Each Step needs attributes t, x, y, z.
        ``None`` or ``[]`` yields empty dicts.

    Returns
    -------
    vertex_xy   : dict[t, set[(x, y)]]         — (x,y) occupied at time t
    edges       : dict[t, set[(x1,y1,x2,y2)]]  — directed move at time t
    active_at_t : dict[t, int]                 — count of prior trips whose
                                                  action-time span covers t

    The action-time span of one trip is [t_e, t_last] where
    ``t_e = trip[0].t - 1`` is the R_1 entry time and
    ``t_last = trip[-1].t`` is the R_4 exit action time.
    """
    vertex_xy = {}
    edges = {}
    active_at_t = {}

    if not prior_trajectories:
        return vertex_xy, edges, active_at_t

    for trip in prior_trajectories:
        if not trip:
            continue
        for step in trip:
            vertex_xy.setdefault(step.t, set()).add((step.x, step.y))
        # Consecutive Steps whose (x,y) differ came from an R_2 MOVE at
        # action-time s2.t - 1, source (s1.x, s1.y) → dest (s2.x, s2.y).
        for i in range(len(trip) - 1):
            s1 = trip[i]
            s2 = trip[i + 1]
            if (s1.x, s1.y) != (s2.x, s2.y):
                edges.setdefault(
                    s2.t - 1, set()
                ).add((s1.x, s1.y, s2.x, s2.y))
        t_e = trip[0].t - 1
        t_last = trip[-1].t
        for t in range(t_e, t_last + 1):
            active_at_t[t] = active_at_t.get(t, 0) + 1

    return vertex_xy, edges, active_at_t


def agent_cap_from_priors(prior_trajectories, global_cap):
    """Build a ``max_agents_at_t`` dict from prior trips and a global cap.

    Returns {t: max(0, global_cap - active_count_at_t)} for every t at
    which some prior agent is active.  Times outside prior-activity
    windows are omitted; the MILP falls back to ``num_agents`` there.

    Phase 5 wires this in when a parallel group shares the fleet cap.
    """
    _, _, active = _derive_prior_state(prior_trajectories)
    return {t: max(0, int(global_cap) - n) for t, n in active.items()}


# ---------------------------------------------------------------------------
# Action-set generators (R_1..R_6, H)  — §4, pp. 6-7
# ---------------------------------------------------------------------------

def enumerate_actions(X, Y, z_max, T):
    """Enumerate (R_1..R_6) ∪ H as a flat list of 9-tuples and 5-tuples.

    Returns (R_list, H_list).  Each action in R_list is a 9-tuple; each
    transition in H_list is a 5-tuple (t, x, y, z, z').

    The sentinels S_SENT / E_SENT stand in for the off-grid start/end
    positions in R_1 and R_4 so the 9-tuple shape stays uniform across R.
    """
    R = []

    border = border_cells(X, Y)
    interior = interior_cells(X, Y)

    # R_1: enter at border, either empty or carrying.  t ∈ {0,...,T-4}.
    #   i = (t, S, S, S, c, M, x', y', 0),  (x',y') ∈ B.
    for t in range(T - 3):
        for (xp, yp) in border:
            for c in (0, 1):
                R.append((t, S_SENT, S_SENT, S_SENT, c, M, xp, yp, 0))

    # R_2: move ±1 height to a neighbor.  t ∈ {1,...,T-3}, (x,y,z) ∈ C.
    #   i = (t, x, y, z, c, M, x', y', z'), (x',y') ∈ N(x,y), |z'-z| ≤ 1.
    # NOTE: We read Fig. 2's C as "all physically valid robot positions":
    #   interior × {0..z_max}  ∪  border × {0}.  Restricting to interior
    #   alone makes entries unreachable from interior via R_2 (since R_1
    #   only drops robots at border).  Destination z' at border must be 0.
    for t in range(1, T - 2):
        for (x, y) in _all_positions(X, Y):
            x_is_border = _is_border(x, y, X, Y)
            z_src_range = [0] if x_is_border else range(z_max + 1)
            for z in z_src_range:
                for (xp, yp) in _neighbors(x, y, X, Y):
                    dst_is_border = _is_border(xp, yp, X, Y)
                    for zp in range(z_max + 1):
                        if dst_is_border and zp != 0:
                            continue
                        if abs(zp - z) <= 1:
                            for c in (0, 1):
                                R.append(
                                    (t, x, y, z, c, M, xp, yp, zp))

    # R_3: wait at same cell.  t ∈ {1,...,T-3}, (x,y,z) ∈ C.
    #   i = (t, x, y, z, c, M, x, y, z).
    for t in range(1, T - 2):
        for (x, y) in _all_positions(X, Y):
            x_is_border = _is_border(x, y, X, Y)
            z_range = [0] if x_is_border else range(z_max + 1)
            for z in z_range:
                for c in (0, 1):
                    R.append((t, x, y, z, c, M, x, y, z))

    # R_4: exit from border cell at ground level.  t ∈ {2,...,T-2}.
    #   i = (t, x, y, 0, c, M, E, E, E), (x,y) ∈ B.
    for t in range(2, T - 1):
        for (x, y) in border:
            for c in (0, 1):
                R.append((t, x, y, 0, c, M, E_SENT, E_SENT, E_SENT))

    # R_5: pickup block from neighbor at same z.  t ∈ {1,...,T-3},
    #   z ∈ {0,...,z_max-1}, (x',y') ∈ N(x,y), (x,y) ∈ P.
    #   i = (t, x, y, z, 0, P, x', y', z).
    # Constraint: robot source must be physically valid (border ⇒ z=0).
    for t in range(1, T - 2):
        for y in range(Y):
            for x in range(X):
                src_is_border = _is_border(x, y, X, Y)
                z_range = [0] if src_is_border else range(z_max)
                for z in z_range:
                    if z > z_max - 1:
                        continue
                    for (xp, yp) in _neighbors(x, y, X, Y):
                        R.append((t, x, y, z, 0, P, xp, yp, z))

    # R_6: deliver block to neighbor at same z.  t ∈ {1,...,T-3},
    #   z ∈ {0,...,z_max-1}, (x',y') ∈ N(x,y), (x,y) ∈ P.
    #   i = (t, x, y, z, 1, D, x', y', z).
    for t in range(1, T - 2):
        for y in range(Y):
            for x in range(X):
                src_is_border = _is_border(x, y, X, Y)
                z_range = [0] if src_is_border else range(z_max)
                for z in z_range:
                    if z > z_max - 1:
                        continue
                    for (xp, yp) in _neighbors(x, y, X, Y):
                        R.append((t, x, y, z, 1, D, xp, yp, z))

    # H: height transitions.  t ∈ {0,...,T-2}, (x,y) ∈ P,
    #   z, z' ∈ {0,...,z_max}, |z'-z| ≤ 1.
    #   h_{t,x,y,z,z'} = 1 iff pillar (x,y) has height z at time t and
    #   height z' at time t+1.
    H = []
    for t in range(T - 1):
        for y in range(Y):
            for x in range(X):
                for z in range(z_max + 1):
                    for zp in range(z_max + 1):
                        if abs(zp - z) <= 1:
                            H.append((t, x, y, z, zp))

    return R, H


# ---------------------------------------------------------------------------
# Index builders — for fast wildcard summation over action sets
# ---------------------------------------------------------------------------

def _build_R_indices(R_list):
    """Build the wildcard-lookup indices used by constraints (7)-(14)."""
    idx = {
        # (t, c, a, xp, yp, zp) → [i]  — actions ending in (xp,yp,zp)
        'by_end_cad': {},
        # (t, x, y, z, c, a) → [i]    — actions starting from (x,y,z)
        'by_src_ca': {},
        # (t, x, y) → [i]             — robot at (x,y) (any z/c/a/dest)
        'by_src_xy': {},
        # (t, a, xp, yp) → [i]        — pickup/deliver at block (xp,yp)
        'by_pd_target_xy': {},
        # (t, x, y, xp, yp) → [i]     — moves (x,y)→(x',y') (any z/z'/c)
        'by_edge_xy': {},
        # t → [i]                     — all actions at time t
        'by_time': {},
        # (t, xp, yp, zp) → [i]       — pickup target (x',y',z')
        'by_pickup_target_xyz': {},
        # (t, xp, yp, zp) → [i]       — deliver target (x',y',z')
        'by_deliver_target_xyz': {},
    }

    for i in R_list:
        t, x, y, z, c, a, xp, yp, zp = i

        k = (t, c, a, xp, yp, zp)
        idx['by_end_cad'].setdefault(k, []).append(i)

        k = (t, x, y, z, c, a)
        idx['by_src_ca'].setdefault(k, []).append(i)

        if x != S_SENT:
            k = (t, x, y)
            idx['by_src_xy'].setdefault(k, []).append(i)

        if a in (P, D):
            k = (t, a, xp, yp)
            idx['by_pd_target_xy'].setdefault(k, []).append(i)

        if a == M and x != S_SENT and xp != E_SENT and (x, y) != (xp, yp):
            k = (t, x, y, xp, yp)
            idx['by_edge_xy'].setdefault(k, []).append(i)

        idx['by_time'].setdefault(t, []).append(i)

        if a == P:
            k = (t, xp, yp, zp)
            idx['by_pickup_target_xyz'].setdefault(k, []).append(i)
        elif a == D:
            k = (t, xp, yp, zp)
            idx['by_deliver_target_xyz'].setdefault(k, []).append(i)

    return idx


def _build_H_indices(H_list):
    """Build wildcard-lookup indices over H for constraints (5), (6), (12)."""
    idx = {
        'by_col_to': {},    # (t, x, y, zp) → [i]  — H ending at height zp
        'by_col_from': {},  # (t, x, y, z)  → [i]  — H starting at height z
        'by_col': {},       # (t, x, y)     → [i]  — all H at (t,x,y)
    }
    for i in H_list:
        t, x, y, z, zp = i
        idx['by_col_to'].setdefault((t, x, y, zp), []).append(i)
        idx['by_col_from'].setdefault((t, x, y, z), []).append(i)
        idx['by_col'].setdefault((t, x, y), []).append(i)
    return idx


# ---------------------------------------------------------------------------
# Main encoder — posts constraints (1)..(16) of Fig. 2 in order
# ---------------------------------------------------------------------------

def build_model(
    init_hm,
    target_hm,
    num_agents,
    T,
    *,
    env=None,
    name='macc_milp',
    time_limit=60.0,
    mip_gap=0.0,
    threads=0,
    presolve=-1,
    output_flag=0,
    prior_trajectories=None,
    max_agents_at_t=None,
):
    """Build the Gurobi MILP for one MACC instance at horizon T.

    Parameters
    ----------
    init_hm, target_hm : ndarray (Y, X), int
        Current and desired pillar heights. ``init_hm == 0`` matches the
        paper's assumption; non-zero init is supported via a modified
        constraint (3) that fixes ``h_0`` to the initial heightmap.
    num_agents : int
        A in constraint (11).
    T : int
        Planning horizon. Must be ≥ 4 (otherwise R_1/R_4 time windows
        are empty).
    env : gurobipy.Env, optional
    name : str
    time_limit, mip_gap, threads, presolve, output_flag
        Gurobi parameters. ``threads=0`` = use all cores.
    prior_trajectories : list[list[Step]], optional
        Previously-committed agent trips from substructures built in
        parallel with this one (§IV-C of the decomposition paper).  Their
        (x,y,t) occupancy is injected as vertex- and edge-collision
        constraints (PAR-V, PAR-E).  Defaults to none.
    max_agents_at_t : dict[t, int], optional
        Per-t override of constraint (11)'s agent cap.  Effective cap at
        each t is ``min(num_agents, max_agents_at_t.get(t, num_agents))``.
        Use ``agent_cap_from_priors`` to derive from prior trips and the
        global fleet size.  Defaults to none (always cap at A).

    Returns
    -------
    dict with keys:
        model           : gurobipy.Model
        R_vars          : dict {action-9-tuple → BinaryVar}
        H_vars          : dict {(t,x,y,z,z')   → BinaryVar}
        X, Y, z_max, T, A : sizes
        init_hm, target_hm : echoed
        R_idx, H_idx    : wildcard-lookup indices
    """
    init_hm = np.asarray(init_hm, dtype=int)
    target_hm = np.asarray(target_hm, dtype=int)
    Y, X = init_hm.shape
    assert target_hm.shape == (Y, X), 'heightmap shapes must match'
    assert T >= 4, f'T={T} too small; need T ≥ 4'
    z_max = int(max(int(init_hm.max()), int(target_hm.max())))
    A = int(num_agents)

    # Border/interior for repeated use.
    border = border_cells(X, Y)
    interior = interior_cells(X, Y)
    all_pos = [(x, y) for y in range(Y) for x in range(X)]

    # Disallow blocks on border cells (paper requires border = 0 always).
    for (bx, by) in border:
        assert target_hm[by, bx] == 0, (
            f'target_hm has a block at border cell ({bx},{by}); '
            f'the paper disallows blocks on border positions.')
        assert init_hm[by, bx] == 0, (
            f'init_hm has a block at border cell ({bx},{by}); '
            f'the paper disallows blocks on border positions.')

    # -----------------------------------------------------------------
    # Enumerate actions and transitions.
    # -----------------------------------------------------------------
    R_list, H_list = enumerate_actions(X, Y, z_max, T)

    # -----------------------------------------------------------------
    # Build the Gurobi model.
    # -----------------------------------------------------------------
    if env is not None:
        m = gp.Model(name, env=env)
    else:
        m = gp.Model(name)
    m.Params.OutputFlag = output_flag
    m.Params.MIPGap = mip_gap
    m.Params.TimeLimit = time_limit
    if threads > 0:
        m.Params.Threads = threads
    if presolve != -1:
        m.Params.Presolve = presolve

    R_vars = {i: m.addVar(vtype=GRB.BINARY, name=_r_name(i)) for i in R_list}
    H_vars = {i: m.addVar(vtype=GRB.BINARY, name=_h_name(i)) for i in H_list}
    m.update()

    R_idx = _build_R_indices(R_list)
    H_idx = _build_H_indices(H_list)

    # -----------------------------------------------------------------
    # (1) Objective: min Σ r_i over actions with (x,y,z) ≠ (S,S,S).
    #     (Entries R_1 have source = (S,S,S) and are excluded.)
    # -----------------------------------------------------------------
    m.setObjective(
        gp.quicksum(R_vars[i] for i in R_list if i[1] != S_SENT),
        GRB.MINIMIZE,
    )

    # -----------------------------------------------------------------
    # (2) Border pillars stay at height 0:
    #     h_{t,x,y,z,z} = 1  ∀ t∈{0..T-3}, (x,y,z) ∈ B.
    #     Since init_hm[border] == 0 we only need z=0.
    # -----------------------------------------------------------------
    for t in range(T - 2):  # {0,...,T-3}
        for (x, y) in border:
            k = (t, x, y, 0, 0)
            if k in H_vars:
                m.addConstr(H_vars[k] == 1, name=f'c2_{t}_{x}_{y}')

    # -----------------------------------------------------------------
    # (3) Initial pillar heights:
    #     paper: h_{0,x,y,0,0}=1 for all (x,y) ∈ P.
    #     Generalized for non-empty init: h_{0,x,y,h0,h0}=1 where
    #     h0 = init_hm[y,x].  (When init_hm == 0 this matches the paper.)
    # -----------------------------------------------------------------
    for (x, y) in all_pos:
        h0 = int(init_hm[y, x])
        k = (0, x, y, h0, h0)
        assert k in H_vars, f'missing H var for init {k}'
        m.addConstr(H_vars[k] == 1, name=f'c3_{x}_{y}')

    # -----------------------------------------------------------------
    # (4) Target reached by T-2 and held through T-1:
    #     h_{T-2, x, y, z̄, z̄} = 1  ∀ (x,y) ∈ P.
    # -----------------------------------------------------------------
    for (x, y) in all_pos:
        zt = int(target_hm[y, x])
        k = (T - 2, x, y, zt, zt)
        assert k in H_vars, f'missing H var for target {k}'
        m.addConstr(H_vars[k] == 1, name=f'c4_{x}_{y}')

    # -----------------------------------------------------------------
    # (5) Height-flow conservation through time, for interior cells:
    #     Σ_{h ∈ H_{t,x,y,*,z}} h_i = Σ_{h ∈ H_{t+1,x,y,z,*}} h_i
    #     ∀ t∈{0..T-3}, (x,y,z) ∈ C.
    # -----------------------------------------------------------------
    for t in range(T - 2):
        for (x, y) in interior:
            for z in range(z_max + 1):
                lhs_h = H_idx['by_col_to'].get((t, x, y, z), [])
                rhs_h = H_idx['by_col_from'].get((t + 1, x, y, z), [])
                m.addConstr(
                    gp.quicksum(H_vars[i] for i in lhs_h)
                    == gp.quicksum(H_vars[i] for i in rhs_h),
                    name=f'c5_{t}_{x}_{y}_{z}',
                )

    # -----------------------------------------------------------------
    # (6) Exactly one height transition active per cell per time:
    #     Σ_{h ∈ H_{t,x,y,*,*}} h_i = 1  ∀ t∈{0..T-2}, (x,y) ∈ P.
    # -----------------------------------------------------------------
    for t in range(T - 1):
        for (x, y) in all_pos:
            lst = H_idx['by_col'].get((t, x, y), [])
            m.addConstr(
                gp.quicksum(H_vars[i] for i in lst) == 1,
                name=f'c6_{t}_{x}_{y}',
            )

    # -----------------------------------------------------------------
    # (7) Empty-robot flow at (x,y,z), t∈{0..T-3}:
    #     Σ (empty moves into (x,y,z) at time t)
    #   + Σ (loaded-then-delivered at (x,y,z) at time t)
    #     = Σ (empty moves out of (x,y,z) at time t+1)
    #   + Σ (pickups by empty robot at (x,y,z) at time t+1)
    #
    # Applied at all robot-reachable cells (interior + border z=0) so
    # entries at border are forced to be followed by outgoing actions.
    # -----------------------------------------------------------------
    valid_cells = _valid_robot_cells(X, Y, z_max)
    for t in range(T - 2):
        for (x, y, z) in valid_cells:
            # Σ r_i @ R_{t,*,*,*,0,M,x,y,z}
            lhs_a = R_idx['by_end_cad'].get((t, 0, M, x, y, z), [])
            # Σ r_i @ R_{t,x,y,z,1,D,*,*,*}
            lhs_b = R_idx['by_src_ca'].get((t, x, y, z, 1, D), [])
            # Σ r_i @ R_{t+1,x,y,z,0,M,*,*,*}
            rhs_a = R_idx['by_src_ca'].get((t + 1, x, y, z, 0, M), [])
            # Σ r_i @ R_{t+1,x,y,z,0,P,*,*,*}
            rhs_b = R_idx['by_src_ca'].get((t + 1, x, y, z, 0, P), [])
            m.addConstr(
                gp.quicksum(R_vars[i] for i in lhs_a)
                + gp.quicksum(R_vars[i] for i in lhs_b)
                ==
                gp.quicksum(R_vars[i] for i in rhs_a)
                + gp.quicksum(R_vars[i] for i in rhs_b),
                name=f'c7_{t}_{x}_{y}_{z}',
            )

    # -----------------------------------------------------------------
    # (8) Loaded-robot flow at (x,y,z), t∈{0..T-3}:
    #     Σ (loaded moves into (x,y,z) at time t)
    #   + Σ (empty-then-picked-up at (x,y,z) at time t)
    #     = Σ (loaded moves out of (x,y,z) at time t+1)
    #   + Σ (deliveries by loaded robot at (x,y,z) at time t+1)
    #
    # See note on (7): applied at interior + border z=0.
    # -----------------------------------------------------------------
    for t in range(T - 2):
        for (x, y, z) in valid_cells:
            lhs_a = R_idx['by_end_cad'].get((t, 1, M, x, y, z), [])
            lhs_b = R_idx['by_src_ca'].get((t, x, y, z, 0, P), [])
            rhs_a = R_idx['by_src_ca'].get((t + 1, x, y, z, 1, M), [])
            rhs_b = R_idx['by_src_ca'].get((t + 1, x, y, z, 1, D), [])
            m.addConstr(
                gp.quicksum(R_vars[i] for i in lhs_a)
                + gp.quicksum(R_vars[i] for i in lhs_b)
                ==
                gp.quicksum(R_vars[i] for i in rhs_a)
                + gp.quicksum(R_vars[i] for i in rhs_b),
                name=f'c8_{t}_{x}_{y}_{z}',
            )

    # -----------------------------------------------------------------
    # (9) Vertex collision:
    #     Σ r_i @ R_{t,x,y,*,*,*,*,*,*}
    #   + Σ r_i @ R_{t,*,*,*,*,P,x,y,*}
    #   + Σ r_i @ R_{t,*,*,*,*,D,x,y,*}
    #     ≤ 1    ∀ t∈{1..T-2}, (x,y) ∈ P.
    # -----------------------------------------------------------------
    for t in range(1, T - 1):
        for (x, y) in all_pos:
            a = R_idx['by_src_xy'].get((t, x, y), [])
            b = R_idx['by_pd_target_xy'].get((t, P, x, y), [])
            c = R_idx['by_pd_target_xy'].get((t, D, x, y), [])
            m.addConstr(
                gp.quicksum(R_vars[i] for i in a)
                + gp.quicksum(R_vars[i] for i in b)
                + gp.quicksum(R_vars[i] for i in c)
                <= 1,
                name=f'c9_{t}_{x}_{y}',
            )

    # -----------------------------------------------------------------
    # (10) Edge collision (swap):
    #      Σ r_i @ R_{t,x,y,*,*,M,x',y',*}
    #    + Σ r_i @ R_{t,x',y',*,*,M,x,y,*}
    #      ≤ 1   ∀ t∈{1..T-2}, (x,y) ∈ P, (x',y') ∈ N(x,y).
    # -----------------------------------------------------------------
    for t in range(1, T - 1):
        for (x, y) in all_pos:
            for (xp, yp) in _neighbors(x, y, X, Y):
                if (x, y) >= (xp, yp):
                    continue  # each unordered pair once
                fwd = R_idx['by_edge_xy'].get((t, x, y, xp, yp), [])
                bwd = R_idx['by_edge_xy'].get((t, xp, yp, x, y), [])
                m.addConstr(
                    gp.quicksum(R_vars[i] for i in fwd)
                    + gp.quicksum(R_vars[i] for i in bwd)
                    <= 1,
                    name=f'c10_{t}_{x}_{y}_{xp}_{yp}',
                )

    # -----------------------------------------------------------------
    # (11) Agent cap:  Σ r_i @ R_{t,*,*,*,*,*,*,*,*} ≤ A  ∀ t ∈ T.
    #      This includes entries (source = S,S,S) — so at any time, the
    #      *number of agents currently in transit or on-grid* ≤ A.
    #
    #      ``max_agents_at_t[t]`` (§IV-C parallel-construction) tightens
    #      the cap on a per-t basis; the effective cap is the min.
    # -----------------------------------------------------------------
    effective_cap_at_t = {}
    for t in range(T):
        cap_t = A
        if max_agents_at_t is not None and t in max_agents_at_t:
            cap_t = min(cap_t, int(max_agents_at_t[t]))
        effective_cap_at_t[t] = max(0, cap_t)

    for t in range(T):
        lst = R_idx['by_time'].get(t, [])
        if not lst:
            continue
        m.addConstr(
            gp.quicksum(R_vars[i] for i in lst) <= effective_cap_at_t[t],
            name=f'c11_{t}',
        )

    # -----------------------------------------------------------------
    # (12) Robot-pillar coupling:
    #      Σ_{h ∈ H_{t,x,y,*,z}} h_i ≥ Σ_{r ∈ R_{t,x,y,z,...}} r_i
    #      ∀ t∈{0..T-2}, (x,y,z) ∈ C.
    #
    # Posted at interior × {0..z_max}.  At border z=0 the coupling is
    # trivially satisfied by (2) + (9), so it is omitted to keep the
    # model size in check.
    # -----------------------------------------------------------------
    for t in range(T - 1):
        for (x, y) in interior:
            for z in range(z_max + 1):
                h_lhs = H_idx['by_col_to'].get((t, x, y, z), [])
                r_rhs = []
                for c in (0, 1):
                    for a in (M, P, D):
                        r_rhs.extend(
                            R_idx['by_src_ca'].get((t, x, y, z, c, a), []))
                if not r_rhs:
                    continue
                m.addConstr(
                    gp.quicksum(H_vars[i] for i in h_lhs)
                    >= gp.quicksum(R_vars[i] for i in r_rhs),
                    name=f'c12_{t}_{x}_{y}_{z}',
                )

    # -----------------------------------------------------------------
    # (13) Pickup ↔ height decrement:
    #      h_{t,x,y,z+1,z} = Σ_{i ∈ R_{t,*,*,*,0,P,x,y,z}} r_i
    #      ∀ t∈{0..T-2}, (x,y) ∈ P, z∈{0..z_max-1}.
    # -----------------------------------------------------------------
    for t in range(T - 1):
        for (x, y) in all_pos:
            for z in range(z_max):
                h_k = (t, x, y, z + 1, z)
                assert h_k in H_vars, f'missing h for pickup {h_k}'
                lst = R_idx['by_pickup_target_xyz'].get((t, x, y, z), [])
                m.addConstr(
                    H_vars[h_k]
                    == gp.quicksum(R_vars[i] for i in lst),
                    name=f'c13_{t}_{x}_{y}_{z}',
                )

    # -----------------------------------------------------------------
    # (14) Deliver ↔ height increment:
    #      h_{t,x,y,z,z+1} = Σ_{i ∈ R_{t,*,*,*,1,D,x,y,z}} r_i
    #      ∀ t∈{0..T-2}, (x,y) ∈ P, z∈{0..z_max-1}.
    # -----------------------------------------------------------------
    for t in range(T - 1):
        for (x, y) in all_pos:
            for z in range(z_max):
                h_k = (t, x, y, z, z + 1)
                assert h_k in H_vars, f'missing h for deliver {h_k}'
                lst = R_idx['by_deliver_target_xyz'].get((t, x, y, z), [])
                m.addConstr(
                    H_vars[h_k]
                    == gp.quicksum(R_vars[i] for i in lst),
                    name=f'c14_{t}_{x}_{y}_{z}',
                )

    # (15), (16) are integrality of H_vars / R_vars — already declared BINARY.

    # -----------------------------------------------------------------
    # (PAR-V) Parallel-substructure vertex blocking (§IV-C).
    #     For each (t, x, y) occupied by a previously-committed trip,
    #     forbid any new agent from being at (x, y) at time t.  Equivalent
    #     to zeroing the LHS of (9) on those (t,x,y) cells.
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # (PAR-E) Parallel-substructure edge blocking (§IV-C).
    #     For each directed (x1,y1)→(x2,y2) move by a prior agent at
    #     action time t, forbid the reverse (x2,y2)→(x1,y1) move at the
    #     same time.  Equivalent to zeroing one side of (10).
    # -----------------------------------------------------------------
    prior_vertex_xy, prior_edges, _ = _derive_prior_state(prior_trajectories)
    for t, xys in prior_vertex_xy.items():
        if t < 1 or t > T - 2:
            continue
        for (xb, yb) in xys:
            src_actions = R_idx['by_src_xy'].get((t, xb, yb), [])
            pickup_acts = R_idx['by_pd_target_xy'].get((t, P, xb, yb), [])
            deliver_acts = R_idx['by_pd_target_xy'].get((t, D, xb, yb), [])
            blocked = src_actions + pickup_acts + deliver_acts
            if not blocked:
                continue
            m.addConstr(
                gp.quicksum(R_vars[i] for i in blocked) == 0,
                name=f'cPARV_{t}_{xb}_{yb}',
            )

    for t, edge_set in prior_edges.items():
        if t < 1 or t > T - 2:
            continue
        for (x1, y1, x2, y2) in edge_set:
            rev_moves = R_idx['by_edge_xy'].get((t, x2, y2, x1, y1), [])
            if not rev_moves:
                continue
            m.addConstr(
                gp.quicksum(R_vars[i] for i in rev_moves) == 0,
                name=f'cPARE_{t}_{x1}_{y1}_{x2}_{y2}',
            )

    m.update()

    return {
        'model': m,
        'R_vars': R_vars,
        'H_vars': H_vars,
        'X': X, 'Y': Y, 'z_max': z_max, 'T': T, 'A': A,
        'init_hm': init_hm, 'target_hm': target_hm,
        'R_idx': R_idx, 'H_idx': H_idx,
    }


# ---------------------------------------------------------------------------
# Var-name helpers (short names keep .lp files readable)
# ---------------------------------------------------------------------------

def _r_name(i):
    t, x, y, z, c, a, xp, yp, zp = i
    return f'r_{t}_{x}_{y}_{z}_{c}{a}_{xp}_{yp}_{zp}'


def _h_name(i):
    t, x, y, z, zp = i
    return f'h_{t}_{x}_{y}_{z}_{zp}'
