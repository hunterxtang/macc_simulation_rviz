"""Decode a solved MACC MILP into per-agent ``Step`` trajectories.

The Lam et al. 2020 MILP is a flow-based formulation — the solver tells
us *which* actions occur but not *which agent* performs each one.  To
produce per-agent plans we walk the flow: pair each entry (R_1) with a
compatible outgoing action in the next timestep, and keep following the
agent until it exits via R_4.

Output schema matches ``macc_rviz.cbs_planner.Step`` so downstream code
(simulator replay loop, RViz animator) does not need special casing.
"""

from macc_rviz.cbs_planner import (
    ACTION_MOVE, ACTION_PICKUP, ACTION_PLACE, ACTION_WAIT, Step,
)
from macc_rviz.planners.milp_encoding import (
    D, E_SENT, M, P, S_SENT,
)


def _active_actions(R_vars, tol=1e-5):
    """Return list of 9-tuples for r_i whose LP value ≈ 1."""
    return [i for i, v in R_vars.items() if v.X > 1.0 - tol]


def _si_at(target_sub, x, y, z):
    """Look up substructure index at voxel (x,y,z); return -1 if unknown."""
    if target_sub is None:
        return -1
    try:
        si = int(target_sub[z, y, x])
    except (IndexError, ValueError):
        return -1
    return si


def extract_agent_trips(R_vars, T, tol=1e-5):
    """Back-compat wrapper: return trips only (Phase 3 API)."""
    trips, _ = extract_agent_trips_with_events(
        R_vars, T, target_sub=None, tol=tol,
    )
    return trips


def extract_agent_trips_with_events(R_vars, T, target_sub=None, tol=1e-5):
    """Partition active actions into per-agent trips + paired events.

    A *trip* is one agent's complete sojourn on the grid: enter at
    R_1 → sequence of R_2/R_3/R_5/R_6 actions → exit at R_4.  Each
    trip is emitted as its own per-agent plan so a scheduler can
    assign trips to physical robots.

    Per-trip events mirror the CBS event schema so a single replay
    path in ``macc_rviz_sim`` can consume either planner's output:

        ('pickup', bx, by, bz, si)
                                — robot now carries a block tagged si.
                                  (bx, by, bz) is the source voxel for
                                  R_5 grid pickups (replay clears it);
                                  (-1, -1, -1) for off-grid carries
                                  (entry-carrying / depot supply).
        ('place', bx, by, bz, si)
                                — block placed at voxel (bx, by, bz);
                                  world/world_sub are updated here

    Synthetic pickups are emitted at the first Step of trips that
    enter with c=1 (agent walked on carrying); these use the off-grid
    sentinel since no grid voxel sourced the block.  The ``si`` tag
    looks ahead to the trip's first R_6 delivery; if none, si=-1
    (scaffold).

    Parameters
    ----------
    R_vars : dict {9-tuple: BinaryVar}
        Solved action variables from ``build_model``.
    T : int
        Planning horizon used at build time.
    target_sub : ndarray (Z, Y, X), optional
        Voxel-level substructure tags (non-negative int = si,
        negative = unassigned).  Used to colour pickup/place events;
        ``None`` yields si=-1 everywhere.
    tol : float
        LP rounding tolerance for "variable is active".

    Returns
    -------
    trips  : list[list[Step]]            — per-trip on-grid step list
    events : list[dict[int, tuple]]      — per-trip event dict keyed by
                                            absolute-t (same t as Step.t)
    """
    active = _active_actions(R_vars, tol=tol)

    # Group actions by (t, source xyz) for forward walk, plus indexed by t.
    by_time = {}
    for i in active:
        by_time.setdefault(i[0], []).append(i)

    # Group entries (R_1) by time — these start new trips.
    entries = [i for i in active if i[1] == S_SENT]

    used = set()  # action-tuple ids consumed by some trip
    trips = []
    events = []

    for entry in sorted(entries, key=lambda e: (e[0], e[6], e[7])):
        if id(entry) in used:
            continue
        trip, evs = _walk_trip(entry, by_time, used, T, target_sub, tol=tol)
        if trip:
            trips.append(trip)
            events.append(evs)

    return trips, events


def _walk_trip(entry, by_time, used, T, target_sub, tol=1e-5):
    """Follow one agent from entry through to exit, collecting Steps + events.

    Returns (steps, events) where events is {absolute_t: event_tuple}.
    """
    steps = []
    evs = {}
    t_e, _, _, _, c_e, _, xp, yp, zp = entry
    used.add(id(entry))

    # Entry action puts agent at (xp, yp, zp) at time t_e + 1.  First
    # on-grid step is always WAIT; for carrying-entry trips, a synthetic
    # ('pickup', si) event is attached at the same tick so the replay
    # turns on the carry flag without inflating ACTION_PICKUP counts.
    steps.append(Step(ACTION_WAIT, xp, yp, zp, t_e + 1))

    t = t_e + 1
    x, y, z = xp, yp, zp
    c = c_e

    # First pass: walk the trip, deferring the entry-carrying si lookup
    # until we've seen the first R_6 (if any).
    deferred_entry_si = None

    while t <= T - 2:
        candidates = by_time.get(t, [])
        chosen = None
        for i in candidates:
            if id(i) in used:
                continue
            if i[1] == S_SENT:
                continue  # an entry, not ours
            if (i[1], i[2], i[3], i[4]) == (x, y, z, c):
                chosen = i
                break
        if chosen is None:
            break

        used.add(id(chosen))
        _, sx, sy, sz, sc, a, dx_, dy_, dz_ = chosen

        if chosen[6] == E_SENT:
            # R_4 exit — agent leaves the grid at time t+1.  No Step
            # is recorded for the off-grid position.
            break

        if a == M:
            if (dx_, dy_, dz_) == (sx, sy, sz):
                act = ACTION_WAIT
            else:
                act = ACTION_MOVE
            x, y, z = dx_, dy_, dz_
            steps.append(Step(act, x, y, z, t + 1))
        elif a == P:
            # R_5: robot at (sx,sy,sz) picks up the top block of pillar
            # (dx_, dy_) — that block occupies voxel (dx_, dy_, dz_).
            act = ACTION_PICKUP
            x, y, z = sx, sy, sz
            c = 1
            si_pick = _si_at(target_sub, dx_, dy_, dz_)
            evs[t + 1] = ('pickup', dx_, dy_, dz_, si_pick)
            steps.append(Step(act, x, y, z, t + 1))
        elif a == D:
            # R_6: robot at (sx,sy,sz) delivers a block to pillar
            # (dx_, dy_); the new block occupies voxel (dx_, dy_, dz_).
            act = ACTION_PLACE
            x, y, z = sx, sy, sz
            c = 0
            si_place = _si_at(target_sub, dx_, dy_, dz_)
            evs[t + 1] = ('place', dx_, dy_, dz_, si_place)
            if deferred_entry_si is None and c_e == 1:
                deferred_entry_si = si_place
            steps.append(Step(act, x, y, z, t + 1))
        else:
            break

        t += 1

    # Finalize the synthetic entry pickup's si tag.  Source coords are
    # the off-grid sentinel because the block came in from outside the
    # grid (R_1 carrying-entry), not from any voxel pillar.
    if c_e == 1:
        si_entry = deferred_entry_si if deferred_entry_si is not None else -1
        evs[t_e + 1] = ('pickup', -1, -1, -1, si_entry)

    return steps, evs
