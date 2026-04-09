import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def set_axes_equal(ax):
    """Forces equal scaling on 3D axes so voxels appear cubic."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range) / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])


def _cube_faces(cx, cy, cz, size=1.0):
    """Return the 6 faces of an axis-aligned cube as a list of 4-vertex polygons."""
    s = size / 2.0
    corners = np.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ])
    face_indices = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    return [corners[fi] for fi in face_indices]


def _draw_cube(ax, cx, cy, cz, color, alpha=0.85, size=1.0):
    faces = _cube_faces(cx, cy, cz, size)
    poly = Poly3DCollection(faces, facecolor=color, edgecolor='k',
                            linewidth=0.4, alpha=alpha)
    ax.add_collection3d(poly)


# ---------------------------------------------------------------------------
# Static visualisations (kept for compatibility)
# ---------------------------------------------------------------------------

def show_structure_voxels(structure, title="Structure"):
    """Displays the full 3D structure using solid cubic voxels."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(structure, facecolors='skyblue', edgecolor='k', alpha=0.9)
    ax.set_title(title)
    ax.view_init(elev=25, azim=45)
    set_axes_equal(ax)
    plt.show()


def show_substructures(substructures):
    """Displays each decomposed substructure in a distinct color."""
    colors = ['red', 'green', 'blue', 'orange', 'purple',
              'cyan', 'magenta', 'yellow', 'lime', 'navy']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, s in enumerate(substructures):
        color = colors[i % len(colors)]
        ax.voxels(s, facecolors=color, edgecolor='k', alpha=0.9)
    ax.set_title("Substructures (Decomposed)")
    ax.view_init(elev=25, azim=45)
    set_axes_equal(ax)
    plt.show()


# ---------------------------------------------------------------------------
# Robot simulation state
# ---------------------------------------------------------------------------

ROBOT_COLORS = [
    '#e74c3c',  # red
    '#2ecc71',  # green
    '#3498db',  # blue
    '#f39c12',  # orange
    '#9b59b6',  # purple
    '#1abc9c',  # teal
]
CARRIED_COLOR = '#f5f5f5'   # near-white for carried block
PLACED_COLOR = '#7fb3d3'    # steel-blue for placed structure blocks
GHOST_COLOR = '#d5e8d4'     # pale green ghost of unbuilt target


class _Robot:
    def __init__(self, rid, color):
        self.id = rid
        self.color = color
        self.x = 0
        self.y = 0
        self.z = 0          # surface z (top of whatever stack robot stands on)
        self.carrying = False


def _heightmap(world):
    return np.sum(world, axis=0).astype(int)  # shape (Y, X)


def _surface_z(hm, x, y):
    return int(hm[y, x])


def _neighbors4(x, y):
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _in_bounds(x, y, X, Y):
    return 0 <= x < X and 0 <= y < Y


def _bfs_path(start, goal, hm):
    """BFS on the heightmap allowing ±1 height steps."""
    Y, X = hm.shape
    sx, sy = start
    gx, gy = goal
    if start == goal:
        return [start]
    q = [(sx, sy)]
    prev = {(sx, sy): None}
    while q:
        x, y = q.pop(0)
        if (x, y) == (gx, gy):
            break
        for nx, ny in _neighbors4(x, y):
            if not _in_bounds(nx, ny, X, Y):
                continue
            if (nx, ny) in prev:
                continue
            if abs(int(hm[ny, nx]) - int(hm[y, x])) <= 1:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    if (gx, gy) not in prev:
        return None
    path, cur = [], (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _blocks_bottom_up(sub):
    """All occupied voxels in sub, sorted bottom-to-top."""
    coords = list(zip(*np.where(sub == 1)))  # (z, y, x) tuples
    coords.sort(key=lambda t: (t[0], t[1], t[2]))
    return [(x, y, z) for (z, y, x) in coords]


# ---------------------------------------------------------------------------
# Animated construction simulation
# ---------------------------------------------------------------------------

def animate_build(structure, substructures, groups, delay=0.06, num_robots=3):
    """
    Runs a step-by-step multi-robot construction simulation.

    Robots are drawn as colored cubes.  When a robot carries a block the
    carried block is drawn stacked on top of the robot cube so it is visually
    obvious which robot holds what.

    Actions per tick mirror the paper primitives:
      move N/S/E/W, climb up one block, descend one block, pick-up, place,
      wait.  Each action advances the animation by one frame.
    """
    H, Y, X = structure.shape

    # Build a flat task list ordered by parallel groups, bottom-up within each
    plan = []
    for group in groups:
        for si in group:
            for xyz in _blocks_bottom_up(substructures[si]):
                plan.append(xyz)

    world = np.zeros_like(structure, dtype=int)

    colors = [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(num_robots)]
    robots = [_Robot(i, colors[i]) for i in range(num_robots)]

    task_queue = list(plan)
    active = {}   # robot id -> state dict

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=28, azim=40)
    plt.ion()
    plt.show()

    step = 0

    def can_place(x, y, z):
        if world[z, y, x] == 1:
            return False
        return z == 0 or world[z - 1, y, x] == 1

    def assign_tasks():
        for r in robots:
            if r.id in active or not task_queue:
                continue
            active[r.id] = {
                "phase": "to_pickup",
                "task": task_queue.pop(0),
                "path": None,
                "path_i": 0,
            }

    def _redraw(title_suffix=""):
        ax.cla()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"MACC Construction  (step {step}){title_suffix}")
        ax.view_init(elev=28, azim=40)

        # Ghost: unbuilt target blocks
        for z in range(H):
            for y in range(Y):
                for x in range(X):
                    if structure[z, y, x] and not world[z, y, x]:
                        _draw_cube(ax, x + 0.5, y + 0.5, z + 0.5,
                                   GHOST_COLOR, alpha=0.18)

        # Placed blocks
        for z in range(H):
            for y in range(Y):
                for x in range(X):
                    if world[z, y, x]:
                        _draw_cube(ax, x + 0.5, y + 0.5, z + 0.5,
                                   PLACED_COLOR, alpha=0.85)

        # Robots (and carried blocks)
        for r in robots:
            rz = r.z + 0.5   # robot cube centre sits one unit above surface
            _draw_cube(ax, r.x + 0.5, r.y + 0.5, rz + 0.5,
                       r.color, alpha=1.0, size=0.7)
            if r.carrying:
                _draw_cube(ax, r.x + 0.5, r.y + 0.5, rz + 1.3,
                           CARRIED_COLOR, alpha=1.0, size=0.55)

        # Legend
        handles = [mpatches.Patch(color=PLACED_COLOR, label='placed'),
                   mpatches.Patch(color=GHOST_COLOR, label='target (unbuilt)')]
        for r in robots:
            handles.append(mpatches.Patch(color=r.color,
                                          label=f'robot {r.id}'))
        ax.legend(handles=handles, loc='upper left', fontsize=7)

        # Keep axes sensibly sized
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_zlim(0, H + 1)
        plt.draw()
        plt.pause(delay)

    def robot_step(r, hm, path_state):
        """Advance robot one step along its stored path. Returns True when done."""
        path = path_state["path"]
        i = path_state["path_i"]
        if path is None or i >= len(path):
            return True
        nx, ny = path[i]
        # climb / descend / lateral move — all count as one action
        r.x, r.y = nx, ny
        r.z = _surface_z(hm, nx, ny)
        path_state["path_i"] = i + 1
        return path_state["path_i"] >= len(path)

    while True:
        hm = _heightmap(world)
        assign_tasks()

        all_idle = all(r.id not in active for r in robots)
        if all_idle and not task_queue:
            break

        for r in robots:
            if r.id not in active:
                r.z = _surface_z(hm, r.x, r.y)
                continue

            state = active[r.id]
            x, y, z = state["task"]

            if state["phase"] == "to_pickup":
                if (r.x, r.y) != (0, 0):
                    if state["path"] is None:
                        p = _bfs_path((r.x, r.y), (0, 0), hm)
                        state["path"] = p if p is not None else [(r.x, r.y)]
                        state["path_i"] = 0
                    done = robot_step(r, hm, state)
                    if done:
                        state["phase"] = "pickup"
                        state["path"] = None
                else:
                    state["phase"] = "pickup"

            elif state["phase"] == "pickup":
                # pick-up action (one tick)
                r.carrying = True
                state["phase"] = "to_place"

            elif state["phase"] == "to_place":
                if not can_place(x, y, z):
                    # block not yet placeable — requeue and drop
                    task_queue.append(state["task"])
                    r.carrying = False
                    del active[r.id]
                    continue

                # move to a neighbour adjacent to target
                candidates = [
                    (nx, ny)
                    for nx, ny in _neighbors4(x, y)
                    if _in_bounds(nx, ny, X, Y)
                    and abs(int(hm[ny, nx]) - z) <= 1
                ]
                if not candidates:
                    task_queue.append(state["task"])
                    r.carrying = False
                    del active[r.id]
                    continue

                goal = min(candidates,
                           key=lambda c: abs(c[0] - r.x) + abs(c[1] - r.y))

                if (r.x, r.y) != goal:
                    if state["path"] is None:
                        p = _bfs_path((r.x, r.y), goal, hm)
                        state["path"] = p if p is not None else [(r.x, r.y)]
                        state["path_i"] = 0
                    done = robot_step(r, hm, state)
                    if done:
                        state["phase"] = "place"
                        state["path"] = None
                else:
                    state["phase"] = "place"

            elif state["phase"] == "place":
                # place action (one tick)
                if can_place(x, y, z):
                    world[z, y, x] = 1
                r.carrying = False
                del active[r.id]

        step += 1
        _redraw()

    _redraw(" — COMPLETE")
    print("\n=== BUILD COMPLETE ===")
    plt.ioff()
    plt.show()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def show_construction_process(structure, substructures, groups,
                               num_robots=3, delay=0.06):
    """
    Shows the target structure, the decomposition, then runs the animated
    robot simulation.
    """
    show_structure_voxels(structure, "Target Structure")
    show_substructures(substructures)
    animate_build(structure, substructures, groups,
                  delay=delay, num_robots=num_robots)
