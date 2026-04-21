import hashlib
import random
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as BuiltinDuration

from macc_rviz.structure_utils import create_random_structure, create_example_structure
from macc_rviz.decomposition import decompose_structure, order_substructures
from macc_rviz.parallel import find_parallel_groups
from macc_rviz.planners import cbs_adapter

# ---------------------------------------------------------------------------
# Playback / timing constants — tune here; step_duration_sec ROS parameter
# overrides STEP_DURATION_SEC at launch (seconds per action, no Hz divisor).
# ---------------------------------------------------------------------------
STEP_DURATION_SEC = 0.4         # seconds per simulation tick (each move/pickup/place)
INTER_GROUP_PAUSE_SEC = 1.5     # pause between parallel groups (highlight + hold)
STAGE_PAUSE_SEC = 3.0           # hold time at the end of each intro stage
SUBSTRUCTURE_REVEAL_SEC = 0.4   # delay between successive reveals in Stage 2

# ---------------------------------------------------------------------------
# Stage identifiers
# ---------------------------------------------------------------------------
_STAGE_INPUT = 1   # Stage 1: full structure, uniform color
_STAGE_DECOMP = 2  # Stage 2: decomposition coloring, progressive reveal
_STAGE_ORDER = 3   # Stage 3: build-order labels added
_STAGE_BUILD = 4   # Stage 4: actual construction
_STAGE_RETURN = 5  # Stage 5: robots return to boundary after build

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
# tab10 colors — one per substructure, always in the same order
_SUB_COLORS = [
    (0.122, 0.467, 0.706),  # blue
    (1.000, 0.498, 0.055),  # orange
    (0.173, 0.627, 0.173),  # green
    (0.839, 0.153, 0.157),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.812),  # cyan
]

# Uniform robot color — pure white, never collides with tab10 colormap entries
_ROBOT_COLOR = (1.0, 1.0, 1.0)   # pure white

# Stage 1 uniform structure color — soft blue (distinct from decomp palette)
_INPUT_COLOR = (0.4, 0.65, 0.9)


def _sub_color(si):
    return _SUB_COLORS[si % len(_SUB_COLORS)]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def heightmap(world):
    return np.sum(world, axis=0).astype(int)   # shape (Y, X)


def in_bounds(x, y, X, Y):
    return 0 <= x < X and 0 <= y < Y


def neighbors4(x, y):
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def bfs_path(start, goal, hm):
    Y, X = hm.shape
    sx, sy = start
    gx, gy = goal
    q = [(sx, sy)]
    prev = {(sx, sy): None}
    while q:
        x, y = q.pop(0)
        if (x, y) == (gx, gy):
            break
        for nx, ny in neighbors4(x, y):
            if not in_bounds(nx, ny, X, Y):
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


def bfs_to_boundary(start, hm):
    """Multi-goal BFS: shortest path from start to any grid boundary cell.

    A boundary cell is one where x==0, x==X-1, y==0, or y==Y-1.
    Returns the path as a list of (x, y) tuples (inclusive of start and goal),
    or None if no path exists.
    """
    Y, X = hm.shape
    sx, sy = start
    if sx == 0 or sx == X - 1 or sy == 0 or sy == Y - 1:
        return [start]   # already on boundary
    q = [(sx, sy)]
    prev = {(sx, sy): None}
    while q:
        x, y = q.pop(0)
        if x == 0 or x == X - 1 or y == 0 or y == Y - 1:
            path, cur = [], (x, y)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nx, ny in neighbors4(x, y):
            if not in_bounds(nx, ny, X, Y):
                continue
            if (nx, ny) in prev:
                continue
            if abs(int(hm[ny, nx]) - int(hm[y, x])) <= 1:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None


def surface_z(hm, x, y):
    return int(hm[y, x])


def blocks_in_substructure_bottom_up(sub):
    coords = []
    for y in range(sub.shape[1]):
        for x in range(sub.shape[2]):
            for z in sorted(np.where(sub[:, y, x] == 1)[0]):
                coords.append((x, y, int(z)))
    coords.sort(key=lambda t: (t[2], t[1], t[0]))
    return coords


# Single fixed depot — all robots pick up blocks from this boundary cell.
# (0, 0) is the grid corner; the paper says "unlimited supply at boundary"
# without prescribing which cell, so one consistent point is sufficient.
DEPOT_X, DEPOT_Y = 0, 0


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------

class Robot:
    def __init__(self, rid):
        self.id = rid
        self.x = 0
        self.y = 0
        self.z = 0
        self.carrying = False
        self.carrying_si = -1   # substructure index of block being carried
        self.wait_streak = 0    # consecutive ticks this robot has waited (deadlock detector)


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class MACCRvizSim(Node):
    def __init__(self):
        super().__init__("macc_rviz_sim")

        # --- publishers ---
        self.blocks_pub = self.create_publisher(MarkerArray, "/macc/blocks", 10)
        self.robots_pub = self.create_publisher(MarkerArray, "/macc/robots", 10)

        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ghost_pub = self.create_publisher(MarkerArray, "/macc/ghost", _latched)
        self.text_pub = self.create_publisher(MarkerArray, "/macc/subtext", _latched)

        # Intro-stage publishers (Stages 1–3)
        self.intro_pub = self.create_publisher(MarkerArray, "/macc/intro", 10)
        self.stage_label_pub = self.create_publisher(
            MarkerArray, "/macc/stage_label", _latched
        )

        # --- parameters ---
        self.declare_parameter("num_robots", 4)
        self.declare_parameter("step_duration_sec", STEP_DURATION_SEC)
        self.declare_parameter("block_scale", 1.0)
        self.declare_parameter("use_example_structure", False)
        # seed=-1 means "pick a fresh random seed each run"; any non-negative
        # value is used as-is for reproducibility.
        self.declare_parameter("seed", -1)
        # planner: "heuristic" (reactive per-tick FSM, default) or "cbs"
        # (precomputed conflict-free joint plan via cbs_planner).
        self.declare_parameter("planner", "heuristic")
        self.declare_parameter("cbs_max_t", 400)
        self.declare_parameter("cbs_branch_limit", 50)

        num_robots = int(self.get_parameter("num_robots").value)
        self.step_duration_sec = float(self.get_parameter("step_duration_sec").value)
        self.block_scale = float(self.get_parameter("block_scale").value)
        use_example = bool(self.get_parameter("use_example_structure").value)
        seed_param = int(self.get_parameter("seed").value)
        self.planner = str(self.get_parameter("planner").value).lower()
        self.cbs_max_t = int(self.get_parameter("cbs_max_t").value)
        self.cbs_branch_limit = int(self.get_parameter("cbs_branch_limit").value)
        if self.planner not in ("heuristic", "cbs"):
            self.get_logger().warning(
                f"Unknown planner '{self.planner}', falling back to 'heuristic'"
            )
            self.planner = "heuristic"

        # Resolve seed: -1 → fresh seed from nanosecond wall-clock.
        if seed_param < 0:
            seed = time.time_ns() % (2 ** 31 - 1)
        else:
            seed = seed_param

        # --- target structure ---
        if use_example:
            self.target = create_example_structure()
            seed = None   # not applicable; log will say so
        else:
            self.target = create_random_structure(x=7, y=7, z=4, density=0.4, seed=seed)
        self.H, self.Y, self.X = self.target.shape

        # --- decomposition (Algorithm 1 + Algorithm 3) ---
        self.substructures = decompose_structure(self.target)
        self.order, deps = order_substructures(self.substructures)
        self.groups = find_parallel_groups(self.substructures, deps)

        # Per-voxel substructure index (0-based); -1 for unassigned (shouldn't occur)
        self.target_sub = np.full_like(self.target, -1, dtype=int)
        for si, sub in enumerate(self.substructures):
            self.target_sub[sub == 1] = si

        # Build-order labels for Stage 3 and Stage 4 sub-text.
        # Serial groups: "1", "2", …  Parallel groups: "1a", "1b", …
        self.sub_labels = {}
        for gi, group in enumerate(self.groups):
            if len(group) == 1:
                self.sub_labels[group[0]] = str(gi + 1)
            else:
                for j, si in enumerate(group):
                    self.sub_labels[si] = f"{gi + 1}{chr(ord('a') + j)}"

        # Precompute stage-label position: centred above the structure
        bs = self.block_scale
        self._label_x = (self.X - 1) / 2.0 * bs
        self._label_y = (self.Y - 1) / 2.0 * bs
        self._label_z = (self.H + 1) * bs

        # --- world state ---
        self.world = np.zeros_like(self.target, dtype=int)
        self.world_sub = np.zeros_like(self.target, dtype=int)

        # --- task planning: one task-list per parallel group ---
        self.group_task_lists = []
        for group in self.groups:
            tasks = []
            for si in group:
                for (x, y, z) in blocks_in_substructure_bottom_up(self.substructures[si]):
                    tasks.append((x, y, z, si))
            tasks.sort(key=lambda t: t[2])   # build bottom-up within each group
            self.group_task_lists.append(tasks)

        self.current_group_idx = 0
        self.task_queue = list(self.group_task_lists[0]) if self.group_task_lists else []
        self.active = {}

        # --- pause / highlight state (Stage 4) ---
        self.pause_ticks = 0
        self.just_completed_sis = set()

        # --- robots — start spread along the -y boundary (y=0, x=0,1,2,...) ---
        self.robots = [Robot(i) for i in range(num_robots)]
        for r in self.robots:
            r.x = r.id % self.X
            r.y = 0
            r.z = 0

        # Return-to-boundary state (used after build completes)
        self.build_complete = False
        self.return_paths = {}
        self.return_done = set()        # robot IDs that have exited the grid
        self.return_start_tick = None   # tick at which return phase began (for timeout)
        self.return_delay = {}          # robot_id → stagger delay in ticks (FIX 3)

        # --- intro stage state machine ---
        self.stage = _STAGE_INPUT
        self.stage_start_time = None   # initialised lazily on first tick

        # --- simulation timer — period = step_duration_sec directly, no divisors ---
        self.timer = self.create_timer(self.step_duration_sec, self.tick)

        # --- tick counter (used for collision-avoidance log messages) ---
        self.tick_count = 0

        block_count = int(self.target.sum())
        grid_hash = hashlib.md5(self.target.tobytes()).hexdigest()[:8]
        seed_str = str(seed) if seed is not None else "n/a (example)"
        self.get_logger().info(f"Using random seed: {seed_str}")
        self.get_logger().info(
            f"Structure: {block_count} blocks, fingerprint={grid_hash} "
            f"(shape Z,Y,X={self.target.shape})"
        )
        self.get_logger().info(f"Robots: {num_robots} (priority order = robot id)")
        self.get_logger().info(f"Substructures: {len(self.substructures)}")
        self.get_logger().info(f"Parallel groups: {self.groups}")
        self.get_logger().info(f"Build order: {self.order}")
        self.get_logger().info(
            f"Step duration: {self.step_duration_sec:.2f}s "
            f"(STEP_DURATION_SEC={STEP_DURATION_SEC})"
        )
        self.get_logger().info(
            "Collision avoidance: priority-based reservation table "
            "(vertex + swap + carried-block, jiggle after 3 waits)"
        )

    # ------------------------------------------------------------------
    # Marker utilities
    # ------------------------------------------------------------------

    def _deleteall(self, pub, *namespaces):
        """Publish Marker.DELETEALL for each namespace to the given publisher."""
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        for ns in namespaces:
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = ns
            m.id = 0
            m.action = Marker.DELETEALL
            ma.markers.append(m)
        pub.publish(ma)

    # ------------------------------------------------------------------
    # Intro stage: publishing
    # ------------------------------------------------------------------

    def _publish_stage_label(self, text):
        """Publish (or update) the single main stage-label text marker."""
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = now
        m.ns = "stage_label"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = self._label_x
        m.pose.position.y = self._label_y
        m.pose.position.z = self._label_z
        m.pose.orientation.w = 1.0
        m.scale.z = bs * 0.7
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        m.lifetime = BuiltinDuration(sec=0, nanosec=0)
        ma = MarkerArray()
        ma.markers.append(m)
        self.stage_label_pub.publish(ma)

    def _publish_stage1_preview(self):
        """All target voxels as uniform soft-blue solid cubes (Stage 1)."""
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        r, g, b = _INPUT_COLOR
        ma = MarkerArray()
        for z in range(self.H):
            for y in range(self.Y):
                for x in range(self.X):
                    if not self.target[z, y, x]:
                        continue
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = "preview"
                    m.id = int(z) * self.Y * self.X + int(y) * self.X + int(x)
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(x) * bs
                    m.pose.position.y = float(y) * bs
                    m.pose.position.z = float(z) * bs
                    m.pose.orientation.w = 1.0
                    m.scale.x = bs
                    m.scale.y = bs
                    m.scale.z = bs
                    m.color.r = float(r)
                    m.color.g = float(g)
                    m.color.b = float(b)
                    m.color.a = 1.0
                    m.lifetime = BuiltinDuration(sec=0, nanosec=0)
                    ma.markers.append(m)
        self.intro_pub.publish(ma)

    def _publish_stage2_preview(self, n_revealed):
        """Voxels for Algorithm-1 substructures 0..n_revealed-1, colored per sub."""
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        ma = MarkerArray()
        for z in range(self.H):
            for y in range(self.Y):
                for x in range(self.X):
                    if not self.target[z, y, x]:
                        continue
                    si = self.target_sub[z, y, x]
                    if si < 0 or si >= n_revealed:
                        continue
                    rc, gc, bc = _sub_color(si)
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = "preview"
                    m.id = int(z) * self.Y * self.X + int(y) * self.X + int(x)
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(x) * bs
                    m.pose.position.y = float(y) * bs
                    m.pose.position.z = float(z) * bs
                    m.pose.orientation.w = 1.0
                    m.scale.x = bs
                    m.scale.y = bs
                    m.scale.z = bs
                    m.color.r = float(rc)
                    m.color.g = float(gc)
                    m.color.b = float(bc)
                    m.color.a = 0.95
                    m.lifetime = BuiltinDuration(sec=0, nanosec=0)
                    ma.markers.append(m)
        self.intro_pub.publish(ma)

    def _publish_stage3_preview(self):
        """All voxels colored by substructure + per-substructure build-order labels."""
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        ma = MarkerArray()

        # Voxels (fully-revealed decomp coloring)
        for z in range(self.H):
            for y in range(self.Y):
                for x in range(self.X):
                    if not self.target[z, y, x]:
                        continue
                    si = self.target_sub[z, y, x]
                    if si < 0:
                        continue
                    rc, gc, bc = _sub_color(si)
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = "preview"
                    m.id = int(z) * self.Y * self.X + int(y) * self.X + int(x)
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(x) * bs
                    m.pose.position.y = float(y) * bs
                    m.pose.position.z = float(z) * bs
                    m.pose.orientation.w = 1.0
                    m.scale.x = bs
                    m.scale.y = bs
                    m.scale.z = bs
                    m.color.r = float(rc)
                    m.color.g = float(gc)
                    m.color.b = float(bc)
                    m.color.a = 0.95
                    m.lifetime = BuiltinDuration(sec=0, nanosec=0)
                    ma.markers.append(m)

        # Build-order text label per substructure (ns="suborder")
        # Serial: "1", "2", …  Parallel: "1a", "1b", …  (matching Fig. 1c)
        for si, sub in enumerate(self.substructures):
            coords = np.argwhere(sub == 1)   # rows: (z, y, x)
            if len(coords) == 0:
                continue
            max_z = int(coords[:, 0].max())
            mean_x = float(coords[:, 2].mean())
            mean_y = float(coords[:, 1].mean())
            label = self.sub_labels.get(si, str(si + 1))

            tm = Marker()
            tm.header.frame_id = "world"
            tm.header.stamp = now
            tm.ns = "suborder"
            tm.id = si
            tm.type = Marker.TEXT_VIEW_FACING
            tm.action = Marker.ADD
            tm.pose.position.x = mean_x * bs
            tm.pose.position.y = mean_y * bs
            tm.pose.position.z = float(max_z + 1) * bs + bs * 0.6
            tm.pose.orientation.w = 1.0
            tm.scale.z = bs * 0.7
            tm.color.r = 1.0
            tm.color.g = 1.0
            tm.color.b = 1.0
            tm.color.a = 1.0
            tm.text = label
            tm.lifetime = BuiltinDuration(sec=0, nanosec=0)
            ma.markers.append(tm)

        self.intro_pub.publish(ma)

    def _enter_stage4(self):
        """Transition from Stage 3 to Stage 4: clear intro, start construction."""
        # Clear all intro-stage markers
        self._deleteall(self.intro_pub, "preview", "suborder")

        # Update stage label
        self._publish_stage_label("Stage 4: Construction")

        # Publish latched overlays that persist throughout construction
        self._publish_ghost()
        self._publish_subtext()

        self.stage = _STAGE_BUILD

        # CBS mode: precompute joint plans before any robot moves.
        if self.planner == "cbs":
            self._precompute_cbs_plans()

    # ------------------------------------------------------------------
    # CBS precompute
    # ------------------------------------------------------------------

    def _precompute_cbs_plans(self):
        """Precompute conflict-free joint plans, one per parallel group.

        Invariant (D-1): a fresh heightmap must be derived from the current
        world state before each group is handed to CBS. ``cbs_adapter``
        asserts this via ``assert_hm_matches_world``; a stale hm would
        silently plan against the wrong free/occupied cells.
        """
        num_robots = len(self.robots)
        # Simulated world that advances group-by-group so each group's
        # heightmap reflects all prior placements.
        world_sim = np.zeros_like(self.target, dtype=int)
        robot_starts = [(r.x, r.y) for r in self.robots]

        groups_meta = []
        for gi, group in enumerate(self.groups):
            subs = [self.substructures[si] for si in group]
            result = cbs_adapter.plan_group(
                substructures=subs,
                substructure_indices=list(group),
                world_init=world_sim,
                robot_starts=robot_starts,
                num_robots=num_robots,
                X=self.X,
                Y=self.Y,
                t_start=0,
                max_t=self.cbs_max_t,
                branch_limit=self.cbs_branch_limit,
            )
            groups_meta.append(result)

            md = result['metadata']
            self.get_logger().info(
                f"[CBS] group {gi} substructures={md['substructure_indices']} "
                f"blocks={md['block_count']} agents={md['num_agents_used']} "
                f"T_min={md['T_min']} T_final={md['T_final']} "
                f"solve={md['solve_time']:.3f}s "
                f"conflicts_resolved={md['conflicts_resolved']} "
                f"fallback={md['used_fallback']}"
            )

            # Advance world_sim so the next group plans against fresh hm.
            for si in group:
                mask = self.substructures[si] == 1
                world_sim[mask] = 1

            # Advance robot starts to each robot's final (x, y) in this group.
            new_starts = []
            for i, plan in enumerate(result['plans']):
                if plan:
                    new_starts.append((plan[-1].x, plan[-1].y))
                else:
                    new_starts.append(robot_starts[i])
            robot_starts = new_starts

        # Return phase — one joint plan from final positions to boundary.
        return_plans = cbs_adapter.plan_return(
            robot_starts=robot_starts,
            X=self.X, Y=self.Y,
            world_state=world_sim,
            t_start=0,
            max_t=max(200, self.cbs_max_t // 2),
        )
        self.get_logger().info(
            f"[CBS] return-phase plan: "
            f"{sum(1 for p in return_plans if p)} agents moving"
        )

        # Replay state
        self.cbs_groups = groups_meta
        self.cbs_group_idx = 0
        self.cbs_step_idx = [0] * num_robots
        self.cbs_return_plans = return_plans
        self.cbs_return_step_idx = [0] * num_robots

    # ------------------------------------------------------------------
    # Static one-shot publishers (called once when entering Stage 4)
    # ------------------------------------------------------------------

    def _publish_ghost(self):
        """Faint ghost of the full target colored by substructure (latched)."""
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        ma = MarkerArray()
        mid = 0
        for z in range(self.H):
            for y in range(self.Y):
                for x in range(self.X):
                    if not self.target[z, y, x]:
                        continue
                    si = self.target_sub[z, y, x]
                    rc, gc, bc = _sub_color(max(0, si))
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = "ghost"
                    m.id = mid
                    mid += 1
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(x) * bs
                    m.pose.position.y = float(y) * bs
                    m.pose.position.z = float(z) * bs
                    m.pose.orientation.w = 1.0
                    m.scale.x = bs
                    m.scale.y = bs
                    m.scale.z = bs
                    m.color.r = float(rc)
                    m.color.g = float(gc)
                    m.color.b = float(bc)
                    m.color.a = 0.15
                    m.lifetime = BuiltinDuration(sec=0, nanosec=0)
                    ma.markers.append(m)
        self.ghost_pub.publish(ma)

    def _publish_subtext(self):
        """TEXT_VIEW_FACING build-order label per substructure (latched)."""
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        ma = MarkerArray()
        for si, sub in enumerate(self.substructures):
            coords = np.argwhere(sub == 1)
            if len(coords) == 0:
                continue
            max_z = int(coords[:, 0].max())
            mean_x = float(coords[:, 2].mean())
            mean_y = float(coords[:, 1].mean())
            label = self.sub_labels.get(si, str(si + 1))

            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "subtext"
            m.id = si
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = mean_x * bs
            m.pose.position.y = mean_y * bs
            m.pose.position.z = float(max_z + 1) * bs + bs * 0.6
            m.pose.orientation.w = 1.0
            m.scale.z = bs * 0.55
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 1.0
            m.text = label
            m.lifetime = BuiltinDuration(sec=0, nanosec=0)
            ma.markers.append(m)
        self.text_pub.publish(ma)

    # ------------------------------------------------------------------
    # Simulation logic (Stage 4)
    # ------------------------------------------------------------------

    def can_place(self, x, y, z):
        if self.world[z, y, x] == 1:
            return False
        return z == 0 or self.world[z - 1, y, x] == 1

    def assign_tasks(self):
        for r in self.robots:
            if r.id in self.active or not self.task_queue:
                continue
            task = self.task_queue.pop(0)
            self.active[r.id] = {
                "phase": "to_pickup",
                "task": task,
                "pickup_cell": (DEPOT_X, DEPOT_Y),
                "path": None,
                "path_i": 0,
            }

    def robot_step_along(self, r, hm, path_state):
        path = path_state["path"]
        i = path_state["path_i"]
        if path is None or i >= len(path):
            return True
        nx, ny = path[i]
        r.z = surface_z(hm, r.x, r.y)
        nz = surface_z(hm, nx, ny)
        if abs(nz - r.z) <= 1:
            r.x, r.y = nx, ny
            r.z = nz
        path_state["path_i"] = i + 1
        return path_state["path_i"] >= len(path)

    # ------------------------------------------------------------------
    # Collision-aware movement (priority-based reservation table)
    # ------------------------------------------------------------------

    def _try_move(self, r, hm, path_state, reservations, committed_moves, t):
        """
        Advance robot r one step along path_state["path"] with collision
        avoidance via the running reservation table.

        Robots are processed in priority order (ascending robot id) within
        each tick; earlier robots' committed moves are visible to later ones.

        Checks:
          - Vertex conflict: target cell already reserved by another robot.
          - Swap conflict: another robot already committed the reverse move.
          - Carried-block conflict: robot's carried block would land on a
            reserved cell (i.e. target (x,y,z+1) is taken).

        On conflict the robot waits this tick and its wait_streak is
        incremented.  If wait_streak > 5, _try_jiggle is called to break
        potential deadlocks.

        Returns True if the robot moved (or path is already exhausted),
        False if the robot waited this tick.
        """
        path = path_state["path"]
        i = path_state["path_i"]
        if path is None or i >= len(path):
            return True  # nothing left to do; caller treats as "done"

        nx, ny = path[i]
        nz = surface_z(hm, nx, ny)

        # Stale-path guard: the world may have grown since BFS was computed.
        if abs(nz - r.z) > 1:
            path_state["path"] = None
            path_state["path_i"] = 0
            r.wait_streak += 1
            return False

        target_cell = (nx, ny, nz)

        # ---- Vertex conflict ------------------------------------------------
        blocker = reservations.get(target_cell)
        if blocker is not None and blocker != r.id:
            self.get_logger().info(
                f"[t={t}] Robot {r.id} waited (vertex conflict with "
                f"Robot {blocker} at {target_cell})"
            )
            r.wait_streak += 1
            self._try_jiggle(r, hm, path_state, reservations, committed_moves, t)
            return False

        # ---- Swap conflict --------------------------------------------------
        for other_id, (from_xy, to_xy) in committed_moves.items():
            if from_xy == (nx, ny) and to_xy == (r.x, r.y):
                self.get_logger().info(
                    f"[t={t}] Robot {r.id} waited (swap conflict with "
                    f"Robot {other_id} between ({r.x},{r.y}) and ({nx},{ny}))"
                )
                r.wait_streak += 1
                self._try_jiggle(
                    r, hm, path_state, reservations, committed_moves, t
                )
                return False

        # ---- Carried-block conflict -----------------------------------------
        if r.carrying:
            carry_cell = (nx, ny, nz + 1)
            carry_blocker = reservations.get(carry_cell)
            if carry_blocker is not None and carry_blocker != r.id:
                self.get_logger().info(
                    f"[t={t}] Robot {r.id} waited (carried-block conflict at "
                    f"{carry_cell} with Robot {carry_blocker})"
                )
                r.wait_streak += 1
                self._try_jiggle(
                    r, hm, path_state, reservations, committed_moves, t
                )
                return False

        # ---- All clear — commit the move ------------------------------------
        reservations.pop((r.x, r.y, r.z), None)
        if r.carrying:
            reservations.pop((r.x, r.y, r.z + 1), None)

        committed_moves[r.id] = ((r.x, r.y), (nx, ny))
        r.x, r.y, r.z = nx, ny, nz
        path_state["path_i"] = i + 1

        reservations[(r.x, r.y, r.z)] = r.id
        if r.carrying:
            reservations[(r.x, r.y, r.z + 1)] = r.id

        r.wait_streak = 0
        return True

    def _try_jiggle(self, r, hm, path_state, reservations, committed_moves, t):
        """
        If robot r has been stuck for more than 5 consecutive ticks, move it
        to any free adjacent cell to break a potential deadlock, then force a
        BFS replan on the next tick.

        The jiggle itself obeys vertex and swap rules against already-committed
        moves so it cannot create a new conflict.
        """
        if r.wait_streak <= 2:
            return

        candidates = list(neighbors4(r.x, r.y))
        random.shuffle(candidates)

        for jx, jy in candidates:
            if not in_bounds(jx, jy, self.X, self.Y):
                continue
            jz = surface_z(hm, jx, jy)
            if abs(jz - r.z) > 1:
                continue
            jcell = (jx, jy, jz)
            # Cell must be free of other robots
            if reservations.get(jcell, r.id) != r.id:
                continue
            # No swap conflict with already-committed moves
            if any(
                from_xy == (jx, jy) and to_xy == (r.x, r.y)
                for _, (from_xy, to_xy) in committed_moves.items()
            ):
                continue

            self.get_logger().info(
                f"[t={t}] Robot {r.id} JIGGLE to ({jx},{jy},{jz}) "
                f"(deadlock breaker after {r.wait_streak} waits)"
            )
            reservations.pop((r.x, r.y, r.z), None)
            if r.carrying:
                reservations.pop((r.x, r.y, r.z + 1), None)

            committed_moves[r.id] = ((r.x, r.y), (jx, jy))
            r.x, r.y, r.z = jx, jy, jz

            reservations[(r.x, r.y, r.z)] = r.id
            if r.carrying:
                reservations[(r.x, r.y, r.z + 1)] = r.id

            r.wait_streak = 0
            path_state["path"] = None   # force BFS replan from jiggled position
            path_state["path_i"] = 0
            return

    def _replan_after_detour(self, r, hm, state, t):
        """
        Immediately replan r's path from its current position after a jiggle or
        forced detour.  Returns True if a path was found and set, False if no path
        exists (caller should drop the task and re-queue it).
        """
        phase = state["phase"]
        if phase == "to_pickup":
            goal = state["pickup_cell"]
            p = bfs_path((r.x, r.y), goal, hm)
            if p is not None:
                state["path"] = p
                state["path_i"] = 0
                self.get_logger().info(
                    f"[t={t}] Robot {r.id} replanned "
                    f"({r.x},{r.y})→{goal} ({len(p) - 1} steps)"
                )
                return True
            self.get_logger().info(
                f"[t={t}] Robot {r.id} no path from ({r.x},{r.y}) "
                f"to pickup {goal} — dropping task"
            )
            return False

        elif phase == "to_place":
            x, y, z, _si = state["task"]
            cands = [
                (nx, ny)
                for nx, ny in neighbors4(x, y)
                if in_bounds(nx, ny, self.X, self.Y)
                and abs(int(hm[ny, nx]) - z) <= 1
            ]
            if not cands:
                self.get_logger().info(
                    f"[t={t}] Robot {r.id} no placement candidates "
                    "after detour — dropping task"
                )
                return False
            goal = min(cands, key=lambda c: abs(c[0] - r.x) + abs(c[1] - r.y))
            p = bfs_path((r.x, r.y), goal, hm)
            if p is not None:
                state["path"] = p
                state["path_i"] = 0
                self.get_logger().info(
                    f"[t={t}] Robot {r.id} replanned "
                    f"({r.x},{r.y})→{goal} ({len(p) - 1} steps)"
                )
                return True
            self.get_logger().info(
                f"[t={t}] Robot {r.id} no path from ({r.x},{r.y}) "
                f"to placement {goal} — dropping task"
            )
            return False

        # pickup / place phases don't navigate — nothing to replan
        state["path"] = None
        state["path_i"] = 0
        return True

    def _execute_forced_detour(self, r, hm, state, reservations, committed_moves, t):
        """
        Move r to any free adjacent cell this tick to resolve a swap conflict.
        Returns 'ok' (detoured + replanned), 'drop' (detoured but no replan path
        — caller should drop task), or 'wait' (no free cell, robot waits).
        """
        candidates = list(neighbors4(r.x, r.y))
        random.shuffle(candidates)
        for jx, jy in candidates:
            if not in_bounds(jx, jy, self.X, self.Y):
                continue
            jz = surface_z(hm, jx, jy)
            if abs(jz - r.z) > 1:
                continue
            jcell = (jx, jy, jz)
            if reservations.get(jcell, r.id) != r.id:
                continue
            if any(
                fxy == (jx, jy) and txy == (r.x, r.y)
                for _, (fxy, txy) in committed_moves.items()
            ):
                continue
            # Detour cell found
            self.get_logger().info(
                f"[t={t}] Robot {r.id} DETOUR to ({jx},{jy},{jz}) "
                "(swap-conflict resolution)"
            )
            reservations.pop((r.x, r.y, r.z), None)
            if r.carrying:
                reservations.pop((r.x, r.y, r.z + 1), None)
            committed_moves[r.id] = ((r.x, r.y), (jx, jy))
            r.x, r.y, r.z = jx, jy, jz
            reservations[(r.x, r.y, r.z)] = r.id
            if r.carrying:
                reservations[(r.x, r.y, r.z + 1)] = r.id
            r.wait_streak = 0
            ok = self._replan_after_detour(r, hm, state, t)
            return 'ok' if ok else 'drop'

        # No free adjacent cell found — robot waits this tick
        self.get_logger().info(
            f"[t={t}] Robot {r.id} swap-detour blocked (no free cell) — waiting"
        )
        r.wait_streak += 1
        return 'wait'

    # ------------------------------------------------------------------
    # CBS replay (Stage 4 / Stage 5)
    # ------------------------------------------------------------------

    def _apply_cbs_step(self, r, step, events, t):
        """Apply a single Step to robot r, updating world/world_sub as needed.

        ``events`` is the per-robot event dict returned by cbs_adapter
        (keyed by absolute t). ``t`` is the step's absolute t within the
        current group.
        """
        from macc_rviz.cbs_planner import (
            ACTION_MOVE, ACTION_WAIT, ACTION_PICKUP, ACTION_PLACE,
        )
        r.x, r.y, r.z = step.x, step.y, step.z
        if step.action == ACTION_PICKUP:
            ev = events.get(t)
            if ev is not None and ev[0] == 'pickup':
                r.carrying = True
                r.carrying_si = ev[1]
        elif step.action == ACTION_PLACE:
            ev = events.get(t)
            if ev is not None and ev[0] == 'place':
                _, bx, by, bz, si = ev
                if self.world[bz, by, bx] == 0:
                    self.world[bz, by, bx] = 1
                    self.world_sub[bz, by, bx] = si + 1
            r.carrying = False
            r.carrying_si = -1
        # MOVE / WAIT: position already updated; nothing else to do.

    def _tick_cbs(self):
        """Stage 4 + Stage 5 replay of precomputed CBS plans."""
        self.tick_count += 1
        t_abs = self.tick_count

        # Return phase (after build complete and inter-group pause elapsed).
        if self.build_complete:
            self._tick_cbs_return()
            return

        # Inter-group highlight pause.
        if self.pause_ticks > 0:
            self.pause_ticks -= 1
            self.publish_blocks()
            self.publish_robots()
            if self.pause_ticks == 0:
                self.just_completed_sis = set()
                if self.cbs_group_idx >= len(self.cbs_groups):
                    self.get_logger().info(
                        "Final group done. Robots returning to boundary."
                    )
                    self.build_complete = True
            return

        group = self.cbs_groups[self.cbs_group_idx]
        plans = group['plans']
        events = group['events']

        # Advance each robot one step (if its plan has one at this group-tick).
        # Plans use absolute t starting at t_start=0; we consume them linearly.
        for i, r in enumerate(self.robots):
            idx = self.cbs_step_idx[i]
            if idx < len(plans[i]):
                step = plans[i][idx]
                self._apply_cbs_step(r, step, events[i], step.t)
                self.cbs_step_idx[i] = idx + 1

        # Group done when every robot has consumed its plan.
        group_done = all(
            self.cbs_step_idx[i] >= len(plans[i])
            for i in range(len(self.robots))
        )
        if group_done:
            self.just_completed_sis = set(group['metadata']['substructure_indices'])
            self.cbs_group_idx += 1
            # Reset per-robot step cursor for next group.
            self.cbs_step_idx = [0] * len(self.robots)
            self.get_logger().info(
                f"[CBS] group {self.cbs_group_idx - 1} replay complete "
                f"at t={t_abs}."
            )
            self.pause_ticks = max(
                1, int(round(INTER_GROUP_PAUSE_SEC / self.step_duration_sec))
            )

        self.publish_blocks()
        self.publish_robots()

    def _tick_cbs_return(self):
        """Replay the CBS return-to-boundary plan; exit robots at boundary."""
        plans = self.cbs_return_plans
        t = self.tick_count

        if self.return_start_tick is None:
            self.return_start_tick = t
            self.get_logger().info(
                f"[CBS] return phase started at t={t} "
                f"({sum(1 for p in plans if p)} agents moving)"
            )

        all_done = True
        for i, r in enumerate(self.robots):
            if r.id in self.return_done:
                continue
            plan = plans[i]
            idx = self.cbs_return_step_idx[i]
            if idx < len(plan):
                step = plan[idx]
                r.x, r.y, r.z = step.x, step.y, step.z
                self.cbs_return_step_idx[i] = idx + 1

            on_boundary = (
                r.x == 0 or r.x == self.X - 1
                or r.y == 0 or r.y == self.Y - 1
            )
            plan_exhausted = self.cbs_return_step_idx[i] >= len(plan)
            if on_boundary and plan_exhausted:
                self.return_done.add(r.id)
                # Park off-grid in the direction of the nearest edge.
                r.x = r.x - 2 if r.x == 0 else (
                    r.x + 2 if r.x == self.X - 1 else r.x
                )
                r.y = r.y - 2 if r.y == 0 else (
                    r.y + 2 if r.y == self.Y - 1 else r.y
                )
                self.get_logger().info(
                    f"[CBS][t={t}] Robot {r.id} reached boundary."
                )
            else:
                all_done = False

        self.publish_robots()
        if all_done:
            self.get_logger().info(
                "Construction complete. All robots returned to boundary."
            )
            self.timer.cancel()

    # ------------------------------------------------------------------
    # Main timer callback — state machine
    # ------------------------------------------------------------------

    def tick(self):
        now = self.get_clock().now()

        # ----------------------------------------------------------------
        # Stages 1–3: intro sequence (no robots, no placed blocks)
        # ----------------------------------------------------------------
        if self.stage != _STAGE_BUILD:
            if self.stage_start_time is None:
                self.stage_start_time = now

            elapsed = (now - self.stage_start_time).nanoseconds * 1e-9

            if self.stage == _STAGE_INPUT:
                self._publish_stage_label("Stage 1: Input Structure")
                if elapsed >= STAGE_PAUSE_SEC:
                    self.get_logger().info("Intro: Stage 1 → Stage 2 (Decomposition)")
                    # Clear solid-blue preview before Stage 2 progressive reveal
                    self._deleteall(self.intro_pub, "preview")
                    self.stage = _STAGE_DECOMP
                    self.stage_start_time = now
                else:
                    self._publish_stage1_preview()
                return

            elif self.stage == _STAGE_DECOMP:
                self._publish_stage_label("Stage 2: Decomposition (Algorithm 1)")
                # Reveal one additional substructure every SUBSTRUCTURE_REVEAL_SEC.
                # Sub 0 (first by Algorithm 1) appears immediately.
                n_revealed = min(
                    len(self.substructures),
                    int(elapsed / SUBSTRUCTURE_REVEAL_SEC) + 1
                )
                self._publish_stage2_preview(n_revealed)
                total_time = (
                    len(self.substructures) * SUBSTRUCTURE_REVEAL_SEC + STAGE_PAUSE_SEC
                )
                if elapsed >= total_time:
                    self.get_logger().info("Intro: Stage 2 → Stage 3 (Build Order)")
                    self.stage = _STAGE_ORDER
                    self.stage_start_time = now
                return

            elif self.stage == _STAGE_ORDER:
                self._publish_stage_label("Stage 3: Build Order (Algorithm 3)")
                self._publish_stage3_preview()
                if elapsed >= STAGE_PAUSE_SEC:
                    self.get_logger().info("Intro: Stage 3 → Stage 4 (Construction)")
                    self._enter_stage4()
                return

        # ----------------------------------------------------------------
        # Stage 4: construction
        # ----------------------------------------------------------------

        # CBS planner: dispatch to precomputed-plan replay.
        if self.planner == "cbs":
            self._tick_cbs()
            return

        # Return-to-boundary phase (after build complete and final pause)
        if self.build_complete:
            hm = heightmap(self.world)
            self.tick_count += 1
            t = self.tick_count

            # FIX 3: On the very first tick of the return phase, assign stagger
            # delays so robots release one tick apart, closest-to-boundary first.
            if self.return_start_tick is None:
                self.return_start_tick = t

                def _dist_to_boundary(r):
                    return min(r.x, self.X - 1 - r.x, r.y, self.Y - 1 - r.y)

                sorted_r = sorted(self.robots, key=_dist_to_boundary)
                for delay, r in enumerate(sorted_r):
                    self.return_delay[r.id] = delay

            # FIX 5: Hard timeout — teleport any remaining robots to their
            # nearest boundary cell and mark them done.
            pending = [r for r in self.robots if r.id not in self.return_done]
            timeout_ticks = len(self.robots) * 20
            if pending and (t - self.return_start_tick) > timeout_ticks:
                for r in pending:
                    options = [
                        (0, r.y), (self.X - 1, r.y),
                        (r.x, 0), (r.x, self.Y - 1),
                    ]
                    bx, by = min(
                        options,
                        key=lambda c: abs(c[0] - r.x) + abs(c[1] - r.y)
                    )
                    self.return_done.add(r.id)
                    # Park just outside the grid on whichever side was chosen.
                    r.x = bx - 2 if bx == 0 else (bx + 2 if bx == self.X - 1 else bx)
                    r.y = by - 2 if by == 0 else (by + 2 if by == self.Y - 1 else by)
                    r.z = 0
                self.get_logger().warning(
                    f"[WARN] Return phase timed out, "
                    f"teleporting {len(pending)} robots to boundary."
                )
                self.publish_robots()
                self.get_logger().info(
                    "Construction complete. All robots returned to boundary."
                )
                self.timer.cancel()
                return

            # FIX 2: Reservation table contains only robots still inside the grid.
            reservations = {
                (r.x, r.y, r.z): r.id
                for r in self.robots
                if r.id not in self.return_done
            }
            committed_moves = {}

            def _mark_done(r):
                """Mark robot r as having exited; park it off-grid."""
                reservations.pop((r.x, r.y, r.z), None)
                self.return_done.add(r.id)
                remaining = sum(
                    1 for ro in self.robots if ro.id not in self.return_done
                )
                self.get_logger().info(
                    f"[t={t}] Robot {r.id} reached boundary at "
                    f"({r.x},{r.y},{r.z}), marked DONE. "
                    f"remaining={remaining}"
                )
                r.x = r.x - 2 if r.x == 0 else (
                    r.x + 2 if r.x == self.X - 1 else r.x
                )
                r.y = r.y - 2 if r.y == 0 else (
                    r.y + 2 if r.y == self.Y - 1 else r.y
                )

            for r in self.robots:
                if r.id in self.return_done:
                    continue

                # Check boundary BEFORE stagger delay: a robot already sitting
                # on an edge cell must exit immediately so it doesn't block
                # others during its stagger window.
                on_boundary = (
                    r.x == 0 or r.x == self.X - 1 or
                    r.y == 0 or r.y == self.Y - 1
                )
                if on_boundary:
                    _mark_done(r)
                    continue

                # FIX 3: Respect stagger delay before starting to move.
                if (t - self.return_start_tick) < self.return_delay.get(r.id, 0):
                    continue

                # Always route to nearest boundary cell via multi-goal BFS;
                # replan immediately if path was cleared by a jiggle.
                path_state = self.return_paths.get(r.id)
                if path_state is None or path_state.get("path") is None:
                    p = bfs_to_boundary((r.x, r.y), hm)
                    goal_str = (
                        f"({p[-1][0]},{p[-1][1]})" if p else "none"
                    )
                    is_replan = path_state is not None  # was cleared by jiggle
                    verb = "replanned" if is_replan else "return path"
                    steps = len(p) - 1 if p else 0
                    self.get_logger().info(
                        f"[t={t}] Robot {r.id} {verb} "
                        f"({r.x},{r.y})->boundary via {goal_str} "
                        f"({steps} steps)"
                    )
                    path_state = {
                        "path": p if p else [(r.x, r.y)],
                        "path_i": 0,
                    }
                    self.return_paths[r.id] = path_state

                self._try_move(r, hm, path_state, reservations, committed_moves, t)

                # Jiggle clears the path — replan immediately with multi-goal
                # BFS so the robot picks the nearest exit from its new position.
                if path_state.get("path") is None:
                    p = bfs_to_boundary((r.x, r.y), hm)
                    goal_str = (
                        f"({p[-1][0]},{p[-1][1]})" if p else "none"
                    )
                    steps = len(p) - 1 if p else 0
                    self.get_logger().info(
                        f"[t={t}] Robot {r.id} replanned "
                        f"({r.x},{r.y})->boundary via {goal_str} "
                        f"({steps} steps)"
                    )
                    path_state["path"] = p if p else [(r.x, r.y)]
                    path_state["path_i"] = 0

                # Check boundary again after the move.
                if (r.x == 0 or r.x == self.X - 1 or
                        r.y == 0 or r.y == self.Y - 1):
                    _mark_done(r)

            self.publish_robots()
            if all(r.id in self.return_done for r in self.robots):
                self.get_logger().info(
                    "Construction complete. All robots returned to boundary."
                )
                self.timer.cancel()
            return

        # Inter-group pause (highlight completed substructures)
        if self.pause_ticks > 0:
            self.pause_ticks -= 1
            self.publish_blocks()
            self.publish_robots()
            if self.pause_ticks == 0:
                self.just_completed_sis = set()
                if self.current_group_idx >= len(self.group_task_lists):
                    self.get_logger().info(
                        "Final group done. Robots returning to boundary."
                    )
                    self.build_complete = True
            return

        # Normal simulation tick
        hm = heightmap(self.world)
        self.assign_tasks()
        self.tick_count += 1
        t = self.tick_count

        # ------------------------------------------------------------------
        # Reservation table: maps (x, y, z) -> robot_id for this timestep.
        # Seeded with every robot's current position before any moves.
        # Idle robots' z may have drifted (if blocks were placed under them);
        # update them first so reservations are accurate.
        # ------------------------------------------------------------------
        for r in self.robots:
            if r.id not in self.active:
                r.z = surface_z(hm, r.x, r.y)

        reservations = {}
        for r in self.robots:
            reservations[(r.x, r.y, r.z)] = r.id
            if r.carrying:
                reservations[(r.x, r.y, r.z + 1)] = r.id

        # Track (from_xy, to_xy) for each robot that has committed a move
        # this tick — used for swap-conflict detection.
        committed_moves = {}

        # ------------------------------------------------------------------
        # Pre-pass: collect each active robot's intended next (x, y) from its
        # current path, then detect head-on swap conflicts before any robot moves.
        # Lower-priority robot (higher id) in each swap pair is added to
        # forced_detour — it will sidestep this tick instead of following its path.
        # ------------------------------------------------------------------
        intended_next = {}   # robot_id → (nx, ny) or None
        for _r in self.robots:
            if _r.id not in self.active:
                intended_next[_r.id] = None
                continue
            _st = self.active[_r.id]
            _path = _st.get("path")
            _pi = _st.get("path_i", 0)
            intended_next[_r.id] = _path[_pi] if (_path and _pi < len(_path)) else None

        forced_detour = set()   # robot IDs that must execute a detour this tick
        for _i, _ra in enumerate(self.robots):
            if _ra.id not in self.active or _ra.id in forced_detour:
                continue
            _ra_next = intended_next.get(_ra.id)
            if _ra_next is None:
                continue
            for _rb in self.robots[_i + 1:]:
                if _rb.id not in self.active or _rb.id in forced_detour:
                    continue
                # Swap: _ra wants _rb's cell AND _rb wants _ra's cell
                if (_rb.x, _rb.y) == _ra_next and \
                        intended_next.get(_rb.id) == (_ra.x, _ra.y):
                    forced_detour.add(_rb.id)   # _ra has priority (lower id)
                    self.get_logger().info(
                        f"[t={t}] SWAP conflict: Robot {_ra.id} "
                        f"({_ra.x},{_ra.y})→{_ra_next} ↔ "
                        f"Robot {_rb.id} ({_rb.x},{_rb.y})"
                        f"→({_ra.x},{_ra.y}). "
                        f"Robot {_rb.id} forced to detour."
                    )
                    break

        for r in self.robots:
            if r.id not in self.active:
                continue

            state = self.active[r.id]
            x, y, z, si = state["task"]

            # Forced detour this tick (swap-conflict resolution)
            if r.id in forced_detour:
                result = self._execute_forced_detour(
                    r, hm, state, reservations, committed_moves, t
                )
                if result == 'drop':
                    if r.carrying:
                        r.carrying = False
                        r.carrying_si = -1
                        reservations.pop((r.x, r.y, r.z + 1), None)
                    self.task_queue.append(state["task"])
                    del self.active[r.id]
                continue

            if state["phase"] == "to_pickup":
                px, py = state["pickup_cell"]
                if (r.x, r.y) != (px, py):
                    if state["path"] is None:
                        p = bfs_path((r.x, r.y), (px, py), hm)
                        state["path"] = p if p is not None else [(r.x, r.y)]
                        state["path_i"] = 0
                    moved = self._try_move(
                        r, hm, state, reservations, committed_moves, t
                    )
                    # If jiggle cleared the path, replan immediately from new position
                    if not moved and state["path"] is None:
                        ok = self._replan_after_detour(r, hm, state, t)
                        if not ok:
                            self.task_queue.append(state["task"])
                            del self.active[r.id]
                            continue
                    if moved and state["path_i"] >= len(state["path"]):
                        state["phase"] = "pickup"
                        state["path"] = None
                else:
                    state["phase"] = "pickup"

            elif state["phase"] == "pickup":
                # Robot stays put — just mark it as carrying.
                # Reserve the carried-block cell so other robots see it
                # immediately within the same tick.
                r.carrying = True
                r.carrying_si = si
                state["phase"] = "to_place"
                reservations[(r.x, r.y, r.z + 1)] = r.id

            elif state["phase"] == "to_place":
                if not self.can_place(x, y, z):
                    self.task_queue.append(state["task"])
                    r.carrying = False
                    r.carrying_si = -1
                    reservations.pop((r.x, r.y, r.z + 1), None)
                    del self.active[r.id]
                    continue

                candidates = [
                    (nx, ny)
                    for nx, ny in neighbors4(x, y)
                    if in_bounds(nx, ny, self.X, self.Y)
                    and abs(int(hm[ny, nx]) - z) <= 1
                ]
                if not candidates:
                    self.task_queue.append(state["task"])
                    r.carrying = False
                    r.carrying_si = -1
                    reservations.pop((r.x, r.y, r.z + 1), None)
                    del self.active[r.id]
                    continue

                goal = min(candidates, key=lambda c: abs(c[0] - r.x) + abs(c[1] - r.y))
                if (r.x, r.y) != goal:
                    if state["path"] is None:
                        p = bfs_path((r.x, r.y), goal, hm)
                        state["path"] = p if p is not None else [(r.x, r.y)]
                        state["path_i"] = 0
                    moved = self._try_move(
                        r, hm, state, reservations, committed_moves, t
                    )
                    # If jiggle cleared the path, replan immediately from new position
                    if not moved and state["path"] is None:
                        ok = self._replan_after_detour(r, hm, state, t)
                        if not ok:
                            r.carrying = False
                            r.carrying_si = -1
                            reservations.pop((r.x, r.y, r.z + 1), None)
                            self.task_queue.append(state["task"])
                            del self.active[r.id]
                            continue
                    if moved and state["path_i"] >= len(state["path"]):
                        state["phase"] = "place"
                        state["path"] = None
                else:
                    state["phase"] = "place"

            elif state["phase"] == "place":
                # Reserve the target voxel so no other robot steps on it or
                # tries to place into it this same tick.
                reservations[(x, y, z)] = r.id
                if self.can_place(x, y, z):
                    self.world[z, y, x] = 1
                    self.world_sub[z, y, x] = si + 1   # store 1-indexed
                reservations.pop((r.x, r.y, r.z + 1), None)
                r.carrying = False
                r.carrying_si = -1
                del self.active[r.id]

        # Check group completion
        group_done = len(self.task_queue) == 0 and len(self.active) == 0
        if group_done:
            self.just_completed_sis = set(self.groups[self.current_group_idx])
            self.current_group_idx += 1
            if self.current_group_idx < len(self.group_task_lists):
                self.task_queue = list(self.group_task_lists[self.current_group_idx])
                self.get_logger().info(
                    f"Group {self.current_group_idx - 1} complete. "
                    f"Pausing {INTER_GROUP_PAUSE_SEC}s before group "
                    f"{self.current_group_idx}."
                )
            else:
                self.get_logger().info(
                    f"Final group {self.current_group_idx - 1} complete. "
                    f"Highlighting for {INTER_GROUP_PAUSE_SEC}s then done."
                )
            self.pause_ticks = max(
                1, int(round(INTER_GROUP_PAUSE_SEC / self.step_duration_sec))
            )

        self.publish_blocks()
        self.publish_robots()

    # ------------------------------------------------------------------
    # Publishing (Stage 4)
    # ------------------------------------------------------------------

    def publish_blocks(self):
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        ma = MarkerArray()

        for z in range(self.H):
            for y in range(self.Y):
                for x in range(self.X):
                    if not self.world[z, y, x]:
                        continue

                    si = self.world_sub[z, y, x] - 1   # 0-indexed
                    rc, gc, bc = _sub_color(si)
                    alpha = 0.9

                    # Brighten recently-completed substructure during inter-group pause
                    if si in self.just_completed_sis:
                        rc = min(1.0, rc * 1.6)
                        gc = min(1.0, gc * 1.6)
                        bc = min(1.0, bc * 1.6)
                        alpha = 1.0

                    mid = int(z) * self.Y * self.X + int(y) * self.X + int(x)
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = "blocks"
                    m.id = mid
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(x) * bs
                    m.pose.position.y = float(y) * bs
                    m.pose.position.z = float(z) * bs
                    m.pose.orientation.w = 1.0
                    m.scale.x = bs
                    m.scale.y = bs
                    m.scale.z = bs
                    m.color.r = float(rc)
                    m.color.g = float(gc)
                    m.color.b = float(bc)
                    m.color.a = float(alpha)
                    ma.markers.append(m)

        self.blocks_pub.publish(ma)

    def publish_robots(self):
        now = self.get_clock().now().to_msg()
        bs = self.block_scale
        ma = MarkerArray()

        for r in self.robots:
            rc, gc, bc = _ROBOT_COLOR
            rz = float(r.z) * bs   # robot center flush with block at same z

            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = f"robot_{r.id}"
            m.id = 0
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(r.x) * bs
            m.pose.position.y = float(r.y) * bs
            m.pose.position.z = rz
            m.pose.orientation.w = 1.0
            m.scale.x = bs
            m.scale.y = bs
            m.scale.z = bs
            m.color.r = float(rc)
            m.color.g = float(gc)
            m.color.b = float(bc)
            m.color.a = 1.0
            ma.markers.append(m)

            # Carried block: stacked directly on top of robot, colored by destination sub
            cb = Marker()
            cb.header.frame_id = "world"
            cb.header.stamp = now
            cb.ns = f"carried_{r.id}"
            cb.id = 0
            cb.type = Marker.CUBE
            if r.carrying and r.carrying_si >= 0:
                cr, cg, cbv = _sub_color(r.carrying_si)
                cb.action = Marker.ADD
                cb.pose.position.x = float(r.x) * bs
                cb.pose.position.y = float(r.y) * bs
                cb.pose.position.z = rz + bs
                cb.pose.orientation.w = 1.0
                cb.scale.x = bs
                cb.scale.y = bs
                cb.scale.z = bs
                cb.color.r = float(cr)
                cb.color.g = float(cg)
                cb.color.b = float(cbv)
                cb.color.a = 1.0
            else:
                cb.action = Marker.DELETE
            ma.markers.append(cb)

        self.robots_pub.publish(ma)


def main():
    rclpy.init()
    node = MACCRvizSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
