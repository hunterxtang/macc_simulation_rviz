import hashlib
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

        num_robots = int(self.get_parameter("num_robots").value)
        self.step_duration_sec = float(self.get_parameter("step_duration_sec").value)
        self.block_scale = float(self.get_parameter("block_scale").value)
        use_example = bool(self.get_parameter("use_example_structure").value)
        seed_param = int(self.get_parameter("seed").value)

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

        # --- intro stage state machine ---
        self.stage = _STAGE_INPUT
        self.stage_start_time = None   # initialised lazily on first tick

        # --- simulation timer — period = step_duration_sec directly, no divisors ---
        self.timer = self.create_timer(self.step_duration_sec, self.tick)

        block_count = int(self.target.sum())
        grid_hash = hashlib.md5(self.target.tobytes()).hexdigest()[:8]
        seed_str = str(seed) if seed is not None else "n/a (example)"
        self.get_logger().info(f"Using random seed: {seed_str}")
        self.get_logger().info(
            f"Structure: {block_count} blocks, fingerprint={grid_hash} "
            f"(shape Z,Y,X={self.target.shape})"
        )
        self.get_logger().info(f"Substructures: {len(self.substructures)}")
        self.get_logger().info(f"Parallel groups: {self.groups}")
        self.get_logger().info(f"Build order: {self.order}")
        self.get_logger().info(
            f"Step duration: {self.step_duration_sec:.2f}s "
            f"(STEP_DURATION_SEC={STEP_DURATION_SEC})"
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

        # Return-to-boundary phase (after build complete and final pause)
        if self.build_complete:
            hm = heightmap(self.world)
            all_home = True
            for r in self.robots:
                bx, by = DEPOT_X, DEPOT_Y
                if (r.x, r.y) == (bx, by):
                    continue
                all_home = False
                if r.id not in self.return_paths:
                    p = bfs_path((r.x, r.y), (bx, by), hm)
                    self.return_paths[r.id] = {
                        "path": p if p else [(r.x, r.y)],
                        "path_i": 0,
                    }
                self.robot_step_along(r, hm, self.return_paths[r.id])
            self.publish_robots()
            if all_home:
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

        for r in self.robots:
            if r.id not in self.active:
                r.z = surface_z(hm, r.x, r.y)
                continue

            state = self.active[r.id]
            x, y, z, si = state["task"]

            if state["phase"] == "to_pickup":
                px, py = state["pickup_cell"]
                if (r.x, r.y) != (px, py):
                    if state["path"] is None:
                        p = bfs_path((r.x, r.y), (px, py), hm)
                        state["path"] = p if p is not None else [(r.x, r.y)]
                        state["path_i"] = 0
                    done = self.robot_step_along(r, hm, state)
                    if done:
                        state["phase"] = "pickup"
                        state["path"] = None
                else:
                    state["phase"] = "pickup"

            elif state["phase"] == "pickup":
                r.carrying = True
                r.carrying_si = si
                state["phase"] = "to_place"

            elif state["phase"] == "to_place":
                if not self.can_place(x, y, z):
                    self.task_queue.append(state["task"])
                    r.carrying = False
                    r.carrying_si = -1
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
                    del self.active[r.id]
                    continue

                goal = min(candidates, key=lambda c: abs(c[0] - r.x) + abs(c[1] - r.y))
                if (r.x, r.y) != goal:
                    if state["path"] is None:
                        p = bfs_path((r.x, r.y), goal, hm)
                        state["path"] = p if p is not None else [(r.x, r.y)]
                        state["path_i"] = 0
                    done = self.robot_step_along(r, hm, state)
                    if done:
                        state["phase"] = "place"
                        state["path"] = None
                else:
                    state["phase"] = "place"

            elif state["phase"] == "place":
                if self.can_place(x, y, z):
                    self.world[z, y, x] = 1
                    self.world_sub[z, y, x] = si + 1   # store 1-indexed
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
            rz = float(r.z) * bs + bs * 0.5   # robot sits one half-block above surface

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
