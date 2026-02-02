import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from macc_rviz.structure_utils import create_random_structure, create_example_structure
from macc_rviz.decomposition import decompose_structure, order_substructures
from macc_rviz.parallel import find_parallel_groups

def heightmap(world):
    H, Y, X = world.shape
    hm = np.zeros((Y, X), dtype=int)
    for y in range(Y):
        for x in range(X):
            col = world[:, y, x]
            hm[y, x] = int(np.sum(col))
    return hm

def in_bounds(x, y, X, Y):
    return 0 <= x < X and 0 <= y < Y

def neighbors4(x, y):
    return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

def bfs_path(start, goal, hm):
    Y, X = hm.shape
    sx, sy = start
    gx, gy = goal
    q = [(sx, sy)]
    prev = { (sx, sy): None }
    while q:
        x, y = q.pop(0)
        if (x, y) == (gx, gy):
            break
        for nx, ny in neighbors4(x, y):
            if not in_bounds(nx, ny, X, Y):
                continue
            if (nx, ny) in prev:
                continue
            if abs(hm[ny, nx] - hm[y, x]) <= 1:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    if (gx, gy) not in prev:
        return None
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def surface_z(hm, x, y):
    return hm[y, x]

def blocks_in_substructure_bottom_up(sub):
    H, Y, X = sub.shape
    coords = []
    for y in range(Y):
        for x in range(X):
            zs = np.where(sub[:, y, x] == 1)[0]
            if len(zs) == 0:
                continue
            for z in sorted(zs):
                coords.append((x, y, int(z)))
    coords.sort(key=lambda t: (t[2], t[1], t[0]))
    return coords

class Robot:
    def __init__(self, rid, x, y, color):
        self.id = rid
        self.x = x
        self.y = y
        self.z = 0
        self.carrying = False
        self.color = color

class MACCRvizSim(Node):
    def __init__(self):
        super().__init__("macc_rviz_sim")

        self.blocks_pub = self.create_publisher(MarkerArray, "/macc/blocks", 10)
        self.robots_pub = self.create_publisher(MarkerArray, "/macc/robots", 10)

        self.declare_parameter("num_robots", 4)
        self.declare_parameter("step_hz", 10.0)
        self.declare_parameter("block_scale", 1.0)
        self.declare_parameter("robot_scale", 0.8)
        self.declare_parameter("use_example_structure", False)
        self.declare_parameter("seed", 42)

        num_robots = int(self.get_parameter("num_robots").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.block_scale = float(self.get_parameter("block_scale").value)
        self.robot_scale = float(self.get_parameter("robot_scale").value)
        use_example = bool(self.get_parameter("use_example_structure").value)
        seed = int(self.get_parameter("seed").value)

        if use_example:
            self.target = create_example_structure()
        else:
            self.target = create_random_structure(x=7, y=7, z=4, density=0.4, seed=seed)

        self.H, self.Y, self.X = self.target.shape
        self.world = np.zeros_like(self.target, dtype=int)

        substructures = decompose_structure(self.target)
        order, deps = order_substructures(substructures)
        self.groups = find_parallel_groups(substructures, deps)

        self.plan = []
        for group in self.groups:
            for si in group:
                coords = blocks_in_substructure_bottom_up(substructures[si])
                for (x, y, z) in coords:
                    self.plan.append((x, y, z, si))

        colors = [
            (1.0, 0.2, 0.2),
            (0.2, 1.0, 0.2),
            (0.2, 0.2, 1.0),
            (1.0, 0.7, 0.2),
            (0.6, 0.2, 1.0),
            (0.2, 1.0, 1.0),
        ]
        self.robots = []
        for i in range(num_robots):
            self.robots.append(Robot(i, 0, 0, colors[i % len(colors)]))

        self.task_queue = list(self.plan)
        self.active = {}
        self.timer = self.create_timer(1.0 / self.step_hz, self.tick)

        self.get_logger().info("Target size z,y,x = " + str(self.target.shape))
        self.get_logger().info("Substructures = " + str(len(substructures)))
        self.get_logger().info("Parallel groups = " + str(self.groups))
        self.get_logger().info("Total block tasks = " + str(len(self.task_queue)))

    def can_place(self, x, y, z):
        if self.world[z, y, x] == 1:
            return False
        if z == 0:
            return True
        return self.world[z-1, y, x] == 1

    def assign_tasks(self):
        for r in self.robots:
            if r.id in self.active:
                continue
            if len(self.task_queue) == 0:
                continue
            self.active[r.id] = {
                "phase": "to_pickup",
                "task": self.task_queue.pop(0),
                "path": None,
                "path_i": 0
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
        else:
            path_state["path_i"] = i + 1
        return path_state["path_i"] >= len(path)

    def tick(self):
        hm = heightmap(self.world)

        self.assign_tasks()

        for r in self.robots:
            if r.id not in self.active:
                r.z = surface_z(hm, r.x, r.y)
                continue

            state = self.active[r.id]
            x, y, z, si = state["task"]

            if state["phase"] == "to_pickup":
                if (r.x, r.y) != (0, 0):
                    if state["path"] is None:
                        p = bfs_path((r.x, r.y), (0, 0), hm)
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
                state["phase"] = "to_place"

            elif state["phase"] == "to_place":
                if not self.can_place(x, y, z):
                    self.task_queue.append(state["task"])
                    r.carrying = False
                    del self.active[r.id]
                    continue

                candidates = []
                for nx, ny in neighbors4(x, y):
                    if not in_bounds(nx, ny, self.X, self.Y):
                        continue
                    if abs(hm[ny, nx] - z) <= 1:
                        candidates.append((nx, ny))
                if len(candidates) == 0:
                    self.task_queue.append(state["task"])
                    r.carrying = False
                    del self.active[r.id]
                    continue

                goal = min(candidates, key=lambda c: abs(c[0]-r.x) + abs(c[1]-r.y))

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
                r.carrying = False
                del self.active[r.id]

        self.publish_blocks()
        self.publish_robots()

        if len(self.task_queue) == 0 and len(self.active) == 0:
            self.get_logger().info("Build complete.")
            self.timer.cancel()

    def publish_blocks(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        mid = 0

        ys, xs = np.where(np.any(self.world == 1, axis=0))
        for y in range(self.Y):
            for x in range(self.X):
                zs = np.where(self.world[:, y, x] == 1)[0]
                for z in zs:
                    m = Marker()
                    m.header.frame_id = "world"
                    m.header.stamp = now
                    m.ns = "blocks"
                    m.id = mid
                    mid += 1
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = float(x) * self.block_scale
                    m.pose.position.y = float(y) * self.block_scale
                    m.pose.position.z = float(z) * self.block_scale
                    m.pose.orientation.w = 1.0
                    m.scale.x = self.block_scale
                    m.scale.y = self.block_scale
                    m.scale.z = self.block_scale
                    m.color.r = 0.4
                    m.color.g = 0.6
                    m.color.b = 1.0
                    m.color.a = 1.0
                    ma.markers.append(m)

        self.blocks_pub.publish(ma)

    def publish_robots(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        for r in self.robots:
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = now
            m.ns = "robots"
            m.id = r.id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(r.x) * self.block_scale
            m.pose.position.y = float(r.y) * self.block_scale
            m.pose.position.z = float(r.z) * self.block_scale + (self.block_scale * 0.5)
            m.pose.orientation.w = 1.0
            m.scale.x = self.robot_scale
            m.scale.y = self.robot_scale
            m.scale.z = self.robot_scale
            if r.carrying:
                m.color.r = 1.0
                m.color.g = 1.0
                m.color.b = 0.2
            else:
                m.color.r, m.color.g, m.color.b = r.color
            m.color.a = 1.0
            ma.markers.append(m)

            if r.carrying:
                b = Marker()
                b.header.frame_id = "world"
                b.header.stamp = now
                b.ns = "carried"
                b.id = r.id
                b.type = Marker.CUBE
                b.action = Marker.ADD
                b.pose.position.x = float(r.x) * self.block_scale
                b.pose.position.y = float(r.y) * self.block_scale
                b.pose.position.z = float(r.z) * self.block_scale + (self.block_scale * 1.3)
                b.pose.orientation.w = 1.0
                b.scale.x = self.block_scale * 0.5
                b.scale.y = self.block_scale * 0.5
                b.scale.z = self.block_scale * 0.5
                b.color.r = 0.9
                b.color.g = 0.9
                b.color.b = 0.9
                b.color.a = 1.0
                ma.markers.append(b)

        self.robots_pub.publish(ma)

def main():
    rclpy.init()
    node = MACCRvizSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
