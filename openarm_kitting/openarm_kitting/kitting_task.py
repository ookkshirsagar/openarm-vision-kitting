"""Vision-guided kitting: picks the parts the camera finds in the inbound zone and places one of each class
in its slot on the kit fixture; parts the kit does not need go to the reject tray.

  ros2 run openarm_kitting kitting_task --ros-args -p config:=<kitting.yaml> [-p cycles:=0] [-p execute:=true]

Each cycle: wait for fresh detections (/kitting/detections), put them in the planning scene as collision
objects, choose a part and its target, plan pick → place → back to `ready` with MoveIt Task Constructor and
execute it. Parts that fail to plan are skipped for that cycle; the task ends when the inbound zone is empty
or nothing left there can be planned.

Motions: free-space transfers are Pilz PTP (a straight line in joint space: the same, predictable path every
time, as on a commissioned cell); OMPL plans only the transfers PTP cannot do without a collision. Approach,
lift, place and retreat are straight Cartesian lines. /kitting/status says what the cell is doing.
"""
import math
import time

from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from moveit.task_constructor import core, stages
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from openarm_kitting.layout import random_layout
import rclcpp                                         # MoveIt Task Constructor's own (C++) node
import rclpy
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header, String
from vision_msgs.msg import Detection3DArray
import yaml


def quat_from_matrix(m):
    """(x, y, z, w) of a 3x3 rotation matrix given as rows."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4)
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + m[i][i] - m[j][j] - m[k][k])
    q = [0.0, 0.0, 0.0, (m[k][j] - m[j][k]) / s]
    q[i], q[j], q[k] = s / 4, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s
    return tuple(q)


def yaw_pose(x, y, z, yaw, frame="world"):
    p = PoseStamped(header=Header(frame_id=frame))
    p.pose.position.x, p.pose.position.y, p.pose.position.z = float(x), float(y), float(z)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def twist(frame, z):
    t = TwistStamped(header=Header(frame_id=frame))
    t.twist.linear.z = float(z)
    return t


def timing_ok(solution):
    """Every sub-trajectory of an MTC solution has strictly increasing waypoint times."""
    for sub in solution.toMsg().sub_trajectory:
        t = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in sub.trajectory.joint_trajectory.points]
        if any(b <= a for a, b in zip(t, t[1:])):
            return False
    return True


def part_height(spec):
    """Height of a catalogue part as it rests: box z, upright cylinder length, lying cylinder diameter."""
    if spec["shape"] == "box":
        return spec["size"][2]
    return spec["size"][1] if spec.get("axis", "y") == "z" else spec["size"][0]


class Part:
    """One detected part: class, pose (centre, yaw of the frame whose +Y is the long side) and catalogue shape."""

    def __init__(self, name, cls, x, y, z, yaw, spec):
        self.name, self.cls, self.x, self.y, self.z, self.yaw, self.spec = name, cls, x, y, z, yaw, spec

    @property
    def height(self):
        return part_height(self.spec)

    @property
    def width(self):
        """The footprint's short side: what the fingers close across."""
        size = self.spec["size"]
        return min(size[:2]) if self.spec["shape"] == "box" else size[0]

    def collision_object(self):
        co = CollisionObject(id=self.name, operation=CollisionObject.ADD)
        co.header.frame_id = "world"
        # 1 mm clear of the surface it rests on, so resting contact is not a collision
        co.pose = yaw_pose(self.x, self.y, self.z + 0.001, self.yaw).pose
        prim, prim_pose = SolidPrimitive(), Pose()
        if self.spec["shape"] == "box":
            sx, sy, sz = self.spec["size"]
            w, length = min(sx, sy), max(sx, sy)
            prim.type, prim.dimensions = SolidPrimitive.BOX, [w, length, sz - 0.002]
            prim_pose.orientation.w = 1.0
        else:
            d, length = self.spec["size"]
            if self.spec.get("axis", "y") == "z":    # upright cylinder
                prim.type, prim.dimensions = SolidPrimitive.CYLINDER, [length - 0.002, d / 2]
                prim_pose.orientation.w = 1.0
            else:                                    # cylinder on its side, axis along the part's +Y
                prim.type, prim.dimensions = SolidPrimitive.CYLINDER, [length, d / 2 - 0.001]
                prim_pose.orientation.x = prim_pose.orientation.w = math.sqrt(0.5)
        co.primitives, co.primitive_poses = [prim], [prim_pose]
        return co


class Kitting(Node):
    def __init__(self):
        super().__init__("kitting_task")
        config = self.declare_parameter("config", "").value
        self.cycles = self.declare_parameter("cycles", 0).value              # 0 = until the inbound zone is empty
        self.execute = self.declare_parameter("execute", True).value
        self.settle = self.declare_parameter("settle_time", 1.0).value       # s after a motion before detecting
        self.max_retries = self.declare_parameter("max_retries", 3).value      # failed grasps in a row before stopping
        # detections:=simulated takes the parts from the seeded layout instead of the camera (mock hardware)
        self.simulated = self.declare_parameter("detections", "camera").value == "simulated"
        seed = self.declare_parameter("seed", 1).value
        n_parts = self.declare_parameter("parts", 4).value
        self.cfg = yaml.safe_load(open(config))["/**"]["ros__parameters"]
        self.arm = self.cfg["arm"]
        self.kit_free = dict(self.cfg["kit_slots"])                          # class → x, y, yaw
        r = self.cfg["reject_slots"]
        self.reject_free = [(r[i], r[i + 1]) for i in range(0, len(r), 2)]
        self.scene_ids = set()
        if self.simulated:
            self.layout = random_layout(self.cfg, n_parts, seed)
        self.detections = None
        self.create_subscription(Detection3DArray, "/kitting/detections", self.on_detections, 10)
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.status = self.create_publisher(String, "/kitting/status", 10)

        self.mtc_node = rclcpp.Node("kitting_task_mtc")
        self.ptp = core.PipelinePlanner(self.mtc_node, "pilz_industrial_motion_planner", "PTP")
        self.ompl = core.PipelinePlanner(self.mtc_node, "ompl", "RRTConnectkConfigDefault")
        self.cartesian = core.CartesianPath()
        for planner in (self.ptp, self.ompl, self.cartesian):
            planner.max_velocity_scaling_factor = self.cfg["max_velocity_scaling"]
            planner.max_acceleration_scaling_factor = self.cfg["max_acceleration_scaling"]
        self.transfer = core.MultiPlanner()                  # each transfer: PTP, or OMPL where PTP collides
        self.transfer.add(self.ptp)
        self.transfer.add(self.ompl)
        self.joint_interp = core.JointInterpolationPlanner()
        self.robot_model = None

    def on_detections(self, msg):
        self.detections = msg

    def say(self, text):
        self.status.publish(String(data=text))

    # ------------------------------------------------------------ perception → planning scene
    def wait_for_parts(self):
        """Parts from the first detection message stamped after now + settle_time."""
        if self.simulated:
            return [self.part(f"part_{i}", cls, x, y, yaw) for i, (cls, x, y, yaw) in enumerate(self.layout)]
        t0 = self.get_clock().now().nanoseconds + int(self.settle * 1e9)
        self.detections = None
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            d = self.detections
            if d is not None and d.header.stamp.sec * 1e9 + d.header.stamp.nanosec >= t0:
                break
        parts = []
        for i, det in enumerate(d.detections):
            c = det.bbox.center
            yaw = 2 * math.atan2(c.orientation.z, c.orientation.w)
            parts.append(self.part(f"part_{i}", det.results[0].hypothesis.class_id, c.position.x, c.position.y, yaw))
        return parts

    def part(self, name, cls, x, y, yaw):
        """A part resting on the worktop; its height comes from the catalogue."""
        spec = self.cfg["parts"][cls]
        return Part(name, cls, x, y, self.cfg["work_z"] + part_height(spec) / 2, yaw, spec)

    def apply_scene(self, parts):
        ps = PlanningScene(is_diff=True)
        for name in self.scene_ids:
            co = CollisionObject(id=name, operation=CollisionObject.REMOVE)
            co.header.frame_id = "world"
            ps.world.collision_objects.append(co)
        ps.world.collision_objects += [p.collision_object() for p in parts]
        self.scene_ids = {p.name for p in parts}
        self.scene.wait_for_service()
        rclpy.spin_until_future_complete(self, self.scene.call_async(ApplyPlanningScene.Request(scene=ps)))

    def keep_in_scene(self, part, new_name):
        """A placed part stays in the scene as an obstacle under a name the next cycle will not remove."""
        self.scene_ids.discard(part.name)
        ps = PlanningScene(is_diff=True)
        co = CollisionObject(id=part.name, operation=CollisionObject.REMOVE)
        co.header.frame_id = "world"
        ps.world.collision_objects.append(co)
        part.name = new_name
        ps.world.collision_objects.append(part.collision_object())
        rclpy.spin_until_future_complete(self, self.scene.call_async(ApplyPlanningScene.Request(scene=ps)))

    # ------------------------------------------------------------ task
    def ik_frame(self, part):
        """Hand frame that coincides with the part frame at the grasp: z up (against the approach), x across the
        part's width, so GenerateGraspPose's 0 and 180 deg samples close the fingers across the part."""
        surface = self.cfg["work_z"]
        top = part.z + part.height / 2
        if "grasp_height" in part.spec:                         # per part: fingertips this far above the surface
            tcp_z = surface + part.spec["grasp_height"]
        else:
            tcp_z = max(surface + self.cfg["tcp_min_height"], top - self.cfg["tcp_max_depth"])
        along = self.cfg["tcp_offset"] - (part.z - tcp_z)       # grasp frame origin → part centre, along the approach
        f = PoseStamped(header=Header(frame_id=f"openarm_{self.arm}_grasp_frame"))
        f.pose.position.x = along
        # columns: ik x = grasp y (finger closing direction), ik y = -grasp z, ik z = -grasp x (against the approach)
        x, y, z, w = quat_from_matrix([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        f.pose.orientation.x, f.pose.orientation.y, f.pose.orientation.z, f.pose.orientation.w = x, y, z, w
        return f

    def pregrasp(self, part):
        """Named finger opening for this part (SRDF): ~30 mm wider than the part at the fingertips. Opening wider
        than needed only lets the fingers hit the tray walls, neighbouring parts and the parts already in the kit."""
        # tip opening ≈ 2 · 100 mm · sin(angle): narrow 0.32 rad ≈ 63 mm, wide 0.48 rad ≈ 92 mm
        return "pregrasp_narrow" if part.width + 0.03 <= 0.063 else "pregrasp_wide"

    def align_round_part(self, part, target_xy, offset):
        """Turn a round part's frame (in the planning scene too) to point from the part to its target, plus offset.
        Two fingers hold a round part at two points and it can pivot about the line between them; closing along the
        way the hand travels puts the pull of every start and stop along that line instead of across it."""
        part.yaw = math.atan2(target_xy[1] - part.y, target_xy[0] - part.x) + offset
        ps = PlanningScene(is_diff=True)
        ps.world.collision_objects.append(part.collision_object())
        rclpy.spin_until_future_complete(self, self.scene.call_async(ApplyPlanningScene.Request(scene=ps)))

    def build_task(self, part, target_xy_yaw, surface_z, aligned=False, place_turn=0.0):
        transfer = self.transfer
        opening = self.pregrasp(part)
        arm_group, hand_group, eef = f"{self.arm}_arm", f"{self.arm}_gripper", f"{self.arm}_hand"
        lo, hi = self.cfg["approach"]
        task = core.Task()
        task.name = f"kit {part.name} ({part.cls})"
        # one robot model for all tasks: the planners keep the model of their first task
        if self.robot_model is None:
            task.loadRobotModel(self.mtc_node)
            self.robot_model = task.getRobotModel()
        else:
            task.setRobotModel(self.robot_model)
        task.add(stages.CurrentState("current"))

        open_hand = stages.MoveTo("open hand", self.joint_interp)
        open_hand.group = hand_group
        open_hand.setGoal(opening)
        task.add(open_hand)

        task.add(stages.Connect("move to pick", [(arm_group, transfer)]))

        grasps = stages.GenerateGraspPose("grasp poses")
        # grasps around the part: a round part aligned with its way (align_round_part) only closing along it, else
        # from any side; any other part only where the fingers close across its width, which repeats with its yaw
        # symmetry (the connector every 180 deg, the block every 90 deg)
        symmetry = part.spec.get("yaw_symmetry", math.pi)
        if aligned:
            grasps.angle_delta = math.pi
        else:
            grasps.angle_delta = math.pi / 4 if symmetry == 0 else min(symmetry, math.pi)
        grasps.pregrasp, grasps.grasp = opening, "closed"
        grasps.setMonitoredStage(task["open hand"])
        grasp = stages.SimpleGrasp(grasps, "grasp")
        grasp.setIKFrame(self.ik_frame(part))
        pick = stages.Pick(grasp, "pick")
        pick.eef, pick.object = eef, part.name
        pick.setApproachMotion(twist("world", -1.0), lo, hi)
        pick.setLiftMotion(twist("world", 1.0), lo, hi)
        task.add(pick)

        task.add(stages.Connect("move to place", [(arm_group, transfer)]))

        x, y, yaw = target_xy_yaw
        if aligned:
            yaw = part.yaw                                      # round: the hand keeps its heading to the pocket
        yaw += place_turn                                       # a symmetric part fits its pocket turned, too
        places = stages.GeneratePlacePose("place poses")
        places.setMonitoredStage(task["pick"])
        places.object = part.name
        places.pose = yaw_pose(x, y, surface_z + part.height / 2 + 0.002, yaw)
        release = stages.SimpleUnGrasp(places, "release")
        release.properties["pregrasp"] = opening
        place = stages.Place(release, "place")
        place.eef, place.object = eef, part.name
        place.eef_frame = f"openarm_{self.arm}_grasp_frame"
        place.setPlaceMotion(twist("world", -1.0), lo, hi)
        place.setRetractMotion(twist("world", 1.0), lo, hi)
        task.add(place)

        home = stages.MoveTo("return", transfer)
        home.group = arm_group
        home.setGoal("ready")
        task.add(home)
        return task

    def target_for(self, part):
        if part.cls in self.kit_free:
            return "kit", self.kit_free[part.cls], self.cfg["kit_surface_z"]
        if self.reject_free:
            x, y = self.reject_free[0]
            return "reject", (x, y, 0.0), self.cfg["reject_surface_z"]
        return None, None, None

    def clear_scene(self):
        """Remove the parts an earlier run left in move_group's planning scene (detected or placed)."""
        self.scene.wait_for_service()
        get = self.create_client(GetPlanningScene, "/get_planning_scene")
        get.wait_for_service()
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        future = get.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        ps = PlanningScene(is_diff=True)
        for co in future.result().scene.world.collision_objects:
            if co.id.startswith(("part_", "placed_")):
                ps.world.collision_objects.append(CollisionObject(id=co.id, operation=CollisionObject.REMOVE,
                                                                  header=Header(frame_id="world")))
        if ps.world.collision_objects:
            rclpy.spin_until_future_complete(self, self.scene.call_async(ApplyPlanningScene.Request(scene=ps)))

    @staticmethod
    def still_there(part, seen_before, parts):
        """True if the picked part is still in the inbound zone: a part of its class within 30 mm of where it was
        picked, or no fewer parts of its class than before the pick (it slipped and rolled further away)."""
        same = [p for p in parts if p.cls == part.cls]
        return len(same) >= seen_before or any(math.hypot(p.x - part.x, p.y - part.y) < 0.03 for p in same)

    def commit(self, part, kind, target, surface, done):
        """The part is where it was placed: its slot is taken and a kitted part stays in the scene as an obstacle
        at its pocket (a rejected part lies inside the bin, whose walls are in the robot model already)."""
        done.append((part.cls, kind))
        if kind == "kit":
            del self.kit_free[part.cls]
            part.x, part.y, part.yaw = target
            part.z = surface + part.height / 2
            self.keep_in_scene(part, f"placed_{len(done)}")
        else:
            self.reject_free.pop(0)

    def run(self):
        done, cycle, pending, retries = [], 0, None, 0
        self.clear_scene()
        t_start = self.get_clock().now()
        while rclpy.ok() and (not self.cycles or cycle < self.cycles):
            cycle += 1
            self.say("detecting parts")
            parts = self.wait_for_parts()
            if pending is not None:
                # the last pick counts once the camera no longer sees the part where it was picked up
                (part, kind, target, surface), seen_before = pending
                pending = None
                if self.still_there(part, seen_before, parts):
                    retries += 1
                    self.get_logger().warn(f"{part.name} ({part.cls}) is still in the inbound zone: "
                                           f"the grasp slipped, retry {retries} of {self.max_retries}")
                    self.say(f"{part.cls} slipped, retrying")
                    if retries > self.max_retries:
                        self.get_logger().error("too many failed grasps, stopping")
                        break
                else:
                    retries = 0
                    self.commit(part, kind, target, surface, done)
            self.apply_scene(parts)
            self.get_logger().info(f"cycle {cycle}: {len(parts)} parts in the inbound zone: "
                                   + ", ".join(f"{p.cls} at {p.x:.3f},{p.y:.3f}" for p in parts))
            # kit parts first, nearest to the robot's right shoulder first
            parts.sort(key=lambda p: (p.cls not in self.kit_free, math.hypot(p.x, p.y + 0.15)))
            picked = None
            for part in parts:
                kind, target, surface = self.target_for(part)
                if kind is None:
                    continue
                t0 = time.monotonic()
                round_part = part.spec.get("yaw_symmetry", math.pi) == 0
                # round parts: closing along the way first, then up to 60 deg off it, then from any side;
                # other parts: into the pocket as given, then turned by their symmetry
                symmetry = part.spec.get("yaw_symmetry", math.pi)
                tries = [math.radians(a) for a in (0, 30, -30, 60, -60)] if round_part else []
                turns = [0.0] if round_part else [k * symmetry for k in range(round(2 * math.pi / symmetry))]
                for offset, turn in [(o, 0.0) for o in tries] + [(None, t) for t in turns]:
                    aligned = offset is not None
                    if aligned:
                        self.align_round_part(part, target[:2], offset)
                    task = self.build_task(part, target, surface, aligned, turn)
                    try:
                        task.plan(max_solutions=4)
                    except Exception as e:                         # noqa: BLE001  (MTC raises on init errors)
                        self.get_logger().error(f"{task.name}: {e}")
                    # a path that doubles back can leave a segment without timing; the controller rejects it
                    solution = next((sol for sol in task.solutions if timing_ok(sol)), None)
                    if solution is not None:
                        break
                    if aligned and offset == tries[-1]:
                        self.get_logger().info(f"{task.name}: no plan closing along the way, trying all sides")
                dt = time.monotonic() - t0
                if solution is None:
                    self.get_logger().warn(f"{task.name}: no plan in {dt:.1f} s, trying the next part")
                    self.get_logger().info(f"{task.name}: " + stage_summary(task))
                    continue
                way = f", closing {math.degrees(offset):+.0f} deg off its way" if offset is not None else ""
                self.get_logger().info(f"{task.name} → {kind}: planned in {dt:.1f} s, cost {solution.cost:.2f}{way}")
                self.say(f"{part.cls} → {'kit' if kind == 'kit' else 'reject bin'}")
                task.publish(solution)
                if self.execute:
                    result = task.execute(solution)
                    if result.val != 1:
                        self.get_logger().error(f"{task.name}: execution failed ({result.val})")
                        return done
                picked = (part, kind, target, surface)
                break
            if picked is None:
                self.get_logger().info("nothing left that can be picked")
                break
            if self.simulated or not self.execute:
                if self.simulated:
                    del self.layout[int(picked[0].name.split("_")[1])]
                self.commit(*picked, done)
            else:
                pending = picked, sum(p.cls == picked[0].cls for p in parts)
        elapsed = (self.get_clock().now() - t_start).nanoseconds * 1e-9
        self.get_logger().info(f"done: {len(done)} parts placed in {elapsed:.1f} s: "
                               + ", ".join(f"{c} → {k}" for c, k in done))
        self.say(f"kit complete: {len(done)} parts placed" if done else "nothing to pick")
        return done


def stage_summary(task):
    """Solutions and failures of the task's top-level stages, for a plan that found nothing."""
    out = []
    for name in ("open hand", "move to pick", "pick", "move to place", "place", "return"):
        try:
            st = task[name]
        except Exception:                                          # noqa: BLE001
            continue
        fails = list(st.failures)
        why = fails[0].comment if fails and getattr(fails[0], "comment", "") else ""
        out.append(f"{name} {len(st.solutions)}/{len(fails)}{' (' + why + ')' if why else ''}")
    return ", ".join(out)


def main():
    import sys
    rclcpp.init(sys.argv)                             # the MoveIt parameters reach the MTC node from the launch file
    rclpy.init()
    node = Kitting()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
