"""Mirror MoveIt's attach and detach into the Isaac Sim scene (hardware:=isaac).

MoveIt attaches a part to the hand when the fingers have closed on it and detaches it when they open. This node
watches the attached objects in move_group's planning scene and sends "ATTACH <arm> <part>" / "DETACH <part>" on
/isaac_attach_cmd. The scene's part is the one whose ground-truth pose (TF world → gt/<part>) is nearest to the
hand at that moment. MoveIt reports the detach only after the retreat that follows; the part is released as soon
as ros2_control commands the fingers open instead (/joint_command), so it is set down in its pocket and not
dropped from above. The commanded target is used, not the measured finger angle: a finger pressed against a held
part springs back by a few hundredths of a radian, which is not the hand opening.
"""
import math

from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

OPEN = 0.1                                                  # rad: a finger target this far from closed is "open"


class AttachBridge(Node):

    def __init__(self):
        super().__init__("isaac_attach_bridge")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(String, "/isaac_attach_cmd", 10)
        self.scene = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.attached = {}                                  # planning-scene id → scene part name
        self.held = {}                                      # planning-scene id → arm
        self.target = {}                                    # arm → commanded finger opening (rad, 0 = closed)
        self.pending = None
        self.create_subscription(JointState, "/joint_command", self.on_command, 10)
        self.create_timer(0.05, self.poll)

    def on_command(self, msg):
        for arm in ("left", "right"):
            name = f"openarm_{arm}_finger_joint1"
            if name in msg.name:
                self.target[arm] = abs(msg.position[msg.name.index(name)])
        for oid, arm in list(self.held.items()):
            if self.target.get(arm, 0.0) > OPEN:                # the hand is told to open
                del self.held[oid]
                self.send(f"DETACH {self.attached[oid]}")

    def poll(self):
        if self.pending is not None or not self.scene.service_is_ready():
            return
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        self.pending = self.scene.call_async(req)
        self.pending.add_done_callback(self.on_scene)

    def on_scene(self, future):
        self.pending = None
        now = {a.object.id: a for a in future.result().scene.robot_state.attached_collision_objects}
        for oid, aco in now.items():
            if oid in self.attached:
                continue
            arm = "left" if "left" in aco.link_name else "right"
            part = self.nearest_part(aco.link_name)
            if part:
                self.attached[oid] = part
                self.held[oid] = arm
                self.send(f"ATTACH {arm} {part}")
        for oid in [o for o in self.attached if o not in now]:
            part = self.attached.pop(oid)
            if self.held.pop(oid, None) is not None:            # not yet released by the fingers opening
                self.send(f"DETACH {part}")

    def nearest_part(self, hand_link):
        """The simulated part closest to the hand's grasp point."""
        try:
            hand = self.tf.lookup_transform("world", hand_link, Time()).transform.translation
        except Exception:                                   # noqa: BLE001
            return None
        best, best_d = None, 0.08
        for line in self.tf.all_frames_as_string().splitlines():
            if not line.startswith("Frame gt/"):
                continue
            name = line.split()[1][3:]
            p = self.tf.lookup_transform("world", "gt/" + name, Time()).transform.translation
            d = math.dist((p.x, p.y, p.z), (hand.x, hand.y, hand.z))
            if d < best_d:
                best, best_d = name, d
        if best is None:
            self.get_logger().warn(f"no simulated part within 8 cm of {hand_link}")
        return best

    def send(self, text):
        self.pub.publish(String(data=text))
        self.get_logger().info(text)


def main():
    rclpy.init()
    rclpy.spin(AttachBridge())
