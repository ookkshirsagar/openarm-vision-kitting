"""Detects the parts on the worktop from one RGB-D camera and publishes their class and pose.

  ros2 run openarm_kitting_perception part_detector --ros-args --params-file <kitting.yaml> -p camera:=ext_zed

Subscribes  /<camera>/depth/depth_registered, /<camera>/rgb/camera_info, /<camera>/rgb/image_rect_color
            /<camera>/instance/image (segmentation:=ground_truth only), TF world → camera optical frame
Publishes   /kitting/detections          vision_msgs/Detection3DArray in world (id = class, bbox = pose and size)
            /kitting/detections_markers  boxes and labels for RViz
            /kitting/detections_image    the colour image with the detections drawn in

segmentation:=depth         everything that stands on the worktop in the search region (no learning, no labels)
segmentation:=ground_truth  the simulator's instance mask (a reference to score the other methods against)
"""
import math

import cv2
from geometry_msgs.msg import Point
import numpy as np
from openarm_kitting_perception.detection import detect_parts, PartClass
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

COLOURS = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (204, 121, 167), (213, 94, 0)]


def image_to_array(msg):
    dtype, channels = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4),
                       "32FC1": (np.float32, 1), "32SC1": (np.int32, 1)}[msg.encoding]
    a = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
    return a[..., 0] if channels == 1 else a


def transform_to_matrix(t):
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [p.x, p.y, p.z]
    return m


class PartDetector(Node):
    def __init__(self):
        super().__init__("part_detector")
        p = self.declare_parameter
        self.camera = p("camera", "ext_zed").value
        self.segmentation = p("segmentation", "depth").value
        self.work_z = p("work_z", 0.364).value
        self.region = list(p("search_region", [0.132, 0.328, -0.438, -0.182]).value)
        self.min_score = p("min_score", 0.3).value
        self.catalogue = []
        for name in p("part_classes", ["connector", "block", "pinion"]).value:
            shape = p(f"parts.{name}.shape", "box").value
            size = list(p(f"parts.{name}.size", [0.03, 0.05, 0.02]).value)
            axis = p(f"parts.{name}.axis", "y").value
            if shape == "box":
                self.catalogue.append(PartClass(name, max(size[:2]), min(size[:2]), size[2]))
            elif axis == "z":                                    # upright cylinder
                self.catalogue.append(PartClass(name, size[0], size[0], size[1], is_round=True))
            else:                                                # cylinder on its side
                self.catalogue.append(PartClass(name, size[1], size[0], size[0]))
        self.class_index = {c.name: i for i, c in enumerate(self.catalogue)}

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.k = self.frame = self.rgb = self.instances = None
        ns = f"/{self.camera}"
        self.create_subscription(CameraInfo, f"{ns}/rgb/camera_info", self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, f"{ns}/rgb/image_rect_color", self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, f"{ns}/depth/depth_registered", self.on_depth, qos_profile_sensor_data)
        if self.segmentation == "ground_truth":
            self.create_subscription(Image, f"{ns}/instance/image", self.on_instances, qos_profile_sensor_data)
        self.pub = self.create_publisher(Detection3DArray, "/kitting/detections", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/kitting/detections_markers", 10)
        self.pub_image = self.create_publisher(Image, "/kitting/detections_image", 10)
        self.get_logger().info(f"camera {self.camera}, segmentation {self.segmentation}, "
                               f"classes {', '.join(c.name for c in self.catalogue)}")

    def on_info(self, msg):
        self.k = np.array(msg.k).reshape(3, 3)
        self.frame = msg.header.frame_id

    def on_rgb(self, msg):
        self.rgb = msg

    def on_instances(self, msg):
        self.instances = image_to_array(msg)

    def on_depth(self, msg):
        if self.k is None:
            return
        try:
            world_from_camera = transform_to_matrix(self.tf.lookup_transform("world", self.frame, rclpy.time.Time()))
        except TransformException as e:
            self.get_logger().warn(f"no transform world → {self.frame}: {e}", throttle_duration_sec=5.0)
            return
        masks = None
        if self.segmentation == "ground_truth":
            if self.instances is None:
                return
            # ids 0 and 1 are background and unlabelled; the region and height filter drops everything but parts
            masks = [self.instances == i for i in np.unique(self.instances) if i > 1]
        detections = detect_parts(image_to_array(msg), self.k, world_from_camera, self.catalogue, self.region,
                                  self.work_z, masks=masks, min_score=self.min_score)
        if masks is not None:
            r = self.region
            detections = [d for d in detections if r[0] < d.position[0] < r[1] and r[2] < d.position[1] < r[3]]
        self.publish(detections, msg.header.stamp)

    def publish(self, detections, stamp):
        out, markers = Detection3DArray(), MarkerArray()
        out.header.frame_id = "world"
        out.header.stamp = stamp
        markers.markers.append(Marker(action=Marker.DELETEALL))
        for i, d in enumerate(detections):
            qz, qw = math.sin(d.yaw / 2), math.cos(d.yaw / 2)
            det = Detection3D()
            det.header = out.header
            det.id = f"{d.cls}_{i}"
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = d.cls, d.score
            det.results.append(hyp)
            c = det.bbox.center
            c.position.x, c.position.y, c.position.z = map(float, d.position)
            c.orientation.z, c.orientation.w = qz, qw
            hyp.pose.pose = c
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = float(d.size[1]), float(d.size[0]), float(d.size[2])
            out.detections.append(det)

            r, g, b = COLOURS[self.class_index[d.cls] % len(COLOURS)]
            box = Marker(type=Marker.CUBE, ns="parts", id=i)
            box.header = out.header
            box.pose, box.scale = c, det.bbox.size
            box.color.r, box.color.g, box.color.b, box.color.a = r / 255, g / 255, b / 255, 0.8
            text = Marker(type=Marker.TEXT_VIEW_FACING, ns="labels", id=i, text=f"{d.cls} {d.score:.2f}")
            text.header = out.header
            text.pose.position = Point(x=c.position.x, y=c.position.y, z=c.position.z + 0.05)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.02
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            markers.markers += [box, text]
        self.pub.publish(out)
        self.pub_markers.publish(markers)
        if self.rgb is not None and self.pub_image.get_subscription_count():
            self.publish_image(detections)

    def publish_image(self, detections):
        img = np.ascontiguousarray(image_to_array(self.rgb)[..., :3])
        for d in detections:
            colour = COLOURS[self.class_index[d.cls] % len(COLOURS)]
            contours, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, colour, 2)
            x, y, w, _ = cv2.boundingRect(np.vstack(contours))
            cv2.putText(img, f"{d.cls} {d.score:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        out = Image(height=img.shape[0], width=img.shape[1], encoding="rgb8", step=img.shape[1] * 3,
                    data=img.tobytes())
        out.header = self.rgb.header
        self.pub_image.publish(out)


def main():
    rclpy.init()
    node = PartDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
