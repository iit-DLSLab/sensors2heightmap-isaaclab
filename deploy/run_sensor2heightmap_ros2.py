"""Runs the terrain reconstructor on the robot and publishes the heightmap on /height_scan_markers, the topic
read by basic-locomotion's run_controller_ros2.py.

    python deploy/run_sensor2heightmap_ros2.py --model <model>.ts [--robot go2]
"""

import os

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from dls2_interface.msg import BlindState, Imu
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robots  # noqa: E402
from sensors import lidar_grid  # noqa: E402
from sensors.sensor_model import TOPICS  # noqa: E402
from terrain_reconstructor import TerrainReconstructor  # noqa: E402


class Sensor2HeightmapROS2(Node):
    def __init__(self, model: str, robot: str, topic: str | None = None):
        super().__init__("Sensor2HeightmapROS2")
        self.reconstructor = TerrainReconstructor(model, robots.load(robot).MESSAGE_JOINT_NAMES)
        config = self.reconstructor.config
        self.imu = self.joints = None
        self.sensor_arrived = False
        self._last_warning_time = 0.0
        heightmap = config["heightmap"]
        rows, cols = heightmap["shape"]
        x = heightmap["offset"][0] - heightmap["size"][0] / 2 + heightmap["resolution"] * np.arange(cols)
        y = heightmap["offset"][1] - heightmap["size"][1] / 2 + heightmap["resolution"] * np.arange(rows)
        self.cells = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)

        topic = topic or TOPICS[self.reconstructor.kind]
        if self.reconstructor.kind == "lidar":
            self.create_subscription(PointCloud2, topic, self.lidar_callback, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, topic, self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, "/imu", self.imu_callback, 1)
        self.create_subscription(BlindState, "/blind_state_legged", self.joints_callback, 1)
        self.publisher_heightmap = self.create_publisher(MarkerArray, "/height_scan_markers", 1)
        self.create_timer(config["step_dt"], self.step)
        self.get_logger().info(
            f"{model}: {config['sensor_model']} on {topic} -> {rows}x{cols} heightmap on /height_scan_markers"
        )

    def lidar_callback(self, msg):
        offsets = {field.name: field.offset for field in msg.fields}
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        points = np.stack([data[:, offsets[axis]:offsets[axis] + 4].copy().view(np.float32)[:, 0] for axis in "xyz"], axis=1)
        # the L2 publishes at the end of its scan: time the message from its arrival, not from the robot's stamp
        self.reconstructor.sensor_message(points, time.monotonic() - self.reconstructor.message_period)
        self._sensor_arrived(f"the first LiDAR message ({len(points)} points)")

    def depth_callback(self, msg):
        if msg.encoding != "16UC1":
            now = time.monotonic()
            if now - self._last_warning_time >= 2.0:
                self.get_logger().warning(f"Ignoring a {msg.encoding} depth image: expected 16UC1 (mm)")
                self._last_warning_time = now
            return
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        self.reconstructor.sensor_message(depth.astype(np.float32) * 0.001, time.monotonic())
        self._sensor_arrived(f"the first depth image ({msg.width}x{msg.height})")

    def _sensor_arrived(self, what: str) -> None:
        if not self.sensor_arrived:
            self.get_logger().info(f"Received {what}")
        self.sensor_arrived = True

    def imu_callback(self, msg):
        self.imu = (np.array(msg.angular_velocity), np.array(msg.orientation))

    def joints_callback(self, msg):
        self.joints = (np.array(msg.joints_position), np.array(msg.joints_velocity))

    def step(self):
        if self.imu is None or self.joints is None or not self.sensor_arrived:
            return
        angular_velocity, (x, y, z, w) = self.imu
        projected_gravity = -np.array((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))  # -R^T z
        heights = self.reconstructor.step(angular_velocity, projected_gravity, *self.joints, time.monotonic()).numpy()
        # cells in base_link: the controller adds the base pose and takes their z
        frame = lidar_grid.yaw_frame(torch.tensor(projected_gravity[None], dtype=torch.float32))[0].numpy()
        points = np.column_stack((self.cells, -heights - lidar_grid.HEIGHT_OFFSET)) @ frame.T
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for index, point in enumerate(points.tolist()):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = "height_scan"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = point
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.02
            marker.color.b = marker.color.a = 1.0
            markers.markers.append(marker)
        self.publisher_heightmap.publish(markers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Reconstructor exported by deploy/terrain_reconstructor.py (.ts).")
    parser.add_argument("--robot", default="go2", help="Robot, a module of robots/: the order of its joints.")
    parser.add_argument("--topic", default=None, help="Sensor topic, if not the default of the sensor kind.")
    args = parser.parse_args()

    rclpy.init()
    node = Sensor2HeightmapROS2(args.model, args.robot, args.topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
