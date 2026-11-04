"""Writes or checks robots/<robot>.py from the robot's URDF, its USD and a rosbag of it standing on flat ground.
See README.md."""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import sqlite3
import struct
import sys
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

import numpy as np
import torch

ROBOTS = Path(__file__).resolve().parent
sys.path[:0] = [str(ROBOTS.parent), str(ROBOTS.parent / "deploy")]
import sensors  # noqa: E402
from sensors.sensor_model import TOPICS  # noqa: E402
from terrain_reconstructor import decimate_depth  # noqa: E402

ENV_ROBOT = "/World/envs/env_.*/Robot"
MESSAGE_TYPES = {"lidar": "sensor_msgs/msg/PointCloud2", "depth": "sensor_msgs/msg/Image"}
LOWSTATE_JOINTS = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
X = np.array([1.0, 0.0, 0.0])


def rotation_rpy(rpy) -> np.ndarray:
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rotation_vector(vector) -> np.ndarray:
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def rotation_xyzw(quaternion) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    m = rotation
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2], np.trace(m)]))
    if i == 3:
        w = np.sqrt(1.0 + np.trace(m)) / 2
        q = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], 4 * w * w]) / (4 * w)
    else:
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s, (m[k, j] - m[j, k]) / s
    return q if q[3] >= 0 else -q


def heading_frame(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Columns: forward projected on the plane normal to up, left, up."""
    forward = forward - (forward @ up) * up
    forward = forward / np.linalg.norm(forward)
    return np.stack((forward, np.cross(up, forward), up), axis=1)


def aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = np.cross(a, b)
    if not axis.any():
        return np.eye(3)
    return rotation_vector(axis / np.linalg.norm(axis) * np.arctan2(np.linalg.norm(axis), a @ b))


class Joint(NamedTuple):
    name: str
    type: str
    parent: str
    origin: np.ndarray
    axis: np.ndarray


def _floats(element, key: str, default: str = "0 0 0") -> np.ndarray:
    text = element.get(key) if element is not None else None
    return np.array([float(value) for value in (text or default).split()])


class Urdf:
    """Minimal URDF reader (plain URDF, not xacro)."""

    def __init__(self, path: str):
        robot = ElementTree.parse(path).getroot()
        self.links = {link.get("name"): link for link in robot.findall("link")}
        self.joints = {}
        for joint in robot.findall("joint"):
            origin = np.eye(4)
            origin[:3, :3] = rotation_rpy(_floats(joint.find("origin"), "rpy"))
            origin[:3, 3] = _floats(joint.find("origin"), "xyz")
            axis = _floats(joint.find("axis"), "xyz", "1 0 0")
            self.joints[joint.find("child").get("link")] = Joint(
                joint.get("name"), joint.get("type"), joint.find("parent").get("link"), origin,
                axis / max(np.linalg.norm(axis), 1e-12),
            )
        self.root = next(link for link in self.links if link not in self.joints)

    def pose(self, link: str, base: str, angles: dict | None = None) -> np.ndarray:
        """Pose of a link in the frame of base, at the given joint angles."""

        def from_root(link: str) -> np.ndarray:
            pose = np.eye(4)
            while link != self.root:
                joint = self.joints[link]
                motion, angle = np.eye(4), (angles or {}).get(joint.name, 0.0)
                if joint.type in ("revolute", "continuous"):
                    motion[:3, :3] = rotation_vector(joint.axis * angle)
                elif joint.type == "prismatic":
                    motion[:3, 3] = joint.axis * angle
                pose, link = joint.origin @ motion @ pose, joint.parent
            return pose

        return np.linalg.inv(from_root(base)) @ from_root(link)

    def moving_joints(self) -> list[str]:
        """Actuated joints, leg by leg."""
        children = {}
        for child, joint in self.joints.items():
            children.setdefault(joint.parent, []).append(child)
        names, stack = [], [self.root]
        while stack:
            link = stack.pop()
            if link in self.joints and self.joints[link].type in ("revolute", "continuous", "prismatic"):
                names.append(self.joints[link].name)
            stack.extend(sorted(children.get(link, []), reverse=True))
        return names

    def base_height(self, base: str, angles: dict, up: np.ndarray) -> float | None:
        """Height of the base above the ground, from the feet (links named *foot*)."""
        bottoms = []
        for link, element in self.links.items():
            if "foot" in link.lower():
                sphere = element.find("collision/geometry/sphere")
                offset = np.append(_floats(element.find("collision/origin"), "xyz"), 1.0)
                centre = self.pose(link, base, angles) @ offset
                bottoms.append(centre[:3] @ up - (float(sphere.get("radius")) if sphere is not None else 0.0))
        return -float(np.mean(bottoms)) if bottoms else None


def usd_robot(path: str) -> dict:
    """Base body and visual meshes of the robot's USD, as prim expressions for the simulation."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    robot = {"base": bodies[0].GetName(), "base_path": ENV_ROBOT + str(bodies[0].GetPath())[len(str(root.GetPath())):],
             "base_meshes": [], "leg_meshes": [], "base_boxes": []}
    for body in bodies:
        names, prims = [], iter(Usd.PrimRange(body))
        next(prims)
        for prim in prims:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prims.PruneChildren()
            elif prim.GetTypeName() == "Mesh" and UsdGeom.Imageable(prim).ComputePurpose() in ("default", "render"):
                names.append(str(prim.GetPath())[len(str(body.GetPath())) + 1:])
                if body == bodies[0]:
                    robot["base_boxes"].append(boxes.ComputeRelativeBound(prim, body).ComputeAlignedRange())
        if names:
            prefix = ENV_ROBOT + str(body.GetPath())[len(str(root.GetPath())):]
            alternatives = "|".join(re.escape(name) for name in names)
            expression = f"{prefix}/{alternatives}" if len(names) == 1 else f"{prefix}/({alternatives})"
            robot["base_meshes" if body == bodies[0] else "leg_meshes"].append(expression)
    return robot


class _Cdr:
    """Little-endian CDR reader."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 4

    def read(self, fmt: str):
        size = struct.calcsize(fmt)
        self.pos += -(self.pos - 4) % min(size, 8)
        value = struct.unpack_from("<" + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def string(self) -> str:
        length = self.read("I")
        value = self.data[self.pos:self.pos + length - 1].decode()
        self.pos += length
        return value

    def header(self) -> None:
        self.read("i"), self.read("I"), self.string()

    def dls2_header(self) -> None:
        self.string(), self.read("I"), self.read("Q")


class Bag:
    """rosbag2 reader (sqlite3 storage), no ROS needed."""

    def __init__(self, path: str):
        files = [path] if path.endswith(".db3") else sorted(glob.glob(os.path.join(path, "*.db3")))
        if not files:
            raise FileNotFoundError(f"No .db3 file in {path}: the rosbag must use the sqlite3 storage.")
        self.database = sqlite3.connect(files[0])
        self.types = dict(self.database.execute("select name, type from topics"))

    def messages(self, topic: str) -> list[bytes]:
        query = "select data from messages join topics on topic_id = topics.id where name = ? order by timestamp"
        return [data for (data,) in self.database.execute(query, (topic,))]

    def topic(self, message_type: str, preferred: str | None = None) -> str | None:
        topics = [name for name, kind in self.types.items() if kind == message_type]
        return preferred if preferred in topics else (topics[0] if len(topics) == 1 else None)


def cloud_points(data: bytes) -> np.ndarray:
    cdr = _Cdr(data)
    cdr.header()
    height, width = cdr.read("I"), cdr.read("I")
    offsets = {}
    for _ in range(cdr.read("I")):
        name, offset = cdr.string(), cdr.read("I")
        cdr.read("B"), cdr.read("I")
        offsets[name] = offset
    cdr.read("B")
    point_step, _, size = cdr.read("I"), cdr.read("I"), cdr.read("I")
    raw = np.frombuffer(cdr.data[cdr.pos:cdr.pos + size], dtype=np.uint8).reshape(height * width, point_step)
    points = np.stack([raw[:, offsets[a]:offsets[a] + 4].copy().view(np.float32)[:, 0] for a in "xyz"], axis=1)
    return points[np.isfinite(points).all(axis=1)].astype(np.float64)


def depth_points(data: bytes, camera) -> np.ndarray:
    """Depth image decimated as on the robot and back-projected with the model's camera (ROS optical frame)."""
    cdr = _Cdr(data)
    cdr.header()
    height, width, encoding = cdr.read("I"), cdr.read("I"), cdr.string()
    cdr.read("B")
    step, size = cdr.read("I"), cdr.read("I")
    if encoding != "16UC1":
        raise ValueError(f"A {encoding} depth image: expected 16UC1 (mm).")
    image = np.frombuffer(cdr.data[cdr.pos:cdr.pos + size], dtype="<u2").reshape(height, step // 2)[:, :width]
    depth = decimate_depth(torch.from_numpy(image.astype(np.float32) * 0.001), camera.height, camera.width).numpy()
    v, u = np.nonzero(depth > 0.0)
    z = depth[v, u].astype(np.float64)
    pixel = lambda index, size: (index + 0.5 - size / 2) / camera.focal_length  # noqa: E731
    return np.stack((pixel(u, camera.width) * z, pixel(v, camera.height) * z, z), axis=1)


def lowstate(data: bytes) -> tuple[np.ndarray, dict]:
    """IMU up vector and joint angles of a unitree_go/LowState."""
    cdr = _Cdr(data)
    for fmt in "BBBB" "II" "II" "H":
        cdr.read(fmt)
    w, x, y, z = (cdr.read("f") for _ in range(4))
    for _ in range(9):
        cdr.read("f")
    cdr.read("b")
    angles = []
    for _ in range(20):
        cdr.read("B")
        angles.append(cdr.read("f"))
        for _ in range(6):
            cdr.read("f")
        cdr.read("b")
        for _ in range(3):
            cdr.read("I")
    return rotation_xyzw((x, y, z, w))[2], dict(zip(LOWSTATE_JOINTS, angles))


def dls2_up(data: bytes) -> np.ndarray:
    cdr = _Cdr(data)
    cdr.dls2_header()
    return rotation_xyzw([cdr.read("d") for _ in range(4)])[2]


def dls2_joints(data: bytes) -> tuple[list[str], list[float]]:
    cdr = _Cdr(data)
    cdr.dls2_header()
    cdr.string()
    names = [cdr.string() for _ in range(cdr.read("I"))]
    return names, [cdr.read("d") for _ in range(cdr.read("I"))]


def standing_state(bag: Bag, joint_names: list[str] | None) -> tuple[np.ndarray, dict, list[str] | None, str]:
    """Mean IMU up vector and joint angles over the bag."""
    topic = bag.topic("unitree_go/msg/LowState", "/lowstate")
    if topic:
        states = [lowstate(data) for data in bag.messages(topic)]
        ups, angles, names, topics = [up for up, _ in states], [state for _, state in states], None, topic
    else:
        imu = bag.topic("dls2_interface/msg/Imu", "/imu")
        joints = bag.topic("dls2_interface/msg/BlindState", "/blind_state_legged")
        if not imu:
            print("[WARNING] No IMU in the bag (/lowstate, /imu): the robot is taken as level.")
        ups = [dls2_up(data) for data in bag.messages(imu)] if imu else [np.array([0.0, 0.0, 1.0])]
        messages = [dls2_joints(data) for data in bag.messages(joints)] if joints else []
        names = messages[0][0] if messages and messages[0][0] else None
        angles = [dict(zip(names or joint_names or [], values)) for _, values in messages]
        topics = ", ".join(topic for topic in (imu, joints) if topic)
    ups = np.array(ups)
    if ups.std(axis=0).max() > 0.005 or (angles and np.std([list(a.values()) for a in angles], axis=0).max() > 0.01):
        print("[WARNING] The robot moves during the bag: the check needs it standing still on flat ground.")
    mean_angles = {name: float(np.mean([a[name] for a in angles])) for name in angles[0]} if angles else {}
    return ups.mean(axis=0) / np.linalg.norm(ups.mean(axis=0)), mean_angles, names, topics


def ground_plane(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    """Floor plane (RANSAC + least squares): normal in the sensor frame, sensor height, floor points."""
    points = points[np.linalg.norm(points, axis=1) < 2.5]
    sample = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    best = (0, None, None)
    for _ in range(1000):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) > 1e-9:
            normal /= np.linalg.norm(normal)
            count = int((np.abs((sample - a) @ normal) < 0.02).sum())
            best = max(best, (count, a, normal), key=lambda candidate: candidate[0])
    _, a, normal = best
    ground = points[np.abs((points - a) @ normal) < 0.02]
    centre = ground.mean(axis=0)
    normal = np.linalg.svd(ground - centre, full_matrices=False)[2][-1]
    height = -normal @ centre
    return (normal, height, ground) if height > 0 else (-normal, -height, ground)


def floor_seen(position, rotation, up_sensor, ground, up_base) -> tuple[float, float]:
    """Floor tilt (deg) and depth below the base (m) with a given mount."""
    tilt = np.degrees(np.arccos(np.clip((rotation @ up_sensor) @ up_base, -1.0, 1.0)))
    return tilt, -float(np.median((ground @ rotation.T + position) @ up_base))


def fitted_mount(reference, up_sensor, sensor_height, up_base, base_height):
    """Mount that puts the floor level.

    A flat floor doesn't show the rotation about the vertical: the sensor is assumed to lean forward or
    backward as in the reference mount, never sideways."""
    position, rotation = (np.array(value, dtype=float) for value in reference)
    axis = np.eye(3)[np.argmax(np.abs(rotation.T @ up_base))]
    lean = axis - (axis @ up_sensor) * up_sensor
    if np.linalg.norm(lean) < np.sin(np.radians(2.0)):
        rotation = aligning(rotation @ up_sensor, up_base) @ rotation
    else:
        side = 1.0 if (rotation @ axis)[0] >= 0.0 else -1.0
        rotation = heading_frame(side * X, up_base) @ heading_frame(lean, up_sensor).T
    if base_height is not None:
        position[2] = (sensor_height - base_height - up_base[:2] @ position[:2]) / up_base[2]
    return position, rotation


def check_sensor(bag: Bag, model: str, references: dict, up_base, base_height, topic, count: int, rng):
    """Checks one sensor of the bag, returns the fitted mount (None without data)."""
    sensor = sensors.load(model)
    if sensor.KIND == "depth" and not hasattr(sensor, "CAMERA"):
        raise ValueError(f"{sensor.__name__} has no CAMERA (height, width, focal_length): its images cannot be read.")
    topic = topic or bag.topic(MESSAGE_TYPES[sensor.KIND], TOPICS[sensor.KIND])
    if topic is None:
        print(f"\n{model}: no {MESSAGE_TYPES[sensor.KIND]} in the bag (--topic {model}=TOPIC): not checked.")
        return None
    messages = bag.messages(topic)
    messages = [messages[i] for i in np.linspace(0, len(messages) - 1, min(count, len(messages))).astype(int)]
    points = np.concatenate(
        [cloud_points(data) if sensor.KIND == "lidar" else depth_points(data, sensor.CAMERA) for data in messages]
    )
    up_sensor, sensor_height, ground = ground_plane(points, rng)
    label, reference = next(iter(references.items()))
    position, rotation = fitted_mount(reference, up_sensor, sensor_height, up_base, base_height)
    print(f"\n{model}: {topic}, {len(messages)} messages, {len(points)} points")
    print(f"  {'mount':<34}{'floor tilt':>12}{'floor below the base':>24}")
    for name, mount in {**references, "fitted on the bag": (position, rotation)}.items():
        tilt, depth = floor_seen(*mount, up_sensor, ground, up_base)
        print(f"  {name:<34}{tilt:>8.1f} deg{depth:>22.4f} m")
    axis = np.argmax(np.abs(rotation[2]))
    print(f"  the sensor {sensor_height:.4f} m above the floor, its {'xyz'[axis]} axis "
          f"{np.degrees(np.arccos(abs(rotation[2, axis]))):.1f} deg from the base's vertical")
    source = f"x, y from {label}; rotation " + (
        "and height fitted on the floor of the bag" if base_height is not None
        else "fitted on the floor of the bag, height of the reference (CHECK: no base height without --urdf)"
    )
    print(f"  fitted: {_mount_text(position, rotation)}; {source}")
    return position, rotation, source


def _mount_text(position, rotation) -> str:
    return (f'"position": ({", ".join(f"{value:.5f}" for value in position)}), '
            f'"rotation": ({", ".join(f"{value:.5f}" for value in quaternion_xyzw(rotation))})')


def _list(items: list[str]) -> str:
    return "[\n" + "".join(f'    "{item}",\n' for item in items) + "]" if items else "[]"


def write_robot(path: Path, sources: str, usd: dict | None, base: str, mounts: dict, joints: list, joints_source: str):
    sensors_text = ""
    for model, (position, rotation, source) in mounts.items():
        inside = usd is not None and any(box.Contains(tuple(position)) for box in usd["base_boxes"])
        meshes = "LEG_MESHES,  # its origin is inside BASE_MESHES" if inside else "BASE_MESHES + LEG_MESHES,"
        position_text, rotation_text = _mount_text(position, rotation).split(", \"rotation\"")
        sensors_text += (f'    "{model}": {{\n        # {source}\n        {position_text},\n'
                         f'        "rotation"{rotation_text},\n        "meshes": {meshes}\n    }},\n')
    base_path = usd["base_path"] if usd else f"{ENV_ROBOT}/Geometry/{base}"
    path.write_text(f'''"""The {path.stem} robot: its sensors' mounts, the meshes they see, its joints' order.
Written by robots/setup_robot.py from {sources}. Check the lines marked CHECK."""

# the body the sensors are attached to{"" if usd else "  # CHECK: as in basic-locomotion's USDs (no --usd)"}
BASE_PRIM_PATH = "{base_path}"

# Visual meshes of the robot in its USD, which the sensors see{"" if usd else " (CHECK: none without --usd)"}
BASE_MESHES = {_list(usd["base_meshes"] if usd else [])}
LEG_MESHES = {_list(usd["leg_meshes"] if usd else [])}

# sensor model (sensors/<name>.py) -> its mount, position (m) and rotation (quaternion x, y, z, w) of its frame in the
# base frame, and the robot's meshes it sees
SENSORS = {{
{sensors_text}}}

# the joints in the order of the robot's /blind_state_legged messages ({joints_source})
MESSAGE_JOINT_NAMES = {_list(joints)}
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", help="The robot's URDF (plain, not xacro).")
    parser.add_argument("--usd", help="The robot's USD (basic-locomotion's robot_model/<robot>/generated_usd).")
    parser.add_argument("--bag", help="rosbag2 (sqlite3) of the robot standing still on flat ground.")
    parser.add_argument("--sensor", action="append", default=[], help="MODEL=LINK: a model of sensors/, its URDF link.")
    parser.add_argument("--topic", action="append", default=[], help="MODEL=TOPIC, if not the default of its kind.")
    parser.add_argument("--messages", type=int, default=40, help="Sensor messages of the bag used (evenly spaced).")
    parser.add_argument("--name_robot", help="robots/<name>.py: checked if it exists, else written.")
    args = parser.parse_args()

    existing = None
    if args.name_robot:
        if not args.name_robot.isidentifier():
            parser.error(f"--name_robot {args.name_robot}: use a Python name (letters, digits, _).")
        if (ROBOTS / f"{args.name_robot}.py").exists():
            existing = importlib.import_module(f"robots.{args.name_robot}")
        elif not args.urdf:
            parser.error(f"robots/{args.name_robot}.py does not exist: writing it needs --urdf.")
    links = dict(item.split("=", 1) if "=" in item else (item, None) for item in args.sensor)
    links = links or dict.fromkeys(getattr(existing, "SENSORS", {}))
    if not links:
        parser.error("Name the sensors: --sensor MODEL=LINK.")

    urdf = Urdf(args.urdf) if args.urdf else None
    usd = usd_robot(args.usd) if args.usd else None
    base = None
    if urdf:
        base = usd["base"] if usd and usd["base"] in urdf.links else urdf.root
        print(f"URDF {args.urdf}: base link {base}, moving joints {urdf.moving_joints()}")
        if usd and base != usd["base"]:
            print(f"[WARNING] The USD's base body {usd['base']} is not a URDF link: the mounts are in the frame of the "
                  f"URDF's root {base}, CHECK it is the same.")
    if usd:
        print(f"USD {args.usd}: base {usd['base_path']}, {len(usd['base_meshes'])} + {len(usd['leg_meshes'])} "
              "mesh expressions (base + others)")

    mounts = {}
    for model, link in links.items():
        mounts[model] = {}
        if urdf and link:
            if link not in urdf.links:
                parser.error(f"--sensor {model}={link}: {link} is not a link of {args.urdf}.")
            pose = urdf.pose(link, base)
            mounts[model][f"URDF (link {link})"] = (pose[:3, 3], pose[:3, :3])
        if model in getattr(existing, "SENSORS", {}):
            mount = existing.SENSORS[model]
            mounts[model][f"robots/{args.name_robot}.py"] = (
                np.array(mount["position"]), rotation_xyzw(mount["rotation"])
            )
        if not mounts[model]:
            parser.error(f"No mount of {model} to check: give its URDF link (--sensor {model}=LINK).")

    joints, joints_source = (urdf.moving_joints() if urdf else None), "the URDF's order: CHECK against the robot"
    fitted = {}
    if args.bag:
        bag = Bag(args.bag)
        up_base, angles, message_joints, state_topics = standing_state(
            bag, getattr(existing, "MESSAGE_JOINT_NAMES", None) or joints
        )
        if message_joints:
            joints, joints_source = message_joints, f"as in {Path(args.bag).name}"
        if urdf and angles and not set(angles) & set(urdf.moving_joints()):
            print(f"[WARNING] The joints of the bag ({list(angles)[:3]}...) are not the URDF's: no base height.")
            angles = {}
        base_height = urdf.base_height(base, angles, up_base) if urdf and angles else None
        tilt = np.degrees(np.arccos(np.clip(up_base[2], -1.0, 1.0)))
        print(f"bag {args.bag}: {state_topics or 'no IMU or joints'}; base tilted {tilt:.1f} deg"
              + (f", {base_height:.4f} m above the floor (feet of the URDF)" if base_height is not None else ""))
        topics, rng = dict(item.split("=", 1) for item in args.topic), np.random.default_rng(0)
        for model, references in mounts.items():
            result = check_sensor(bag, model, references, up_base, base_height, topics.get(model), args.messages, rng)
            if result is not None:
                fitted[model] = result

    if existing:
        print(f"\nrobots/{args.name_robot}.py exists: checked, not written.")
    elif args.name_robot:
        urdf_source = "the URDF link {}: CHECK it is the frame of the sensor's data"
        chosen = {
            model: fitted.get(model) or (*mounts[model][f"URDF (link {link})"], urdf_source.format(link))
            for model, link in links.items()
        }
        path = ROBOTS / f"{args.name_robot}.py"
        sources = ", ".join(Path(name).name for name in (args.urdf, args.usd, args.bag) if name)
        write_robot(path, sources, usd, base, chosen, joints, joints_source)
        print(f"\nWrote {path}: check the lines marked CHECK.")


if __name__ == "__main__":
    main()
