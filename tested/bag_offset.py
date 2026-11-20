"""The reconstructor on robot rosbags against the floor under the stance feet.

The model (.ts) runs as the robot's node runs it (deploy/terrain_reconstructor.py): the proprioception at every control
step, a prediction at every LiDAR message. The map is read under each stance foot (forward kinematics) and compared
with the foot's height: + = the map below the feet. Stance from the foot force of /lowstate, or the lowest feet.
With --published, also the maps the node published in the bag (/height_scan_markers): their difference to the replay
is ~1 mm when the bag was recorded with that model (rosbags of 18/11), so the replay is what the robot shows.

    python tested/bag_offset.py --model <model>.ts --bags <rosbag dir> [...] [--published]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robots  # noqa: E402
from deploy.terrain_reconstructor import TerrainReconstructor  # noqa: E402
from robots.setup_robot import Bag, _Cdr, cloud_points, rotation_xyzw  # noqa: E402
from sensors import lidar_grid  # noqa: E402
from sensors.odometry import feet  # noqa: E402

EMPTY_ZONE = (-0.15, 0.15)  # the zone under the body of deploy/config.py
FOOT_RADIUS = 0.022
FORCE_STANCE = 30.0  # /lowstate foot force of a foot on the ground
LOWEST_FEET = 0.015  # m, without foot force: the feet this close to the lowest one are on the ground
STANDING_JOINT_SPEED = 0.2  # rad/s, every joint slower: the robot stands still
CLOCK = 4.2e5  # s, a monotonic clock as on the robot's computer (large, as the node's float times)


def _imu(data: bytes) -> list[float]:
    cdr = _Cdr(data)
    cdr.dls2_header()
    quat = [cdr.read("d") for _ in range(4)]
    for _ in range(12):
        cdr.read("d")
    return quat + [cdr.read("d") for _ in range(3)]


def _joints(data: bytes) -> list[float]:
    cdr = _Cdr(data)
    cdr.dls2_header()
    cdr.string()
    for _ in range(cdr.read("I")):
        cdr.string()
    return [cdr.read("d") for _ in range(cdr.read("I"))] + [cdr.read("d") for _ in range(cdr.read("I"))]


def _foot_force(data: bytes) -> list[float]:
    """FL, FR, RL, RR of a unitree_go/LowState."""
    cdr = _Cdr(data)
    for fmt in "BBBBIIIIH" "ffff" "fff" "ffffff" "b":
        cdr.read(fmt)
    for _ in range(20):
        for fmt in "Bfffffffb" "III":
            cdr.read(fmt)
    for fmt in "BBBBiH" "bbbb":
        cdr.read(fmt)
    for _ in range(15):
        cdr.read("H")
    force = dict(zip(("FR", "FL", "RR", "RL"), (cdr.read("h") for _ in range(4))))
    return [force[leg] for leg in ("FL", "FR", "RL", "RR")]


def _marker_positions(data: bytes) -> list[list[float]]:
    """Sphere positions of a MarkerArray of the node (the Marker of ROS 2 Jazzy, as on the robot's computer)."""
    cdr = _Cdr(data)
    positions = []
    for _ in range(cdr.read("I")):
        cdr.read("i"), cdr.read("I")
        if (cdr.string(), cdr.string()) != ("base_link", "height_scan"):
            raise ValueError("Not a heightmap of deploy/run_sensor2heightmap_ros2.py, or not ROS 2 Jazzy's Marker.")
        cdr.read("i"), cdr.read("i"), cdr.read("i")  # id, type, action
        positions.append([cdr.read("d") for _ in range(3)])
        for fmt in "d" * 7 + "f" * 4 + "iI?":  # orientation, scale, color, lifetime, frame_locked
            cdr.read(fmt)
        for size, fmt in ((3, "d"), (4, "f")):  # points, colors
            for _ in range(cdr.read("I") * size):
                cdr.read(fmt)
        cdr.string(), cdr.header(), cdr.string()  # texture_resource, texture header and format
        for size, fmt in ((1, "B"), (2, "f")):  # texture data, uv_coordinates
            for _ in range(cdr.read("I") * size):
                cdr.read(fmt)
        cdr.string(), cdr.string(), cdr.string()  # text, mesh_resource, mesh_file filename
        for _ in range(cdr.read("I")):  # mesh_file data
            cdr.read("B")
        cdr.read("?")  # mesh_use_embedded_materials
    return positions


def read_bag(path: str) -> dict:
    """Times (s from the first message) and messages of the LiDAR, the IMU, the joints, the foot force and the
    heightmaps the node published."""
    bag = Bag(path)
    query = "select timestamp, data from messages join topics on topic_id = topics.id where name = ? order by timestamp"
    topics = {"cloud": ("/utlidar/cloud", cloud_points), "imu": ("/imu", _imu),
              "joints": ("/blind_state_legged", _joints), "force": ("/lowstate", _foot_force),
              "markers": ("/height_scan_markers", _marker_positions)}
    data = {}
    for key, (topic, parse) in topics.items():
        rows = list(bag.database.execute(query, (topic,)))
        if not rows and key not in ("force", "markers"):
            raise ValueError(f"{path}: no {topic} messages.")
        data[f"{key}_t"] = np.array([stamp for stamp, _ in rows], dtype=np.float64) * 1e-9
        data[key] = [parse(message) for _, message in rows] if key == "cloud" else np.array([parse(m) for _, m in rows])
    start = min(data[f"{key}_t"][0] for key in ("cloud", "imu", "joints", "force") if len(data[f"{key}_t"]))
    for key in topics:
        data[f"{key}_t"] -= start
    return data


def _latest(times: np.ndarray, t: float) -> int:
    return max(int(np.searchsorted(times, t, side="right")) - 1, 0)


def replay(model: str | Path, bag: dict, joint_names: list[str], empty_zone=None) -> dict:
    """The model as the node runs it: the maps, and the joints and gravity at each map."""
    reconstructor = TerrainReconstructor(model, joint_names, "cpu", empty_zone)
    dt, period = reconstructor.config["step_dt"], reconstructor.message_period
    start = max(bag["imu_t"][0], bag["joints_t"][0], bag["cloud_t"][0]) + dt
    end = min(bag["imu_t"][-1], bag["joints_t"][-1])
    ticks = [(t, 0, i) for i, t in enumerate(np.arange(start, end, dt))]
    events = sorted(ticks + [(t, 1, i) for i, t in enumerate(bag["cloud_t"]) if start <= t < end])
    out = {"t": [], "heights": [], "joints": [], "gravity": []}
    first = True
    for t, kind, i in events:
        k, j = _latest(bag["imu_t"], t), _latest(bag["joints_t"], t)
        gravity = -rotation_xyzw(bag["imu"][k, :4])[2]
        if kind == 0:
            reconstructor.proprio(bag["imu"][k, 4:7], gravity, bag["joints"][j, :12], bag["joints"][j, 12:24])
            continue
        reconstructor.sensor_message(bag["cloud"][i], CLOCK + t - period)
        heights = reconstructor.predict(CLOCK + t).numpy()
        if first:  # the node publishes the flat map at the first message
            first = False
            continue
        for key, value in (("t", t), ("heights", heights), ("joints", bag["joints"][j]), ("gravity", gravity)):
            out[key].append(value)
    return {key: np.array(value) for key, value in out.items()} | {"heightmap": reconstructor.config["heightmap"]}


def published(bag: dict, heightmap: dict) -> dict:
    """The maps the node published, as replay returns its maps. The node sends the cells (x, y, -height - 0.5) turned
    by the yaw frame of that moment into base_link: the frame comes back from the cells' x and y (least squares)."""
    rows, cols = heightmap["shape"]
    x = heightmap["offset"][0] - heightmap["size"][0] / 2 + heightmap["resolution"] * np.arange(cols)
    y = heightmap["offset"][1] - heightmap["size"][1] / 2 + heightmap["resolution"] * np.arange(rows)
    cells = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    out = {"t": bag["markers_t"], "heights": [], "joints": [], "gravity": []}
    for t, points in zip(bag["markers_t"], bag["markers"]):
        if len(points) != len(cells):
            raise ValueError(f"The node published {len(points)} cells, the model's heightmap has {len(cells)}.")
        forward, left = np.linalg.lstsq(points, cells, rcond=None)[0].T
        up = np.cross(forward, left)
        up /= np.linalg.norm(up)
        out["heights"].append(-(points @ up) - lidar_grid.HEIGHT_OFFSET)
        out["gravity"].append(-up)
        out["joints"].append(bag["joints"][_latest(bag["joints_t"], t)])
    return {key: np.array(value) for key, value in out.items()} | {"heightmap": heightmap}


def difference(maps: dict, reference: dict) -> float:
    """Mean |maps - reference| over the cells (the latest reference map), median over the maps, mm."""
    index = np.searchsorted(reference["t"], maps["t"], side="right") - 1
    valid = index >= 0
    return float(np.median(np.abs(maps["heights"][valid] - reference["heights"][index[valid]]).mean(axis=1)) * 1000)


def _map_at(heights: np.ndarray, heightmap: dict, x: float, y: float) -> float:
    """Bilinear reading of the map (rows along y, columns along x) at (x, y) in the yaw-aligned base frame."""
    rows, cols = heightmap["shape"]
    resolution = heightmap["resolution"]
    x0 = heightmap["offset"][0] - heightmap["size"][0] / 2
    y0 = heightmap["offset"][1] - heightmap["size"][1] / 2
    fx = np.clip((x - x0) / resolution, 0, cols - 1 - 1e-6)
    fy = np.clip((y - y0) / resolution, 0, rows - 1 - 1e-6)
    i, j, a, b = int(fx), int(fy), fx - int(fx), fy - int(fy)
    h = heights.reshape(rows, cols)
    return h[j, i] * (1 - a) * (1 - b) + h[j, i + 1] * a * (1 - b) + h[j + 1, i] * (1 - a) * b + h[j + 1, i + 1] * a * b


def score(maps: dict, bag: dict, kinematics: dict) -> dict:
    """Offset (median, mm) of the map under the stance feet, walking and standing, under the front and the hind feet
    (a map tilted against the floor); error and jitter standing still."""
    force_ok = np.ones(4, dtype=bool)
    if len(bag["force_t"]):  # a foot force sensor that never reads a stance is broken (RR on 6/11)
        force_ok = bag["force"].max(axis=0) > FORCE_STANCE
    frames = lidar_grid.yaw_frame(torch.tensor(maps["gravity"], dtype=torch.float32)).numpy()
    base = feet(torch.tensor(maps["joints"][:, :12], dtype=torch.float32), kinematics).numpy()
    level = np.einsum("nfj,njk->nfk", base, frames)
    standing = np.abs(maps["joints"][:, 12:24]).max(axis=1) < STANDING_JOINT_SPEED
    errors, error_standing, by_pair = [], [], ([], [])  # by_pair: front feet (FL, FR), hind feet (RL, RR)
    for n, t in enumerate(maps["t"]):
        z = level[n, :, 2]
        if len(bag["force_t"]):
            stance = force_ok & (bag["force"][_latest(bag["force_t"], t)] > FORCE_STANCE)
        else:
            stance = z < z.min() + LOWEST_FEET
            stance &= stance.sum() >= 2
        for leg in np.nonzero(stance)[0]:
            x, y, _ = level[n, leg]
            floor = -(z[leg] - FOOT_RADIUS) - lidar_grid.HEIGHT_OFFSET
            error = _map_at(maps["heights"][n], maps["heightmap"], x, y) - floor
            (error_standing if standing[n] else errors).append(error)
            by_pair[leg // 2].append(error)
    both = np.r_[errors, error_standing]
    steps = standing[1:] & standing[:-1]
    change = np.abs(np.diff(maps["heights"], axis=0))[steps]
    return {
        "offset_mm": np.median(both) * 1000,
        "offset_walking_mm": np.median(errors) * 1000 if errors else np.nan,
        "offset_standing_mm": np.median(error_standing) * 1000 if error_standing else np.nan,
        "offset_front_feet_mm": np.median(by_pair[0]) * 1000 if by_pair[0] else np.nan,
        "offset_hind_feet_mm": np.median(by_pair[1]) * 1000 if by_pair[1] else np.nan,
        "feet_mae_mm": np.abs(both).mean() * 1000,
        "standing_change_per_step_mm": change.mean() * 1000 if len(change) else np.nan,
        "standing_jumps_over_5cm_pct": (change.max(axis=1) > 0.05).mean() * 100 if len(change) else np.nan,
        "standing_s": float(standing.mean() * (maps["t"][-1] - maps["t"][0])),
    }


def bag_name(path: str) -> str:
    """rosbag2_2026_11_06-18_18_45 -> 2026_11_06-18_18_45"""
    return Path(path).name.removeprefix("rosbag2_")


def evaluate(model: str | Path, bags: list[str], empty_zone=EMPTY_ZONE,
             with_published: bool = False) -> dict[str, float]:
    """bag/<name>/<metric>, and bag/<name>/empty_zone/<metric> with the zone under the body emptied. with_published:
    also bag/<name>/published/<metric> of the maps the robot published, and their difference to the replay (a few mm
    if the model ran on the robot as replayed)."""
    extra = {"config.json": ""}
    torch.jit.load(str(model), map_location="cpu", _extra_files=extra)
    config = json.loads(extra["config.json"])
    robot = robots.load(config["robot"])
    metrics = {}
    for path in bags:
        bag, name = read_bag(path), bag_name(path)
        replayed = replay(model, bag, robot.MESSAGE_JOINT_NAMES)
        runs = {f"bag/{name}": replayed,
                f"bag/{name}/empty_zone": replay(model, bag, robot.MESSAGE_JOINT_NAMES, empty_zone)}
        if with_published and len(bag["markers_t"]):
            robot_maps = published(bag, config["heightmap"])
            runs[f"bag/{name}/published"] = robot_maps
            metrics[f"bag/{name}/published/difference_to_replay_mm"] = difference(robot_maps, replayed)
        for prefix, maps in runs.items():
            metrics.update({f"{prefix}/{key}": value for key, value in score(maps, bag, robot.LEG_KINEMATICS).items()})
    return metrics


def print_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        print(f"{key}: {value:+.1f}" if key.endswith("_mm") else f"{key}: {value:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offset of the reconstructed map against the feet on robot rosbags.")
    parser.add_argument("--model", required=True, nargs="+", help="Exported reconstructor(s) (.ts).")
    parser.add_argument("--bags", required=True, nargs="+", help="rosbag2 directories with /utlidar/cloud, /imu, "
                        "/blind_state_legged (and /lowstate for the foot force).")
    parser.add_argument("--published", action="store_true",
                        help="Also score the maps the robot published in the bags and their difference to the replay.")
    parser.add_argument("--wandb", action="store_true", help="Log each model's metrics to a W&B run <model>_bags.")
    parser.add_argument("--wandb_project", default="sensors2heightmap", help="W&B project name.")
    args = parser.parse_args()
    torch.set_num_threads(8)
    for model in args.model:
        print(f"== {model}")
        metrics = evaluate(model, args.bags, with_published=args.published)
        print_metrics(metrics)
        if args.wandb:
            import wandb

            run = wandb.init(project=args.wandb_project, name=f"{Path(model).stem}_bags", job_type="bag_offset",
                             config={"model": str(Path(model).resolve()), "bags": args.bags})
            run.log(metrics)
            run.summary.update(metrics)
            run.finish()
