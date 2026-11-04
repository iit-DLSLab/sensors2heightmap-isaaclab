"""Isaac Lab helpers shared by train/ and tested/."""

from __future__ import annotations

import importlib
import math
import os
import sys
import zipfile
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robots  # noqa: E402
import sensors  # noqa: E402
from terrain_reconstruction_mlp import ProprioNoise  # noqa: E402

HEIGHTMAP_HEIGHT_OFFSET = 0.5
ERROR_THRESHOLDS = (0.01, 0.03, 0.05)
SENSOR_NAME = "terrain_sensor"
PROPRIO_HISTORY = 5


def as_torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def heightmap_spec(env_cfg) -> dict:
    """Size, resolution and offset of the policy heightmap."""
    scanner = env_cfg.perceptive_height_scanner
    return {"size": tuple(scanner.pattern_cfg.size), "resolution": scanner.pattern_cfg.resolution,
            "offset": tuple(scanner.offset.pos[:2])}


class _SensorMount(dict):
    """Creates the sensor when the env adds its robot to the scene.

    Newton's ray casters look for the robot's meshes when they are created, so the sensor can't be in the scene
    cfg, which is built before the robot."""

    def __init__(self, scene, sensor_cfg):
        super().__init__(scene.articulations)
        self.scene, self.sensor_cfg = scene, sensor_cfg

    def __setitem__(self, name, articulation):
        super().__setitem__(name, articulation)
        if name == "robot":
            self.scene.sensors[SENSOR_NAME] = self.sensor_cfg.class_type(self.sensor_cfg)


def attach_sensor(task: str, sensor_name: str, mount, robot_name: str, debug_vis: bool = False) -> None:
    """Mounts a sensor model on the task's robot (direct envs with a 'robot' articulation)."""
    robot = robots.load(robot_name)
    sensor_cfg = sensors.load(sensor_name).sim_cfg(
        robot.BASE_PRIM_PATH, *mount, robot.SENSORS[sensor_name]["meshes"]
    )
    sensor_cfg.debug_vis = debug_vis
    env_class = gym.spec(task).entry_point
    if isinstance(env_class, str):
        module, name = env_class.split(":")
        env_class = getattr(importlib.import_module(module), name)
    setup_scene = getattr(env_class, "_setup_scene_without_sensor", env_class._setup_scene)

    def _setup_scene(env) -> None:
        env.scene._articulations = _SensorMount(env.scene, sensor_cfg)
        setup_scene(env)
        env.scene._articulations = dict(env.scene._articulations)
        if SENSOR_NAME not in env.scene.sensors:
            raise RuntimeError(f"{task} has no 'robot' articulation in its scene: cannot mount {sensor_name} on it.")

    env_class._setup_scene_without_sensor = setup_scene
    env_class._setup_scene = _setup_scene


class ProprioHistory:
    """Last steps of proprioception (ang vel, gravity, joint pos and vel), zeros before the episode start."""

    def __init__(self, env: RslRlVecEnvWrapper, length: int = PROPRIO_HISTORY, noise: bool = False):
        self.robot = env.unwrapped.scene.articulations["robot"]
        self.default_joint_pos = as_torch(self.robot.data.default_joint_pos).clone()
        frame = self.frame()
        self.history = torch.zeros(frame.shape[0], length, frame.shape[1], device=frame.device)
        self.noise_std = self.bias_std = None
        if noise:
            stds = ProprioNoise(env.unwrapped.cfg.observation_noise_std, self.robot.num_joints, length)
            self.noise_std, self.bias_std = stds.noise_std.to(frame.device), stds.bias_std.to(frame.device)
            self.bias = torch.randn_like(frame) * self.bias_std
        self.metadata = {
            "proprio_history": length,
            "proprio_joint_names": list(self.robot.joint_names),
            "proprio_default_joint_pos": self.default_joint_pos[0].tolist(),
        }

    def frame(self) -> torch.Tensor:
        data = self.robot.data
        return torch.cat((as_torch(data.root_ang_vel_b), as_torch(data.projected_gravity_b),
                          as_torch(data.joint_pos) - self.default_joint_pos, as_torch(data.joint_vel)), dim=1)

    def __call__(self, dones: torch.Tensor) -> torch.Tensor:
        dones = dones.bool().view(-1, 1)
        frame = self.frame()
        if self.noise_std is not None:
            self.bias = torch.where(dones, torch.randn_like(frame) * self.bias_std, self.bias)
            frame = frame + self.bias + torch.randn_like(frame) * self.noise_std
        history = torch.where(dones.unsqueeze(-1), 0.0, self.history)
        self.history = torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)
        return self.history.flatten(1)


class ReconstructorInputs:
    """Sensor input and proprioception of the reconstructor, at every step."""

    def __init__(self, env: RslRlVecEnvWrapper, sensor_name: str, mount, robot_name: str, errors: dict,
                 history: int = PROPRIO_HISTORY, proprio_noise: bool = False):
        sensor = sensors.load(sensor_name)
        unwrapped = env.unwrapped
        self.sensor_input = sensor(
            env, unwrapped.scene.sensors[SENSOR_NAME], mount, heightmap_spec(unwrapped.cfg), **errors
        )
        missing = set(sensors.REQUIRED_METADATA[sensor.KIND]) - set(self.sensor_input.metadata)
        if missing:
            raise ValueError(f"{sensor.__name__}.metadata lacks {sorted(missing)}: the robot needs them.")
        self.proprio = ProprioHistory(env, history, proprio_noise)
        self.metadata = {
            "sensor": sensor.KIND,
            "sensor_model": sensor_name,
            "robot": robot_name,
            "sensor_mount": tuple(tuple(value) for value in mount),
            "sensor_errors": dict(errors),
            **self.sensor_input.metadata,
            **self.proprio.metadata,
        }

    @classmethod
    def from_metadata(cls, env: RslRlVecEnvWrapper, metadata: dict, proprio_noise: bool = False) -> ReconstructorInputs:
        """Inputs as in the reconstructor's training data."""
        return cls(env, metadata["sensor_model"], metadata["sensor_mount"], metadata["robot"],
                   metadata["sensor_errors"], metadata["proprio_history"], proprio_noise)

    def __call__(self, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sensor_input(dones), self.proprio(dones)


def _forward_commands(env, env_ids: torch.Tensor | None = None) -> None:
    """Forward-only commands (0.3-0.8 m/s, turns of at most 90 deg), no stops."""

    def sample(num: int, duration_s: float) -> torch.Tensor:
        commands = torch.zeros(num, 3, device=env.device)
        commands[:, 0].uniform_(0.3, 0.8)
        heading_change = torch.empty(num, device=env.device).uniform_(-0.5 * math.pi, 0.5 * math.pi)
        commands[:, 2] = (heading_change / duration_s).clamp(-0.5, 0.5)
        return commands

    if env_ids is not None:
        env._commands[env_ids, :3] = sample(len(env_ids), (env.max_episode_length - 400) * env.step_dt)
    for steps_left, duration in ((400, 250), (150, 150)):
        resample = env.episode_length_buf == env.max_episode_length - steps_left
        if resample.any():
            env._commands[resample, :3] = sample(int(resample.sum()), duration * env.step_dt)


def set_commands(mode: str) -> None:
    """Velocity commands: 'default', 'forward', or 'external' (set by the caller)."""
    if mode == "default":
        return
    from basic_locomotion_isaaclab.tasks import custom_events

    commands = {"forward": _forward_commands, "external": lambda env, env_ids=None: None}[mode]
    custom_events._get_new_random_commands = commands


def configure_from_checkpoint(task: str, metadata: dict, commands: str | None = None, debug_vis: bool = False) -> None:
    """Sensor and commands as in the reconstructor's training data."""
    attach_sensor(task, metadata["sensor_model"], metadata["sensor_mount"], metadata["robot"], debug_vis)
    set_commands(commands or ("forward" if metadata.get("commands") == "forward" else "default"))


def true_heightmap(env: RslRlVecEnvWrapper) -> torch.Tensor:
    """Policy heightmap without noise, (num_envs, cells)."""
    scanner = env.unwrapped._perceptive_height_scanner
    heights = as_torch(scanner.data.pos_w)[:, 2:3] - as_torch(scanner.data.ray_hits_w)[..., 2] - HEIGHTMAP_HEIGHT_OFFSET
    return torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=-1.0).clip(-1.0, 1.0)


def heightmap_slice(obs, env: RslRlVecEnvWrapper) -> slice:
    """Position of the heightmap in the policy observation."""
    common_dim, cells = obs["common"].shape[1], true_heightmap(env).shape[1]
    where = slice(common_dim, common_dim + cells)
    offset = obs["policy"][:, where] - true_heightmap(env)
    if not torch.allclose(obs["policy"][:, :common_dim], obs["common"]) or offset.std(dim=1).max() > 1e-4:
        raise RuntimeError("The policy observation is not ['common', heightmap]: cannot replace the heightmap.")
    return where


def policy_checkpoint_path(args_cli, agent_cfg, model_metadata: dict) -> str:
    """--checkpoint, else the policy of the reconstructor's data, else the latest run."""
    if getattr(args_cli, "checkpoint", None):
        return retrieve_file_path(args_cli.checkpoint)
    if Path(str(model_metadata.get("checkpoint_path", ""))).is_file():
        return str(model_metadata["checkpoint_path"])
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def policy_run_dir(checkpoint: str) -> str:
    folder = Path(checkpoint).resolve().parent
    return str(folder.parent if folder.name == "exported" else folder)


def configure_heightmap_from_policy(env_cfg, checkpoint: str) -> None:
    """Uses the heightmap the policy was trained with (<run>/params/env.yaml)."""
    env_yaml = Path(policy_run_dir(checkpoint)) / "params" / "env.yaml"
    if not env_yaml.is_file():
        print(f"[WARN] {env_yaml} not found: the heightmap is the task's default.")
        return
    with open(env_yaml) as file:
        training_env = yaml.unsafe_load(file)
    training_scanner = training_env["perceptive_height_scanner"]
    resolution_heightmap = training_scanner["pattern_cfg"]["resolution"]
    num_rows_heightmap = round(training_scanner["pattern_cfg"]["size"][0] / resolution_heightmap) + 1
    num_cols_heightmap = round(training_scanner["pattern_cfg"]["size"][1] / resolution_heightmap) + 1

    pattern_cfg = env_cfg.perceptive_height_scanner.pattern_cfg
    task_cells = (round(pattern_cfg.size[0] / pattern_cfg.resolution) + 1) * (
        round(pattern_cfg.size[1] / pattern_cfg.resolution) + 1
    )
    pattern_cfg.resolution = resolution_heightmap
    pattern_cfg.size = tuple(training_scanner["pattern_cfg"]["size"])
    env_cfg.perceptive_height_scanner.offset.pos = tuple(training_scanner["offset"]["pos"])
    env_cfg.observation_space += num_rows_heightmap * num_cols_heightmap - task_cells
    print(
        f"[INFO] Heightmap from {env_yaml}: {num_rows_heightmap}x{num_cols_heightmap}, resolution "
        f"{resolution_heightmap} m, offset {env_cfg.perceptive_height_scanner.offset.pos}"
    )


def is_torchscript(path: str) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any(name.split("/")[1:2] == ["code"] for name in archive.namelist())


class TorchScriptPolicy:
    """Policy exported with torch.jit, run without its runner (stateless policies only)."""

    def __init__(self, path: str, obs_groups: list[str], device):
        self.model = torch.jit.load(path, map_location=device).eval()
        if hasattr(self.model, "rnn"):
            raise ValueError(
                f"{path} is a recurrent policy, whose export keeps a single hidden state: use its training checkpoint "
                "(model_*.pt) instead."
            )
        self.obs_groups = list(obs_groups)

    def __call__(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.model(torch.cat([obs[group] for group in self.obs_groups], dim=-1))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Stateless: nothing to reset."""


def load_policy(env: RslRlVecEnvWrapper, agent_cfg, checkpoint: str):
    """TorchScript export or rsl_rl checkpoint."""
    if checkpoint.endswith(".onnx"):
        raise ValueError(
            f"{checkpoint}: the ONNX export has batch size 1, use the TorchScript one next to it (policy.pt)."
        )
    if is_torchscript(checkpoint):
        actor = "student" if agent_cfg.class_name == "DistillationRunner" else "actor"
        return TorchScriptPolicy(checkpoint, agent_cfg.obs_groups[actor], env.unwrapped.device)
    runner_class = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}[agent_cfg.class_name]
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=env.unwrapped.device)


def marker_legend(mode: str = "tiles", show_ground_truth: bool = False, thresholds=ERROR_THRESHOLDS) -> str:
    t1, t2, t3 = (round(threshold * 100) for threshold in thresholds)
    legend = {
        "tiles": f"tiles = predicted heightmap, coloured by error: green <{t1} cm | ground too high: yellow {t1}-{t2}, "
        f"orange {t2}-{t3}, red >{t3} | too low: cyan, blue, purple",
        "spheres": "blue spheres = predicted ground height (the error is their distance from the terrain)",
        "hidden": "heightmap not drawn",
    }[mode]
    return legend + (" | white dots = true ground" if show_ground_truth and mode != "hidden" else "")


class HeightmapSurface:
    """Draws the predicted heightmap of one env as tiles coloured by the error, or as spheres."""

    MODES = ("tiles", "spheres", "hidden")
    ERROR_COLORS = ((0.1, 0.9, 0.1), (1.0, 0.9, 0.1), (1.0, 0.5, 0.05), (0.9, 0.05, 0.05),
                    (0.2, 0.9, 1.0), (0.1, 0.35, 1.0), (0.5, 0.1, 0.8))
    SPHERE_COLOR = (0.15, 0.45, 1.0)

    def __init__(self, cell_size: float = 0.1, show_ground_truth: bool = False, mode: str = "tiles",
                 sphere_radius: float = 0.02, thresholds=ERROR_THRESHOLDS):
        self.thresholds, self.thickness, self.lift, self.radius, self.mode = thresholds, 0.01, 0.01, sphere_radius, mode

        def markers(path: str, shapes: dict) -> VisualizationMarkers:
            return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers=shapes))

        def material(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
            return sim_utils.PreviewSurfaceCfg(diffuse_color=color)

        tile = (0.9 * cell_size, 0.9 * cell_size, self.thickness)
        self.tiles = markers("/Visuals/DepthHeightmapSurface", {
            f"error_{i}": sim_utils.CuboidCfg(size=tile, visual_material=material(color))
            for i, color in enumerate(self.ERROR_COLORS)
        })
        self.spheres = markers("/Visuals/DepthHeightmapPrediction", {
            "prediction": sim_utils.SphereCfg(radius=sphere_radius, visual_material=material(self.SPHERE_COLOR))
        })
        self.truth = None
        if show_ground_truth:
            self.truth = markers("/Visuals/DepthHeightmapTarget", {
                "target": sim_utils.SphereCfg(radius=0.009, visual_material=material((1.0, 1.0, 1.0)))
            })
        self.set_visibility(True)

    def set_visibility(self, visible: bool) -> None:
        self.visible, self.shown_mode = visible, self.mode
        self.tiles.set_visibility(visible and self.mode == "tiles")
        self.spheres.set_visibility(visible and self.mode == "spheres")
        if self.truth is not None:
            self.truth.set_visibility(visible and self.mode != "hidden")

    def next_mode(self) -> None:
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]

    def update(self, env: RslRlVecEnvWrapper, env_id: int, predicted: torch.Tensor) -> torch.Tensor:
        """Draws the prediction, returns predicted minus true ground height per cell."""
        if not self.visible or self.shown_mode != self.mode:
            self.set_visibility(True)
        scanner = env.unwrapped._perceptive_height_scanner
        sensor_z = as_torch(scanner.data.pos_w)[env_id, 2]
        ray_hits = as_torch(scanner.data.ray_hits_w)[env_id]
        predicted_z = sensor_z - predicted.reshape(-1) - HEIGHTMAP_HEIGHT_OFFSET
        error = predicted_z - ray_hits[:, 2]
        valid = torch.isfinite(predicted_z) & torch.isfinite(ray_hits).all(dim=1)

        if self.mode == "tiles":
            level = sum((error.abs() >= threshold).long() for threshold in self.thresholds)
            indices = torch.where(error >= 0, level, torch.where(level > 0, level + 3, 0))
            centers = torch.cat((ray_hits[:, :2], (predicted_z + self.lift + 0.5 * self.thickness).unsqueeze(1)), dim=1)
            x, y, z, w = as_torch(scanner.data.quat_w)[env_id].unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            heading = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)))
            self.tiles.visualize(
                translations=centers[valid], orientations=heading.expand(int(valid.sum()), 4), marker_indices=indices[valid]
            )
        elif self.mode == "spheres":
            centers = torch.cat((ray_hits[:, :2], (predicted_z + self.radius).unsqueeze(1)), dim=1)
            self.spheres.visualize(translations=centers[valid])
        if self.truth is not None and self.mode != "hidden":
            self.truth.visualize(translations=ray_hits[valid] + torch.tensor([0.0, 0.0, 0.009], device=ray_hits.device))
        return error.view_as(predicted)
