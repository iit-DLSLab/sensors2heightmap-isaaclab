# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Records the sensor input, the proprioception and the target heightmap at every step, to train the reconstructor."""


import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect terrain reconstruction sequences with a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="Directory for the chunk files. Defaults to <checkpoint_dir>/terrain_reconstruction_sequences.",
)
parser.add_argument("--num_steps", type=int, default=1200, help="Steps to record; every env is recorded at every step.")
parser.add_argument(
    "--sensor",
    required=True,
    help="Sensor model, a file of sensors/ (unitree_l2_lidar, realsense_d435).",
)
parser.add_argument("--robot", default="go2", help="Robot, a module of robots/: where it carries the sensor.")
parser.add_argument(
    "--sensor_errors",
    default="sensor",
    help="A key of the sensor's ERRORS, e.g. bag for the L2 errors measured on the Go2 rosbag.",
)
parser.add_argument(
    "--forward_commands",
    action="store_true",
    help="Velocity commands always forward (0.3-0.8 m/s), turning at most 90 deg per command, no pauses.",
)
parser.add_argument(
    "--steps_per_chunk",
    type=int,
    default=100,
    help="Steps per chunk file (all envs). With 496 envs a step of fp16 106x60 depth images is about 6 MB.",
)
parser.add_argument(
    "--depth_dtype",
    type=str,
    choices=("float16", "float32"),
    default="float16",
    help="Storage dtype of the sensor input. float16 halves the dataset memory (~1 mm precision in the 0-2 m range).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as importlib_metadata

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import basic_locomotion_isaaclab.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import sim_helpers


class SequenceChunkWriter:
    """Writes the recorded steps to disk in chunks, (steps, envs, ...)."""

    def __init__(self, directory: str, steps_per_chunk: int, depth_dtype: torch.dtype, metadata: dict):
        self.directory = directory
        self.steps_per_chunk = steps_per_chunk
        self.depth_dtype = depth_dtype
        self.metadata = metadata
        self.steps: dict[str, list[torch.Tensor]] = {}
        self.num_steps = 0
        self.chunk_index = 0
        self.chunk_first_step = 0
        os.makedirs(directory, exist_ok=True)

    def add_step(self, **tensors: torch.Tensor) -> None:
        for name, tensor in tensors.items():
            tensor = tensor.detach().to(self.depth_dtype) if name in FRAME_KEYS.values() else tensor.detach()
            self.steps.setdefault(name, []).append(tensor.to("cpu", copy=True))
        self.num_steps += 1
        if self.num_steps - self.chunk_first_step == self.steps_per_chunk:
            self.flush()

    def flush(self) -> None:
        if not self.steps:
            return
        chunk = {name: torch.stack(steps) for name, steps in self.steps.items()}
        chunk["metadata"] = {
            **self.metadata,
            "chunk_index": self.chunk_index,
            "first_step": self.chunk_first_step,
            "num_steps": self.num_steps - self.chunk_first_step,
        }
        path = os.path.join(self.directory, f"chunk_{self.chunk_index:03d}.pt")
        torch.save(chunk, path)
        print(f"[INFO] Saved steps {self.chunk_first_step}-{self.num_steps - 1} to: {path}")
        self.steps = {}
        self.chunk_index += 1
        self.chunk_first_step = self.num_steps


FRAME_KEYS = {"depth": "depth_data", "lidar": "lidar_grid"}


def _get_heightmap_targets(env: RslRlVecEnvWrapper) -> torch.Tensor:
    pattern_cfg = env.unwrapped.cfg.perceptive_height_scanner.pattern_cfg
    cols, rows = (round(size / pattern_cfg.resolution) + 1 for size in pattern_cfg.size)
    return sim_helpers.true_heightmap(env).view(-1, 1, rows, cols)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib_metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = sim_helpers.policy_run_dir(resume_path)
    if args_cli.dataset_path:
        dataset_dir = os.path.abspath(args_cli.dataset_path)
    else:
        dataset_dir = os.path.join(log_dir, "terrain_reconstruction_sequences")
    env_cfg.log_dir = log_dir
    sim_helpers.configure_heightmap_from_policy(env_cfg, resume_path)
    sensor_mount = sim_helpers.robots.load(args_cli.robot).SENSORS[args_cli.sensor]
    mount = (sensor_mount["position"], sensor_mount["rotation"])
    sim_helpers.attach_sensor(args_cli.task, args_cli.sensor, mount, args_cli.robot)
    sim_helpers.set_commands("forward" if args_cli.forward_commands else "default")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "collect_sensors2heightmap"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during dataset collection.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    policy = sim_helpers.load_policy(env, agent_cfg, resume_path)
    obs = env.get_observations()

    errors = sim_helpers.sensors.load(args_cli.sensor).ERRORS[args_cli.sensor_errors]
    inputs = sim_helpers.ReconstructorInputs(env, args_cli.sensor, mount, args_cli.robot, errors)
    grid_cfg = env.unwrapped.cfg.perceptive_height_scanner
    metadata = {
        "format": "sequence",
        **inputs.metadata,
        "heightmap": sim_helpers.heightmap_spec(env.unwrapped.cfg),
        "heightmap_ordering": grid_cfg.pattern_cfg.ordering,
        "heightmap_height_offset": sim_helpers.HEIGHTMAP_HEIGHT_OFFSET,
        "step_dt": env.unwrapped.step_dt,
        "task": args_cli.task,
        "checkpoint_path": resume_path,
        "seed": agent_cfg.seed,
        "heightmap_size": tuple(_get_heightmap_targets(env).shape[-2:]),
        "num_envs": env.unwrapped.num_envs,
        "sequence_steps": args_cli.num_steps,
        "steps_per_chunk": args_cli.steps_per_chunk,
        "depth_dtype": args_cli.depth_dtype,
        "commands": "forward" if args_cli.forward_commands else "default",
    }
    writer = SequenceChunkWriter(
        directory=dataset_dir,
        steps_per_chunk=args_cli.steps_per_chunk,
        depth_dtype=getattr(torch, args_cli.depth_dtype),
        metadata=metadata,
    )

    while simulation_app.is_running() and writer.num_steps < args_cli.num_steps:
        with torch.no_grad():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            sensor_input, proprio = inputs(dones)
            writer.add_step(
                **{FRAME_KEYS[metadata["sensor"]]: sensor_input},
                proprio=proprio,
                heightmaps=_get_heightmap_targets(env),
                dones=dones.bool(),
            )
        if writer.num_steps % 100 == 0:
            print(f"[INFO] Recorded {writer.num_steps} / {args_cli.num_steps} steps.")

    writer.flush()
    print(f"[INFO] Saved {writer.num_steps} steps x {metadata['num_envs']} envs in {writer.chunk_index} chunks to: {dataset_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
