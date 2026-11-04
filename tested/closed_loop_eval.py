# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed loop: the policy walks on the true, predicted or flat heightmap and its falls and tracking are
measured."""


import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--terrain_model_path", type=str, required=True, help="Checkpoint of the terrain reconstructor.")
parser.add_argument("--heightmap", choices=("true", "predicted", "flat"), required=True, help="Map the policy gets.")
parser.add_argument("--task", type=str, default="Locomotion-Go2-Rough-Vision", help="Isaac Lab task to run.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of simulated environments.")
parser.add_argument("--num_steps", type=int, default=3000, help="Control steps to run (50 per second), random commands.")
parser.add_argument("--scenario", action="store_true", help="Scripted commands (SCENARIO), the same for every robot.")
parser.add_argument("--results", type=str, required=True, help="CSV file the results lines are appended to.")
parser.add_argument("--seed", type=int, default=7, help="Seed used for the environment.")
parser.add_argument(
    "--observation_noise",
    choices=("off", "robot", "collection"),
    default="off",
    help="robot: noisy proprioception for policy and reconstructor; collection: as in the recorded data.",
)
parser.add_argument("--trace", type=str, default=None, help="Save per-step, per-robot map errors and base state here.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent configuration entry point."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import basic_locomotion_isaaclab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import sim_helpers
from terrain_reconstruction_mlp import build_terrain_reconstructor

SCENARIO = (
    ("stand", 3.0, 0.0),
    ("slow 0.2", 10.0, 0.2),
    ("stand after slow", 5.0, 0.0),
    ("fast 0.8", 10.0, 0.8),
    ("stand after fast", 5.0, 0.0),
    ("medium 0.5", 8.0, 0.5),
    ("stand after medium", 5.0, 0.0),
    ("backward -0.5", 8.0, -0.5),
    ("stand after backward", 5.0, 0.0),
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    model_path = Path(args_cli.terrain_model_path).expanduser().resolve()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_metadata = checkpoint.get("dataset_metadata", {})

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    sim_helpers.configure_from_checkpoint(args_cli.task, model_metadata, "external" if args_cli.scenario else None)
    if args_cli.observation_noise == "off":
        env_cfg.observation_noise_model = None
    env_cfg.perceptive_height_scanner.debug_vis = False
    if args_cli.scenario:
        env_cfg.episode_length_s = sum(duration for _, duration, _ in SCENARIO) + 10.0
    policy_checkpoint = sim_helpers.policy_checkpoint_path(args_cli, agent_cfg, model_metadata)
    env_cfg.log_dir = sim_helpers.policy_run_dir(policy_checkpoint)
    sim_helpers.configure_heightmap_from_policy(env_cfg, policy_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = sim_helpers.load_policy(env, agent_cfg, policy_checkpoint)

    device = env.unwrapped.device
    model = build_terrain_reconstructor(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    mean, std = checkpoint["target_normalization"]["mean"], checkpoint["target_normalization"]["std"]
    rows, cols = checkpoint["model_config"]["heightmap_size"]
    inputs = sim_helpers.ReconstructorInputs.from_metadata(
        env, model_metadata, proprio_noise=args_cli.observation_noise == "robot"
    )
    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    step_dt = float(unwrapped.step_dt)
    if args_cli.scenario:
        phases = [(name, round(duration / step_dt), speed) for name, duration, speed in SCENARIO]
    else:
        phases = [("random commands", args_cli.num_steps, None)]

    obs = env.get_observations()
    if args_cli.scenario:
        env.episode_length_buf = torch.zeros_like(unwrapped.episode_length_buf)
    heightmap_slice = sim_helpers.heightmap_slice(obs, env)
    print(f"[INFO] Heightmap given to the policy: {args_cli.heightmap} | reconstructor: {model_path.stem} "
          f"({model_metadata['sensor_model']}) | observation noise: {args_cli.observation_noise} | "
          f"{sum(n for _, n, _ in phases)} steps in {len(phases)} phase(s)")

    keys = ("falls", "timeouts", "lin_vel_error", "yaw_rate_error", "thigh_collisions", "foot_stumbles", "reward",
            "map_error")
    totals = [dict.fromkeys(keys, 0.0) for _ in phases]
    flat_map = torch.full((num_envs, rows * cols), mean, device=device)
    fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
    hidden_state = None
    trace = None
    if args_cli.trace:
        trace = {key: [] for key in ("ground_error", "abs_ground_error", "base_height", "forward_velocity", "ready", "dones")}
        pose_scanner = unwrapped._pose_height_scanner
    with torch.inference_mode():
        for phase, (name, phase_steps, speed) in enumerate(phases):
            total = totals[phase]
            for step in range(phase_steps):
                if speed is not None:
                    unwrapped._commands[:, 0] = speed
                    unwrapped._commands[:, 1:3] = 0.0
                true_map = sim_helpers.true_heightmap(env)
                frame, proprio = inputs(fresh)
                prediction = model(depth_data=frame[:, None], robot_info=proprio[:, None], hidden_state=hidden_state)
                hidden_state = torch.where(fresh.view(1, -1, 1), torch.zeros_like(prediction.hidden_state), prediction.hidden_state)
                ready = ~fresh
                predicted_map = (prediction.refined_heightmap.reshape(num_envs, -1) * std + mean).clip(-1.0, 1.0)
                predicted_map = torch.where(ready[:, None], predicted_map, flat_map)
                cell_error = (predicted_map - true_map).abs().view(num_envs, rows, cols)
                total["map_error"] += cell_error.mean(dim=(1, 2)).sum().item()
                if trace is not None:
                    ground_error = (true_map - predicted_map).view(num_envs, rows, cols)
                    trace["ground_error"].append(ground_error.mean(dim=1).half().cpu())
                    trace["abs_ground_error"].append(cell_error.mean(dim=1).half().cpu())
                    base_height = (sim_helpers.as_torch(pose_scanner.data.pos_w)[:, 2]
                                   - sim_helpers.as_torch(pose_scanner.data.ray_hits_w)[..., 2].mean(dim=1))
                    trace["base_height"].append(base_height.cpu())
                    trace["forward_velocity"].append(sim_helpers.as_torch(unwrapped._robot.data.root_lin_vel_b)[:, 0].cpu())
                    trace["ready"].append(ready.cpu())
                if args_cli.heightmap == "predicted":
                    given_map = predicted_map
                elif args_cli.heightmap == "flat":
                    given_map = flat_map
                else:
                    given_map = true_map
                policy_obs = obs["policy"]
                if args_cli.observation_noise == "collection":
                    given_map = given_map + policy_obs[:, heightmap_slice] - true_map
                obs["policy"] = torch.cat((policy_obs[:, :heightmap_slice.start], given_map, policy_obs[:, heightmap_slice.stop:]), dim=1)

                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)
                dones = dones.bool()
                if hasattr(policy, "reset"):
                    policy.reset(dones)
                fresh = dones
                if trace is not None:
                    trace["dones"].append(dones.cpu())

                time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
                total["falls"] += (dones & ~time_outs).sum().item()
                total["timeouts"] += (dones & time_outs).sum().item()
                commands = unwrapped._commands
                lin_vel = sim_helpers.as_torch(unwrapped._robot.data.root_lin_vel_b)
                ang_vel = sim_helpers.as_torch(unwrapped._robot.data.root_ang_vel_b)
                total["lin_vel_error"] += (commands[:, :2] - lin_vel[:, :2]).norm(dim=1).sum().item()
                total["yaw_rate_error"] += (commands[:, 2] - ang_vel[:, 2]).abs().sum().item()
                forces = sim_helpers.as_torch(unwrapped._contact_sensor.data.net_forces_w)
                thighs = forces[:, unwrapped._thigh_contact_sensor_ids]
                total["thigh_collisions"] += (thighs.norm(dim=-1) > 1.0).any(dim=1).sum().item()
                feet = forces[:, unwrapped._feet_contact_sensor_ids]
                total["foot_stumbles"] += (feet[..., :2].norm(dim=-1) > 5.0 * feet[..., 2].abs()).any(dim=1).sum().item()
                total["reward"] += rewards.sum().item()
                if (step + 1) % 250 == 0 or step + 1 == phase_steps:
                    done_steps = (step + 1) * num_envs
                    print(f"[INFO] {name}: step {step + 1}/{phase_steps}, falls {int(total['falls'])}, "
                          f"map MAE {1000 * total['map_error'] / done_steps:.1f} mm")

    results = Path(args_cli.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    for (name, phase_steps, _), total in zip(phases, totals):
        env_steps = phase_steps * num_envs
        row = {
            "model": model_path.stem,
            "heightmap": args_cli.heightmap,
            "observation_noise": args_cli.observation_noise,
            "seed": agent_cfg.seed,
            "phase": name,
            "env_steps": env_steps,
            "falls": int(total["falls"]),
            "timeouts": int(total["timeouts"]),
            "falls_per_1000_steps": 1000 * total["falls"] / env_steps,
            "lin_vel_error_m_s": total["lin_vel_error"] / env_steps,
            "yaw_rate_error_rad_s": total["yaw_rate_error"] / env_steps,
            "thigh_collision_steps_pct": 100 * total["thigh_collisions"] / env_steps,
            "foot_stumble_steps_pct": 100 * total["foot_stumbles"] / env_steps,
            "reward_per_step": total["reward"] / env_steps,
            "map_mae_mm": 1000 * total["map_error"] / env_steps,
        }
        print("[INFO] " + ", ".join(f"{key} {value:.4g}" if isinstance(value, float) else f"{key} {value}" for key, value in row.items()))
        new_file = not results.exists()
        if not new_file and results.read_text().split("\n", 1)[0] != ",".join(row):
            raise ValueError(f"{results} has other columns: write the results to a new file")
        with open(results, "a") as file:
            if new_file:
                file.write(",".join(row) + "\n")
            file.write(",".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row.values()) + "\n")
    if trace is not None:
        trace = {key: torch.stack(values) for key, values in trace.items()}
        trace.update(phases=[(name, phase_steps) for name, phase_steps, _ in phases], model=model_path.stem,
                     heightmap=args_cli.heightmap, observation_noise=args_cli.observation_noise, seed=agent_cfg.seed)
        torch.save(trace, args_cli.trace)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
