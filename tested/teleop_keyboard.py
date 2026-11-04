"""Drive one robot with the keyboard, the policy walking on the reconstructed heightmap."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--terrain_model_path", type=str, default=None, help="Terrain reconstructor checkpoint; omit for the true map."
)
parser.add_argument("--task", type=str, default="Locomotion-Go2-Rough-Vision", help="Isaac Lab task to run.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (terrain).")
parser.add_argument(
    "--draw",
    choices=("tiles", "spheres", "hidden"),
    default="tiles",
    help="How to draw the prediction (C switches while driving).",
)
parser.add_argument(
    "--blind",
    choices=("zeros", "flat"),
    default=None,
    help="Give the policy an all-zero or a flat map instead (the reconstructor still runs and is drawn).",
)
parser.add_argument(
    "--show_ground_truth", action="store_true", help="Also draw small white dots at the true ground height."
)
parser.add_argument(
    "--show_depth_rays",
    action="store_true",
    help="The ray-hit debug visualization of the sensor (depth camera or LiDAR): shows which ground it sees.",
)
parser.add_argument(
    "--no_observation_noise",
    action="store_true",
    help="Clean proprioception for policy and reconstructor (noisy by default, as on the robot).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent configuration entry point."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata
import time

import gymnasium as gym
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_visualizers.kit import KitVisualizerCfg

import isaaclab_tasks  # noqa: F401
import basic_locomotion_isaaclab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import sim_helpers
from terrain_reconstruction_mlp import build_terrain_reconstructor

COMMAND_LIMITS = (0.8, 0.25, 0.5)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    checkpoint, model_metadata = None, {}
    if args_cli.terrain_model_path:
        checkpoint = torch.load(args_cli.terrain_model_path, map_location="cpu", weights_only=False)
        model_metadata = checkpoint.get("dataset_metadata", {})

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = 3600.0
    if args_cli.no_observation_noise:
        env_cfg.observation_noise_model = None
    env_cfg.perceptive_height_scanner.debug_vis = False
    if checkpoint is not None:
        sim_helpers.configure_from_checkpoint(args_cli.task, model_metadata, "external", args_cli.show_depth_rays)
    else:
        sim_helpers.set_commands("external")
    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(eye=(-1.6, -1.2, 1.0), lookat=(0.6, 0.0, 0.0), origin_type="asset", origin_track_path="robot")
    ]
    policy_checkpoint = sim_helpers.policy_checkpoint_path(args_cli, agent_cfg, model_metadata)
    env_cfg.log_dir = sim_helpers.policy_run_dir(policy_checkpoint)
    sim_helpers.configure_heightmap_from_policy(env_cfg, policy_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = sim_helpers.load_policy(env, agent_cfg, policy_checkpoint)
    unwrapped, device = env.unwrapped, env.unwrapped.device

    model = None
    if checkpoint is not None:
        model = build_terrain_reconstructor(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        mean, std = checkpoint["target_normalization"]["mean"], checkpoint["target_normalization"]["std"]
        surface = sim_helpers.HeightmapSurface(show_ground_truth=args_cli.show_ground_truth, mode=args_cli.draw)
        inputs = sim_helpers.ReconstructorInputs.from_metadata(
            env, model_metadata, proprio_noise=not args_cli.no_observation_noise
        )

    keyboard = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=COMMAND_LIMITS[0], v_y_sensitivity=COMMAND_LIMITS[1],
                                          omega_z_sensitivity=COMMAND_LIMITS[2], sim_device=device))
    reset_requested = []
    keyboard.add_callback("R", lambda: reset_requested.append(True))
    print(keyboard)
    print("\tReset the robot: R")
    if model is not None:

        def switch_drawing():
            surface.next_mode()
            print(f"[INFO] {sim_helpers.marker_legend(surface.mode, args_cli.show_ground_truth)}")

        keyboard.add_callback("C", switch_drawing)
        print("\tSwitch the heightmap drawing (tiles coloured by error / blue spheres / hidden): C")
        print(f"[INFO] {sim_helpers.marker_legend(surface.mode, args_cli.show_ground_truth)}")
    print(f"[INFO] Policy: {policy_checkpoint}")
    print(f"[INFO] Heightmap: {'predicted by ' + args_cli.terrain_model_path if model else 'true'}"
          + (f", the policy gets a {args_cli.blind} map instead" if args_cli.blind else ""))
    print(f"[INFO] Observation noise: {'off' if args_cli.no_observation_noise else 'on (policy and reconstructor)'}"
          + (f" | sensor errors: {model_metadata.get('sensor_errors')}" if model else ""))

    limits = torch.tensor(COMMAND_LIMITS, device=device)
    obs = env.get_observations()
    map_slice = sim_helpers.heightmap_slice(obs, env)
    start_map, blind_map = sim_helpers.true_heightmap(env)[0], None
    if args_cli.blind == "zeros":
        blind_map = torch.zeros_like(start_map)
    elif args_cli.blind == "flat":
        blind_map = torch.full_like(start_map, start_map.mean().item())
    hidden_state, fresh = None, True
    step_dt = float(unwrapped.step_dt)
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            unwrapped._commands[:, :3] = torch.maximum(torch.minimum(keyboard.advance().to(device), limits), -limits)
            if model is not None:
                frame, proprio = inputs(torch.tensor([fresh], device=device))
                prediction = model(depth_data=frame[:, None], robot_info=proprio[:, None],
                                   hidden_state=hidden_state)
                hidden_state = None if fresh else prediction.hidden_state
                flat = torch.full_like(obs["policy"][0, map_slice], mean)
                given = flat if fresh else prediction.refined_heightmap.reshape(-1) * std + mean
                surface.update(env, 0, given)
                given = given.clip(-1.0, 1.0)
            else:
                given = sim_helpers.true_heightmap(env)[0]
            if blind_map is not None:
                given = blind_map
            policy_obs = obs["policy"].clone()
            policy_obs[0, map_slice] = given
            obs["policy"] = policy_obs
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            fresh = bool(dones[0])
            if reset_requested:
                reset_requested.clear()
                obs, _ = env.reset()
                fresh = True
            if hasattr(policy, "reset"):
                policy.reset(dones.bool() | fresh)
        sleep_time = step_dt - (time.time() - start_time)
        if sleep_time > 0.0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
