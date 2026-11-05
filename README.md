## Overview

An Isaac Lab set of tools for heightmap reconstruction from a LiDAR or a depth camera: a network predicts the
heightmap of a vision locomotion policy from the sensor and the robot's proprioception, so the policy runs on the robot
without a height scanner. No need to distill your vision-based policy!

The model (`train/terrain_reconstruction_mlp.py`): at every control step one sensor input (a depth image, or the latest
LiDAR message on a grid around the heightmap) and the last 5 steps of proprioception (IMU angular velocity and gravity,
joint positions and velocities). A CNN encodes the sensor, an MLP the proprioception; another MLP merges them into a
GRU, whose state is the memory across steps. The heightmap is decoded from the state and refined with the sensor input.

## Structure

```
sensors/         sensor models (see sensors/README.md)
robots/          sensor mounts and joint order per robot (see robots/README.md)
train/           data collection and training
tested/          closed-loop evaluation, visualization, keyboard teleop
deploy/          TorchScript export and ROS 2 node
example_policy/  Go2 vision policy to collect data with
```

Requires Isaac Lab with [basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab) (main).
Run everything from the repository root with Isaac Lab's Python.

## Usage

Collect (train seeds 42-45, test seeds 7-8, with and without `--forward_commands`; at most 500 envs):

```bash
python train/collect_sensors2heightmap.py --task=Locomotion-Go2-Rough-Vision --num_envs=496 \
  --checkpoint=example_policy/go2/concurrent_se_vision/exported/policy.pt \
  --sensor unitree_l2_lidar --sensor_errors bag --forward_commands --seed=42 --num_steps=1200 \
  --dataset_path=<data>/lidar_fwd_train_seed42
```

Train (`--proprio_noise` adds the robot's IMU and joint noise):

```bash
python train/terrain_reconstruction_mlp.py --dataset_path <data>/lidar_*train_seed* \
  --test_dataset_path <data>/lidar_*test_seed* --model_path <models>/lidar.pt --proprio_noise --wandb
```

Test in simulation:

```bash
python tested/teleop_keyboard.py --terrain_model_path <models>/lidar.pt
python tested/closed_loop_eval.py --terrain_model_path <models>/lidar.pt --heightmap predicted \
  --observation_noise robot --results <models>/closed_loop.csv
```

## Deploy

The deploy needs no Isaac Lab: a [pixi](https://pixi.prefix.dev/latest/installation/) environment with ROS 2 (humble
or lyrical) and CPU torch, and the robot's messages built in `deploy/ros2_ws`:

```bash
pixi run --manifest-path deploy/installation/pixi.toml -e humble build   # once, builds deploy/ros2_ws
pixi shell --manifest-path deploy/installation/pixi.toml -e humble       # or -e lyrical
```

```bash
python deploy/terrain_reconstructor.py --checkpoint <models>/lidar.pt --output <models>/lidar.ts
python deploy/run_sensor2heightmap_ros2.py --model <models>/lidar.ts   # on the robot's computer
```

The node reads the sensor, `/imu` and `/blind_state_legged` and publishes `/height_scan_markers` for basic-locomotion's
`run_controller_ros2.py`. Use it with the policy whose data trained the reconstructor.

A trained L2 reconstructor for the example policy: `example_policy/go2/concurrent_se_vision/reconstructor/`.

## License

BSD 3-Clause, see [LICENSE](LICENSE).

## Maintainers

- Valerio Belli (valerio.belli@iit.it)
