<div align="center">

# sensors2heightmap-isaaclab

![Video](https://img.shields.io/badge/Video-coming%20soon-lightgrey.svg)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-green.svg)](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

</div>

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
example_policy/  Go2 vision policy to collect data with and a trained reconstructor model
```

Requires [Isaac Lab 3.0.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) with [basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab) (main).
Run everything from the repository root with Isaac Lab's Python.

## Usage

Collect (train seeds 42-45, test seeds 7-8, with and without `--forward_commands`; at most 500 envs). The L2 in
`robots/go2.py` is the one the reference model's data were collected with:

```bash
python train/collect_sensors2heightmap.py --task=Locomotion-Go2-Rough-Vision --num_envs=496 \
  --checkpoint=example_policy/go2/concurrent_se_vision/exported/policy.pt \
  --sensor unitree_l2_lidar --forward_commands --seed=42 --num_steps=1200 \
  --dataset_path=<data>/lidar_fwd_train_seed42
```

Train (the defaults are the reference recipe, `l2_relief_w3`; the `.ts` is exported next to the `.pt`):

```bash
python train/terrain_reconstruction_mlp.py --dataset_path <data>/lidar_*train_seed* \
  --test_dataset_path <data>/lidar_*test_seed* --model_path <models>/lidar.pt --wandb
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
`run_controller_ros2.py`. Use it with the policy whose data trained the reconstructor. `deploy/config.py` sets how often
the heightmap is published (`PUBLISH_FREQUENCY`; the reconstructor always steps at the step of its training) and where
the reconstructor runs (`DEVICE`, `"cuda"` or `"cpu"`; GPU needs a CUDA torch in the pixi environment).
`EMPTY_ZONE_X` empties a band of the LiDAR grid before the network, e.g. `(-0.15, 0.15)` under the body. Models trained
with a LiDAR window run with their window; the `.ts` carries it.

The node runs torch on one thread (`--threads`). The controller shares the CPU, so give it one torch and one onnxruntime
thread as well, or both loops slow down: the policy then runs far below 50 Hz and the robot falls.

The reference L2 reconstructor for the example policy: `example_policy/go2/concurrent_se_vision/reconstructor/l2_relief_w3.ts`.

## License

BSD 3-Clause, see [LICENSE](LICENSE).

## Author

- Valerio Belli, developer and maintainer (valerio.belli@iit.it)

## Citation

If you use this code, please cite it using [CITATION.cff](CITATION.cff).
