## Overview

An Isaac Lab set of tools for heightmap reconstruction from a LiDAR or a depth camera: a network predicts the
heightmap of a vision locomotion policy from the sensor and the robot's proprioception, so the policy runs on the robot
without a height scanner. No need to distill your vision-based policy!

The model (`train/terrain_reconstruction_mlp.py`, `RecurrentTerrainReconstructor`): at every control step one sensor
input (a depth image, or the latest LiDAR message on a grid around the heightmap) and the proprioception of the last 5
steps (base angular velocity and projected gravity, joint positions and velocities: what the robot measures itself, no
commands or actions of the policy). A CNN encodes the sensor input, an MLP the proprioception; another MLP merges them
into the input of a GRU, whose state is the memory across steps (restarted after every reset). The heightmap is decoded
from the state and refined with the current sensor input.

## Structure

```
sensors/        the sensor models: unitree_l2_lidar.py, realsense_d435.py (see sensors/README.md)
robots/         one module per robot: sensor mounts, meshes, joint order; setup_robot.py (see robots/README.md)
train/          collect_sensors2heightmap.py (data), terrain_reconstruction_mlp.py (model and training), helpers
tested/         closed_loop_eval.py, visualize_heightmap_prediction_isaaclab.py, teleop_keyboard.py
deploy/         terrain_reconstructor.py (export, runtime), run_sensor2heightmap_ros2.py (ROS 2 node)
example_policy/ a Go2 vision policy (concurrent state estimator, 11 x 11 heightmap around the base) to collect with
```

The tasks and the robots come from [basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab)
(main), installed in Isaac Lab; the sensors do not: the pipeline mounts the chosen sensor model on the task's robot at
the mount of `robots/<robot>.py` (`train/sim_helpers.py`, `attach_sensor`), and collection, training, tests and deploy
use it.

A new sensor is a subclass of `SensorModel` in `sensors/` (see [sensors/README.md](sensors/README.md)); a new robot is a
new `robots/<robot>.py`, written from its URDF, its USD and a rosbag by `robots/setup_robot.py` (see
[robots/README.md](robots/README.md)).

## Pipeline

All commands from the repository root, with Isaac Lab's Python. `--sensor` is `unitree_l2_lidar` or `realsense_d435`,
`--robot` defaults to `go2`.

1. Train a vision locomotion policy in basic-locomotion-isaaclab (or use `example_policy/`).

2. Record sequences with the policy (every step of every env): training sets and test sets with other seeds, both with
forward commands only (`--forward_commands`: 0.3-0.8 m/s, no stops) and with the task's. At most 500 envs: beyond,
the task keeps the first 500 standing still. With the L2, `--sensor_errors bag` adds the larger errors of the Go2's
rosbag: a reconstructor trained without them takes the real L2's errors on flat ground for terrain.

```bash
python train/collect_sensors2heightmap.py --task=Locomotion-Go2-Rough-Vision --num_envs=496 \
  --checkpoint=example_policy/go2/concurrent_se_vision/exported/policy.pt --sensor unitree_l2_lidar --sensor_errors bag \
  --forward_commands --seed=42 --num_steps=1200 --dataset_path=<data>/lidar_fwd_train_seed42
```

The recipe of the reference model: seeds 42-45 for training (1200 steps), 7-8 for testing (1000 steps forward, 600
with the task's commands), 496 envs. A step of 496 envs takes 4.5 MB with the L2 (grid of 30 x 30 cells for the
11 x 11 heightmap), 6.3 MB with the D435 (106 x 60 image): about 57 and 80 GB for the whole recipe.

3. Train the reconstructor (the defaults are the reference recipe: segments of 96 steps, 100 epochs). The recordings
keep the proprioception clean; `--proprio_noise` adds the robot's reading noise in training (the task's
`observation_noise_std`: a white noise per reading, a bias per segment). The test is scored with clean and with noisy
proprioception:

```bash
python train/terrain_reconstruction_mlp.py --dataset_path <data>/lidar_*train_seed* \
  --test_dataset_path <data>/lidar_*test_seed* --model_path <models>/lidar_rnn_mlp.pt --proprio_noise --wandb
```

4. Test it in simulation, with its sensor mounted where it was in its data and the sensor's errors of the data: the
policy walking on the predicted map (`tested/closed_loop_eval.py --heightmap true|predicted|flat`, `--scenario` for
scripted commands, `--observation_noise robot` for noisy proprioception as on the robot), its map drawn on the terrain
(`tested/visualize_heightmap_prediction_isaaclab.py`, `--record` for a video), one robot driven with the keyboard
(`tested/teleop_keyboard.py`, noisy proprioception for the policy and the reconstructor unless
`--no_observation_noise`):

```bash
python tested/closed_loop_eval.py --terrain_model_path <models>/lidar_rnn_mlp.pt --heightmap predicted --num_envs 496 \
  --observation_noise robot --results <models>/closed_loop.csv
python tested/teleop_keyboard.py --terrain_model_path <models>/lidar_rnn_mlp.pt
```

## Deploy

Export the trained model to TorchScript; the file carries in its config.json what the robot needs (sensor, mount,
grid or image size, joints, heightmap):

```bash
python deploy/terrain_reconstructor.py --checkpoint <models>/lidar_rnn_mlp.pt --output <models>/lidar_rnn_mlp.ts
```

On the robot's computer, next to basic-locomotion's controller (`deploy/run_controller_ros2.py`, which comes from that
repository), with torch and numpy:

```bash
python deploy/run_sensor2heightmap_ros2.py --model <models>/lidar_rnn_mlp.ts
```

The node subscribes to the sensor (`/utlidar/cloud` for a LiDAR, `/camera/camera/depth/image_rect_raw` for a depth
camera, `--topic` for another one), `/imu` and `/blind_state_legged`, runs the reconstructor at the policy rate (~3 ms
per step on a CPU) and publishes `/height_scan_markers`, the heightmap the controller reads for a vision policy (the
same message as basic-locomotion's `run_simulator_ros2.py`). The policy must be the one whose data trained the
reconstructor (same heightmap). `deploy/terrain_reconstructor.py` also runs without ROS (`TerrainReconstructor`).

## Sensor models

**Unitree L2** (`sensors/unitree_l2_lidar.py`): scan pattern, timing and range errors identified from a Go2 rosbag:
non-repetitive scan, arcs of 300 firings from horizon to horizon over the zenith at 215.8 Hz, head turning at 4.37
rev/s; messages of 64.9 ms (15.4 Hz, ~4200 points), each point in the L2 frame at its firing time (no motion
compensation); range noise of 1-5 mm, a systematic error of up to 4 cm oscillating with the distance, 0.34% missing
points, nothing closer than 0.2 m. The legs hide the ground and return points, as on the robot. On top of the sensor
(`UnitreeL2LidarModel`): each message arrives 0-40 ms after its end, and the grid is computed with a mount off by up to
1 cm / 1 deg and a gravity direction off by up to 1 deg, drawn per episode. The grid (`sensors/lidar_grid.py`): cells
of half the heightmap's, over the heightmap and 2 of its cells around it, 5 channels (mean, min and max height of the
points, number of points, age of the message). A message covers only part of the heightmap: the memory joins
consecutive ones.

The L2 mount on the Go2 (`robots/go2.py`) is calibrated on the rosbag of 2026-10-30 (robot standing on flat ground):
the ground plane of the cloud against the IMU gravity and the feet from the joint angles. Unlike the Unitree URDF, the
L2 x axis points forward-right, and with the URDF rotation the real ground comes out tilted by 29 deg. Check it again
with a new bag if the sensor or its mount change:

```bash
python robots/setup_robot.py --name_robot go2 --bag ~/Desktop/rosbag2_2026_10_30-11_37_52
```

**RealSense D435** (`sensors/realsense_d435.py`): the decimated depth stream of the robot's camera (848 x 480 -> 106
x 60), with its noise, its holes and the bands that the right imager does not see; the robot decimates its stream the
same way (`deploy/terrain_reconstructor.py`, `decimate_depth`).
