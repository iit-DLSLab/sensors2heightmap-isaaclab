## Overview
An IsaacLab set of tools for heightmap reconstruction from lidar and camera


The module will generate a network that is able to predict heightmap from depth and lidar. So no need of distill your vision based policy!

## How to use

1. First, you need to train a locomotion policy in your env


2. Launch the collection step on your policy.

```bash
python train/collect_sensors2heightmap.py --task=Locomotion-Go2-Rough-Vision --num_envs=81 --checkpoint=./tested_policies/go2/concurrent_se_vision/exported/policy.pt --sensor lidar

python train/collect_sensors2heightmap.py --task=Locomotion-Go2-Rough-Vision --num_envs=81 --checkpoint=./example_policy/go2/concurrent_se_vision/exported/policy.pt --sensor depth
```

3. Train the network
```bash
python train/terrain_reconstruction_mlp.py
```

## Deploy

Currently, the deploy folder does not contain any utils for this.