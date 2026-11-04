# Sensors

A sensor model says how a real sensor is simulated and what the terrain reconstructor gets from it. Collection,
training, tests and deploy work with any model: `--sensor <file name>`.

| File | Sensor | Kind |
|---|---|---|
| `unitree_l2_lidar.py` | Unitree L2 LiDAR | lidar |
| `realsense_d435.py` | RealSense D435 depth camera | depth |

## Add a sensor

1. Create `sensors/<name>.py` with a subclass of `SensorModel` (`sensor_model.py`) and fill in:

   | | |
   |---|---|
   | `KIND` | `"lidar"` (point cloud, put on the grid of `lidar_grid.py`) or `"depth"` (depth image) |
   | `ERRORS` | named sets of extra errors, keyword arguments of `__init__` (`--sensor_errors <set>`) |
   | `sim_cfg()` | the Isaac Lab sensor at the mount it gets |
   | `__init__()` | call `super().__init__(env, sensor, mount, heightmap)`, set `self.metadata` |
   | `__call__(dones)` | the reconstructor's input at every control step, `(num_envs, C, H, W)` |
   | `metadata` | lidar: `lidar_grid`, `lidar_message_period`; depth: `depth_image_size`, `depth_max_range` |

2. Mount it on the robot: `SENSORS["<name>"]` in `robots/<robot>.py` (position, rotation, meshes it sees; see
   `robots/README.md`).
3. Use it: `python train/collect_sensors2heightmap.py ... --sensor <name>`.

On the robot, `deploy/` rebuilds the same input from `metadata`: LiDAR points (sensor frame) on the same grid, or the
depth stream decimated to `depth_image_size`.

## Examples

A variant of a sensor: subclass its model, change only what differs.

```python
from .realsense_d435 import RealSenseD435Cfg, RealSenseD435Model


class RealSenseD457Model(RealSenseD435Model):
    CAMERA = RealSenseD435Cfg(focal_length=54.39, baseline=0.0952, min_depth=0.25)
```

A new sensor:

```python
from .sensor_model import SensorModel


class MyCameraModel(SensorModel):
    KIND = "depth"

    @classmethod
    def sim_cfg(cls, prim_path, position, rotation, meshes):
        return ...  # e.g. a MultiMeshRayCasterCameraCfg: intrinsics, range, mount, terrain and meshes

    def __init__(self, env, sensor, mount, heightmap):
        super().__init__(env, sensor, mount, heightmap)
        self.metadata = {"depth_image_size": (60, 106), "depth_max_range": 2.0}

    def __call__(self, dones):
        return ...  # (num_envs, 1, 60, 106), m, 0 without depth
```
