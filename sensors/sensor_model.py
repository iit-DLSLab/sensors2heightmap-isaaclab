"""Base class of the sensor models, see README.md."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

REQUIRED_METADATA = {
    "lidar": ("lidar_grid", "lidar_message_period"),
    "depth": ("depth_image_size", "depth_max_range"),
}
TOPICS = {"lidar": "/utlidar/cloud", "depth": "/camera/camera/depth/image_rect_raw"}


class SensorModel(ABC):
    """A sensor simulated in Isaac Lab. Subclasses fill in KIND, ERRORS, sim_cfg, __init__ and __call__."""

    KIND: str
    ERRORS: dict[str, dict] = {"sensor": {}}

    @classmethod
    @abstractmethod
    def sim_cfg(cls, prim_path: str, position: Sequence[float], rotation: Sequence[float], meshes: Sequence[str]):
        """Isaac Lab sensor cfg at the given mount (quaternion x, y, z, w), seeing the terrain and meshes."""

    def __init__(self, env, sensor, mount, heightmap: dict):
        self.env, self.sensor, self.mount, self.heightmap = env.unwrapped, sensor, mount, heightmap
        self.metadata: dict = {}

    @abstractmethod
    def __call__(self, dones: torch.Tensor) -> torch.Tensor:
        """Reconstructor input at the current step, (num_envs, C, H, W)."""
