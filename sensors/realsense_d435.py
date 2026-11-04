"""Intel RealSense D435 model: intrinsics, range and noise of the robot's camera."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils.configclass import configclass

from .sensor_model import SensorModel


@configclass
class RealSenseD435Cfg:
    """Depth stream at 848x480, decimated by 8 to 106x60 on the robot."""

    stream_width: int = 848
    stream_height: int = 480
    decimation: int = 8
    width: int = 106
    height: int = 60
    focal_length: float = 53.19  # px, after decimation (measured: fx = fy = 425.49 at 848x480)

    min_depth: float = 0.2
    max_depth: float = 10.0
    baseline: float = 0.0499
    noise_relative_std: float = 0.01  # std = noise_relative_std * z^2
    noise_max_dropout: float = 0.01  # fraction of random holes per image, up to

    def pattern_cfg(self) -> patterns.PinholeCameraPatternCfg:
        return patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                self.focal_length, 0.0, self.width / 2, 0.0, self.focal_length, self.height / 2, 0.0, 0.0, 1.0
            ],
            width=self.width,
            height=self.height,
        )


def seen_by_right_imager(depth: torch.Tensor, cfg: RealSenseD435Cfg) -> torch.Tensor:
    """Pixels whose point the right imager also sees."""
    inf = torch.tensor(float("inf"), device=depth.device, dtype=depth.dtype)
    column = torch.arange(depth.shape[-1], device=depth.device, dtype=depth.dtype) + 0.5
    x = torch.where(depth > 0.0, column - cfg.focal_length * cfg.baseline / depth.clamp_min(1e-6), -inf)
    landing = torch.where(depth > 0.0, x, inf).flip(-1).cummin(-1).values.flip(-1)
    right_of = torch.cat((landing[..., 1:], inf.expand_as(landing[..., :1])), -1)
    return (x >= 0.0) & (right_of > x)


def realsense_d435_noise(depth: torch.Tensor, cfg: RealSenseD435Cfg) -> torch.Tensor:
    """D435-like depth from the clean ray-cast one: noise, holes, stereo shadows, range limits."""
    noisy = depth + cfg.noise_relative_std * depth**2 * torch.randn_like(depth)
    rate = cfg.noise_max_dropout * torch.rand((*depth.shape[:-2], 1, 1), device=depth.device, dtype=depth.dtype)
    keep = torch.rand_like(depth) >= rate
    valid = (depth > 0.0) & keep & seen_by_right_imager(depth, cfg) & (noisy >= cfg.min_depth) & (noisy <= cfg.max_depth)
    return torch.where(valid, noisy, torch.zeros_like(noisy))


class RealSenseD435Model(SensorModel):
    """Latest depth image at every step, with the camera's noise."""

    KIND = "depth"
    ERRORS = {"sensor": {"noise": True}, "none": {"noise": False}}
    CAMERA = RealSenseD435Cfg()
    MAX_RANGE = 2.0

    @classmethod
    def sim_cfg(cls, prim_path: str, position, rotation, meshes: Sequence[str]) -> MultiMeshRayCasterCameraCfg:
        return MultiMeshRayCasterCameraCfg(
            prim_path=prim_path,
            update_period=1 / 60,
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=tuple(position), rot=tuple(rotation), convention="ros"),
            mesh_prim_paths=[
                "/World/ground", *[MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr=mesh) for mesh in meshes]
            ],
            pattern_cfg=cls.CAMERA.pattern_cfg(),
            data_types=["distance_to_image_plane"],
            max_distance=cls.CAMERA.max_depth,
            depth_clipping_behavior="zero",
            debug_vis=False,
        )

    def __init__(self, env, sensor, mount, heightmap: dict, noise: bool = True):
        super().__init__(env, sensor, mount, heightmap)
        self.noise = noise
        self.metadata = {
            "depth_image_size": (self.CAMERA.height, self.CAMERA.width),
            "depth_max_range": self.MAX_RANGE,
            "depth_focal_length": self.CAMERA.focal_length,
            "depth_noise": noise,
        }

    def __call__(self, dones: torch.Tensor) -> torch.Tensor:
        depth = self.sensor.data.output["distance_to_image_plane"]
        depth = torch.nan_to_num(getattr(depth, "torch", depth)[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        if self.noise:
            depth = realsense_d435_noise(depth, self.CAMERA)
        return depth.clip(0.0, self.MAX_RANGE).unsqueeze(1)
