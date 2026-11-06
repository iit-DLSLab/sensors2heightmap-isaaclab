"""Unitree L2 LiDAR model. Scan pattern, timing and errors identified from a Go2 rosbag."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils.configclass import configclass

from . import lidar_grid
from .sensor_model import SensorModel


def unitree_l2_scan(
    cfg: "UnitreeL2PatternCfg", t_end: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray directions (L2 frame) and firing times of the message that ends at t_end."""
    device = t_end.device
    t_start = t_end - cfg.message_period
    arcs_per_message = int(math.ceil(cfg.message_period * cfg.arc_frequency)) + 1
    arc_ids = torch.floor(t_start * cfg.arc_frequency)[:, None] + torch.arange(
        arcs_per_message, dtype=torch.float64, device=device
    )
    sample_ids = torch.arange(cfg.samples_per_arc, dtype=torch.float64, device=device)
    t = (arc_ids[:, :, None] / cfg.arc_frequency + sample_ids * cfg.sample_period).flatten(1)
    valid = (t >= t_start[:, None]) & (t < t_end[:, None])

    azimuth = torch.remainder(t * math.radians(cfg.rotation_speed), 2.0 * math.pi).float()
    half_span = 0.5 * (cfg.samples_per_arc - 1)
    x = (sample_ids.float() - half_span) / half_span
    arc_angle = torch.deg2rad(90.0 + sum(c * x ** (2 * i + 1) for i, c in enumerate(cfg.arc_angle_coefficients)))
    arc_angle = arc_angle.repeat(arcs_per_message)

    cos_arc = torch.cos(arc_angle)
    ray_directions = torch.stack(
        (cos_arc * torch.cos(azimuth), cos_arc * torch.sin(azimuth), torch.sin(arc_angle).expand_as(azimuth)), dim=-1
    )
    return ray_directions, t - t_start[:, None], valid


def unitree_l2_non_repetitive_pattern(
    cfg: "UnitreeL2PatternCfg", device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    t_end = torch.tensor([cfg.message_period], dtype=torch.float64, device=device)
    ray_directions, _, _ = unitree_l2_scan(cfg, t_end)
    ray_directions = ray_directions[0]
    return torch.zeros_like(ray_directions), ray_directions


@configclass
class UnitreeL2PatternCfg:
    """L2 scan pattern and range errors."""

    func: Callable = unitree_l2_non_repetitive_pattern

    sample_period: float = 7.726e-6  # s between firings
    samples_per_arc: int = 300  # horizon -> zenith -> opposite horizon
    arc_frequency: float = 215.83  # Hz
    rotation_speed: float = 1573.1  # deg/s of the head about z
    arc_angle_coefficients: tuple[float, ...] = (78.98, 10.92)  # angle along the arc: 90 + c1 x + c3 x^3 deg
    message_period: float = 0.0649  # s, 15.4 Hz

    min_range: float = 0.2
    dropout_probability: float = 0.0034
    range_noise_std_per_meter: float = 0.0045
    range_noise_std_range: tuple[float, float] = (0.001, 0.006)
    range_resolution: float = 0.001
    # systematic error a sin(2 pi d / wavelength + phase), drawn at every reset
    range_bias_amplitude_range: tuple[float, float] = (0.0, 0.04)
    range_bias_wavelength_range: tuple[float, float] = (0.1, 0.5)
    # each ray cast from the pose at its firing time (points are not motion compensated on the robot)
    simulate_motion_distortion: bool = True


class _UnitreeL2LidarMixin:
    """Moves the rays along the L2 scan and applies the range errors."""

    def update(self, dt: float, force_recompute: bool = False):
        # scan every message period, also with lazy sensor updates
        super().update(dt, force_recompute=True)

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        super().reset(env_ids, env_mask)
        if env_ids is not None:
            env_ids = torch.as_tensor(env_ids, device=self._device)
        elif env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        else:
            env_ids = torch.arange(self._view_count, device=self._device)
        self._sample_scanner_state(env_ids)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        if self.cfg.ray_alignment != "base":
            raise ValueError(f"UnitreeL2Lidar requires ray_alignment='base'. Received: {self.cfg.ray_alignment}.")
        num_envs, device = self._view_count, self._device

        self._offset_quat_l2 = torch.tensor(list(self.cfg.offset.rot), device=device).repeat(self.num_rays, 1)
        self._scanner_clock = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self._range_bias_amplitude = torch.zeros(num_envs, device=device)
        self._range_bias_wavenumber = torch.zeros(num_envs, device=device)
        self._range_bias_phase = torch.zeros(num_envs, device=device)
        self._last_update_time = torch.full((num_envs,), -math.inf, dtype=torch.float64, device=device)
        self._last_update_pos_w = torch.zeros(num_envs, 3, device=device)
        self._last_update_quat_w = torch.zeros(num_envs, 4, device=device)
        self._firing_lag = torch.zeros(num_envs, self.num_rays, device=device)
        self._updating_env_ids = None

        self.points_l2 = torch.full((num_envs, self.num_rays, 3), math.inf, device=device)
        self.point_times = torch.zeros(num_envs, self.num_rays, device=device)
        self.point_valid = torch.zeros(num_envs, self.num_rays, dtype=torch.bool, device=device)
        self._sample_scanner_state(torch.arange(num_envs, device=device))

    def _sample_scanner_state(self, env_ids: torch.Tensor):
        pattern_cfg = self.cfg.pattern_cfg
        num_envs = len(env_ids)
        self._scanner_clock[env_ids] = 1000.0 * torch.rand(num_envs, dtype=torch.float64, device=self._device)
        self._range_bias_amplitude[env_ids] = math_utils.sample_uniform(
            *pattern_cfg.range_bias_amplitude_range, (num_envs,), device=self._device
        )
        wavelength = math_utils.sample_uniform(
            *pattern_cfg.range_bias_wavelength_range, (num_envs,), device=self._device
        )
        self._range_bias_wavenumber[env_ids] = 2.0 * math.pi / wavelength
        self._range_bias_phase[env_ids] = math_utils.sample_uniform(0.0, 2.0 * math.pi, (num_envs,), self._device)
        self._last_update_time[env_ids] = -math.inf

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return super()._update_buffers_impl(env_mask)
        pattern_cfg = self.cfg.pattern_cfg

        t_end = self._scanner_clock[env_ids] + wp.to_torch(self._timestamp)[env_ids].double()
        ray_directions_l2, point_times, in_message = unitree_l2_scan(pattern_cfg, t_end)
        self.ray_directions.torch[env_ids] = math_utils.quat_apply(
            self._offset_quat_l2.expand(len(env_ids), -1, -1), ray_directions_l2
        )
        self._firing_lag[env_ids] = (pattern_cfg.message_period - point_times).float()
        self._updating_env_ids = env_ids
        self._updating_t_end = t_end

        super()._update_buffers_impl(env_mask)
        self._apply_range_model(env_ids, ray_directions_l2, point_times, in_message)

        self._last_update_time[env_ids] = t_end
        self._last_update_pos_w[env_ids] = wp.to_torch(self._data._pos_w)[env_ids]
        self._last_update_quat_w[env_ids] = wp.to_torch(self._data._quat_w)[env_ids]
        self._updating_env_ids = None

    def _update_ray_infos(self, env_mask: wp.array):
        super()._update_ray_infos(env_mask)
        env_ids = self._updating_env_ids
        if env_ids is None or not self.cfg.pattern_cfg.simulate_motion_distortion:
            return

        elapsed = self._updating_t_end - self._last_update_time[env_ids]
        interpolable = elapsed <= 2.0 * self.cfg.pattern_cfg.message_period
        env_ids, elapsed = env_ids[interpolable], elapsed[interpolable].float()
        if len(env_ids) == 0:
            return
        pos_w = wp.to_torch(self._data._pos_w)[env_ids]
        quat_w = wp.to_torch(self._data._quat_w)[env_ids]
        rotation_to_last = math_utils.axis_angle_from_quat(
            math_utils.quat_mul(math_utils.quat_inv(quat_w), self._last_update_quat_w[env_ids])
        )
        fraction = (self._firing_lag[env_ids] / elapsed[:, None]).clamp(max=1.0).unsqueeze(-1)
        rotation = fraction * rotation_to_last[:, None, :]
        quat_w = quat_w[:, None, :].expand(-1, self.num_rays, -1)

        ray_starts = self.ray_starts.torch[env_ids]
        ray_starts_w = wp.to_torch(self._ray_starts_w)
        ray_starts_w[env_ids] += fraction * (self._last_update_pos_w[env_ids] - pos_w)[:, None, :] + (
            math_utils.quat_apply(quat_w, _rotate(rotation, ray_starts) - ray_starts)
        )
        wp.to_torch(self._ray_directions_w)[env_ids] = math_utils.quat_apply(
            quat_w, _rotate(rotation, self.ray_directions.torch[env_ids])
        )

    def _apply_range_model(
        self,
        env_ids: torch.Tensor,
        ray_directions_l2: torch.Tensor,
        point_times: torch.Tensor,
        in_message: torch.Tensor,
    ):
        pattern_cfg = self.cfg.pattern_cfg
        ray_hits_w = wp.to_torch(self._data._ray_hits_w)
        ray_starts_w = wp.to_torch(self._ray_starts_w)[env_ids]
        ray_directions_w = wp.to_torch(self._ray_directions_w)[env_ids]

        distance = torch.linalg.norm(ray_hits_w[env_ids] - ray_starts_w, dim=-1)
        hit = torch.isfinite(distance)
        distance = torch.where(hit, distance, 0.0)

        noise_std = (pattern_cfg.range_noise_std_per_meter * distance).clamp(*pattern_cfg.range_noise_std_range)
        bias = self._range_bias_amplitude[env_ids, None] * torch.sin(
            self._range_bias_wavenumber[env_ids, None] * distance + self._range_bias_phase[env_ids, None]
        )
        measured = distance + bias + noise_std * torch.randn_like(distance)
        if pattern_cfg.range_resolution > 0.0:
            measured = torch.round(measured / pattern_cfg.range_resolution) * pattern_cfg.range_resolution

        valid = (
            in_message
            & hit
            & (measured >= pattern_cfg.min_range)
            & (torch.rand_like(distance) >= pattern_cfg.dropout_probability)
        ).unsqueeze(-1)
        measured = measured.unsqueeze(-1)
        ray_hits_w[env_ids] = torch.where(valid, ray_starts_w + measured * ray_directions_w, math.inf)
        self.points_l2[env_ids] = torch.where(valid, measured * ray_directions_l2, math.inf)
        self.point_times[env_ids] = point_times.float()
        self.point_valid[env_ids] = valid.squeeze(-1)


def _rotate(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotation, dim=-1, keepdim=True)
    axis = rotation / angle.clamp(min=1e-9)
    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    return (
        vectors * cos_angle
        + torch.cross(axis, vectors, dim=-1) * sin_angle
        + axis * (axis * vectors).sum(dim=-1, keepdim=True) * (1.0 - cos_angle)
    )


class UnitreeL2Lidar:
    """L2 on top of the MultiMeshRayCaster of the active physics backend."""

    _sensor_classes: dict[type, type] = {}

    def __new__(cls, cfg):
        from isaaclab.sensors import MultiMeshRayCaster

        backend_class = MultiMeshRayCaster.resolve_class(cfg)
        if backend_class not in cls._sensor_classes:
            cls._sensor_classes[backend_class] = type(
                f"UnitreeL2Lidar{backend_class.__name__}", (_UnitreeL2LidarMixin, backend_class), {}
            )
        return cls._sensor_classes[backend_class](cfg)


def _small_rotations(num: int, max_angle: float, device) -> torch.Tensor:
    rotation_vector = (2.0 * torch.rand(num, 3, device=device) - 1.0) * max_angle
    angle = rotation_vector.norm(dim=-1, keepdim=True).unsqueeze(-1)
    k = torch.zeros(num, 3, 3, device=device)
    axis = rotation_vector / rotation_vector.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    k = k - k.transpose(1, 2)
    return torch.eye(3, device=device) + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * k @ k


class UnitreeL2LidarModel(SensorModel):
    """L2 messages as the robot gets them (latency, mount and gravity errors), put on the grid."""

    KIND = "lidar"

    @classmethod
    def sim_cfg(cls, prim_path: str, position, rotation, meshes: Sequence[str]) -> MultiMeshRayCasterCfg:
        pattern_cfg = UnitreeL2PatternCfg()
        return MultiMeshRayCasterCfg(
            class_type=UnitreeL2Lidar,
            prim_path=prim_path,
            update_period=pattern_cfg.message_period,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=tuple(position), rot=tuple(rotation)),
            ray_alignment="base",
            pattern_cfg=pattern_cfg,
            mesh_prim_paths=[
                "/World/ground", *[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=mesh) for mesh in meshes]
            ],
            max_distance=2.0,
            debug_vis=False,
        )

    def __init__(self, env, sensor, mount, heightmap: dict, latency=(0.0, 0.04), mount_position_error: float = 0.01,
                 mount_angle_error: float = math.radians(1.0), gravity_angle_error: float = math.radians(1.0)):
        super().__init__(env, sensor, mount, heightmap)
        self.robot = self.env.scene.articulations["robot"]
        self.grid = lidar_grid.grid_spec(heightmap)
        self.latency, self.position_error = latency, mount_position_error
        self.mount_angle_error, self.gravity_angle_error = mount_angle_error, gravity_angle_error
        num, rays, device = self.env.num_envs, self.sensor.num_rays, self.env.device
        self.mount_position = torch.tensor(mount[0], device=device)
        self.mount_rotation = lidar_grid.rotation_matrix(mount[1], device)
        self.points = torch.full((num, rays, 3), math.inf, device=device)
        self.end = torch.zeros(num, device=device)
        self.pending = torch.full((num, rays, 3), math.inf, device=device)
        self.pending_end = torch.zeros(num, device=device)
        self.arrival = torch.full((num,), math.inf, device=device)
        self.new_message = torch.zeros(num, dtype=torch.bool, device=device)  # a message arrived at this step
        self.rotation = self.mount_rotation.repeat(num, 1, 1)
        self.position = self.mount_position.repeat(num, 1)
        self.gravity_rotation = torch.eye(3, device=device).repeat(num, 1, 1)
        self.last_update = self._sensor_time("_timestamp_last_update").clone()
        self.reset(torch.arange(num, device=device))
        self.metadata = {
            "lidar_grid": self.grid,
            "lidar_message_period": self.sensor.cfg.pattern_cfg.message_period,
            "lidar_latency": tuple(latency),
            "lidar_mount_position_error": mount_position_error,
            "lidar_mount_angle_error": mount_angle_error,
            "lidar_gravity_angle_error": gravity_angle_error,
        }

    def _sensor_time(self, name: str) -> torch.Tensor:
        return wp.to_torch(getattr(self.sensor, name)).float()

    def reset(self, env_ids: torch.Tensor) -> None:
        num, device = len(env_ids), self.env.device
        self.points[env_ids] = math.inf
        self.arrival[env_ids] = math.inf
        self.end[env_ids] = -1.0
        self.last_update[env_ids] = self._sensor_time("_timestamp_last_update")[env_ids]
        self.rotation[env_ids] = _small_rotations(num, self.mount_angle_error, device) @ self.mount_rotation
        self.position[env_ids] = self.mount_position + (2.0 * torch.rand(num, 3, device=device) - 1.0) * self.position_error
        self.gravity_rotation[env_ids] = _small_rotations(num, self.gravity_angle_error, device)

    def __call__(self, dones: torch.Tensor) -> torch.Tensor:
        dones = dones.bool()
        if dones.any():
            self.reset(dones.nonzero().flatten())
        now, last_update = self._sensor_time("_timestamp"), self._sensor_time("_timestamp_last_update")
        new = (last_update != self.last_update) & ~dones
        self.last_update = last_update.clone()
        if new.any():
            latency = self.latency[0] + (self.latency[1] - self.latency[0]) * torch.rand(int(new.sum()), device=now.device)
            self.pending[new] = self.sensor.points_l2[new]
            self.pending_end[new] = last_update[new]
            self.arrival[new] = last_update[new] + latency
        arrived = self.arrival <= now + 1e-6
        self.new_message = arrived
        if arrived.any():
            self.points[arrived] = self.pending[arrived]
            self.end[arrived] = self.pending_end[arrived]
            self.arrival[arrived] = math.inf
        projected_gravity = self.robot.data.projected_gravity_b
        projected_gravity = getattr(projected_gravity, "torch", projected_gravity)
        gravity = (self.gravity_rotation @ projected_gravity.unsqueeze(-1)).squeeze(-1)
        age = torch.where(self.end >= 0.0, now - self.end, 1.0)
        return lidar_grid.lidar_grid(self.points, gravity, age, self.rotation, self.position, self.grid)
