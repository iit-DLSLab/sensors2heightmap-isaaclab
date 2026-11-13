"""Base motion from the proprioception alone, the same code in simulation and on the robot: the yaw from the gyro, the
translation from the feet on the ground (the lowest ones, from the joint angles).

A reading is one control step of proprioception: angular velocity (3), projected gravity (3), joint positions minus
the default ones and joint velocities, the joints leg by leg (FL, FR, RL, RR), hip, thigh and calf."""

from __future__ import annotations

import torch
from torch import Tensor

from .lidar_grid import yaw_frame

LEGS = ("FL", "FR", "RL", "RR")
STANCE_MARGIN = 0.02  # m: feet this close to the lowest one are on the ground


def feet(joints: Tensor, kinematics: dict) -> Tensor:
    """Feet in the base frame (..., 4, 3) from the joint angles (..., 12)."""
    q = joints.view(*joints.shape[:-1], 4, 3)
    hip_x, hip_y = kinematics["hip"]
    side = torch.tensor((1.0, -1.0, 1.0, -1.0), device=joints.device)
    front = torch.tensor((1.0, 1.0, -1.0, -1.0), device=joints.device)
    thigh, calf = kinematics["thigh"], kinematics["calf"]
    knee, ankle = q[..., 1], q[..., 1] + q[..., 2]
    x = -thigh * torch.sin(knee) - calf * torch.sin(ankle)  # in the hip frame, before the abduction
    z = -thigh * torch.cos(knee) - calf * torch.cos(ankle)
    y = side * kinematics["thigh_offset"]
    cos, sin = torch.cos(q[..., 0]), torch.sin(q[..., 0])
    return torch.stack((front * hip_x + x, side * hip_y + cos * y - sin * z, sin * y + cos * z), dim=-1)


def _feet_level(readings: Tensor, default_joint_pos: Tensor, kinematics: dict) -> tuple[Tensor, Tensor]:
    """Feet in the yaw-aligned base frame (..., 4, 3) and which are on the ground (..., 4)."""
    gravity, joints = readings[..., 3:6], readings[..., 6:18] + default_joint_pos
    frame = yaw_frame(gravity.reshape(-1, 3)).view(*gravity.shape[:-1], 3, 3)
    level = feet(joints, kinematics) @ frame
    stance = level[..., 2] < level[..., 2].amin(dim=-1, keepdim=True) + STANCE_MARGIN
    return level, stance


def _turned(points: Tensor, yaw: Tensor) -> Tensor:
    cos, sin = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    return torch.stack((cos * points[..., 0] - sin * points[..., 1], sin * points[..., 0] + cos * points[..., 1],
                        points[..., 2]), dim=-1)


def odometry(readings: Tensor, default_joint_pos: Tensor, kinematics: dict, step_dt: float) -> Tensor:
    """Base poses (S, 4: x, y, z, yaw) along S consecutive control steps from the first one, readings (S, 30)."""
    up = -readings[:, 3:6] / readings[:, 3:6].norm(dim=-1, keepdim=True)
    yaw_rate = (up * readings[:, :3]).sum(dim=-1)
    yaw = torch.cat((yaw_rate.new_zeros(1), torch.cumsum(yaw_rate[:-1] * step_dt, dim=0)))
    level, stance = _feet_level(readings, default_joint_pos, kinematics)
    world = _turned(level, yaw)
    both = (stance[1:] & stance[:-1]).float()
    step = ((world[:-1] - world[1:]) * both[..., None]).sum(dim=1) / both.sum(dim=1, keepdim=True).clamp(min=1.0)
    position = torch.cat((step.new_zeros(1, 3), torch.cumsum(step, dim=0)))
    return torch.cat((position, yaw[:, None]), dim=-1)


class Odometry:
    """The same estimate one control step at a time, for N robots."""

    def __init__(self, num: int, default_joint_pos: Tensor, kinematics: dict, step_dt: float, device):
        self.default_joint_pos, self.kinematics, self.step_dt = default_joint_pos.to(device), kinematics, step_dt
        self.pose = torch.zeros(num, 4, device=device)
        self.world = torch.zeros(num, 4, 3, device=device)
        self.stance = torch.zeros(num, 4, dtype=torch.bool, device=device)

    def step(self, readings: Tensor, restart: Tensor) -> Tensor:
        """Pose (N, 4) after this reading (N, 30); restart (N,) starts a robot again from zero."""
        level, stance = _feet_level(readings, self.default_joint_pos, self.kinematics)
        self.pose[restart] = 0.0
        world = _turned(level, self.pose[:, 3])
        both = (stance & self.stance & ~restart[:, None]).float()
        step = ((self.world - world) * both[..., None]).sum(dim=1) / both.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.pose[:, :3] += step
        up = -readings[:, 3:6] / readings[:, 3:6].norm(dim=-1, keepdim=True)
        pose = self.pose.clone()
        self.pose[:, 3] += (up * readings[:, :3]).sum(dim=-1) * self.step_dt
        self.world, self.stance = world, stance
        return pose
