"""LiDAR points -> grid around the robot, the same code in simulation and on the robot."""

from __future__ import annotations

import math

import torch
from torch import Tensor

MARGIN = 2  # heightmap cells around the heightmap
HEIGHT_OFFSET = 0.5  # as in the policy heightmap: base z - ground z - 0.5
CHANNELS = ("mean height", "min height", "max height", "points", "age")
COUNT_SCALE = math.log1p(64.0)
AGE_SCALE = 0.1


def grid_spec(heightmap: dict) -> dict:
    """Grid over the heightmap plus MARGIN cells around it, cells half the heightmap's."""
    resolution = heightmap["resolution"]
    (size_x, size_y), (x, y) = heightmap["size"], heightmap["offset"][:2]
    border = (MARGIN + 0.5) * resolution
    x_range = (x - size_x / 2 - border, x + size_x / 2 + border)
    y_range = (y - size_y / 2 - border, y + size_y / 2 + border)
    shape = (round((y_range[1] - y_range[0]) / resolution * 2), round((x_range[1] - x_range[0]) / resolution * 2))
    return {"resolution": resolution / 2, "x_range": x_range, "y_range": y_range, "shape": shape,
            "crop": (MARGIN, MARGIN), "channels": CHANNELS}


def rotation_matrix(quat_xyzw, device: str | torch.device | None = None) -> Tensor:
    x, y, z, w = quat_xyzw
    return torch.tensor(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        device=device,
    )


def yaw_frame(projected_gravity: Tensor) -> Tensor:
    """Axes of the yaw-aligned frame in the base frame, as columns."""
    up = -projected_gravity / projected_gravity.norm(dim=-1, keepdim=True)
    forward = torch.zeros_like(up)
    forward[:, 0] = 1.0
    forward = forward - up[:, :1] * up
    forward = forward / forward.norm(dim=-1, keepdim=True)
    return torch.stack((forward, torch.cross(up, forward, dim=-1), up), dim=2)


def lidar_grid(
    points: Tensor,
    projected_gravity: Tensor,
    age: Tensor,
    rotation: Tensor,
    position: Tensor,
    grid: dict,
) -> Tensor:
    """Grid of N robots: (N, 5, rows, cols), mean/min/max height, point count and message age per cell.

    points are in the sensor frame, not finite where missing; age is the time since the end of the message."""
    num_robots, device = points.shape[0], points.device
    rows, cols = grid["shape"]
    resolution, (x_min, _), (y_min, _) = grid["resolution"], grid["x_range"], grid["y_range"]
    valid = torch.isfinite(points).all(dim=-1)
    points = torch.where(valid.unsqueeze(-1), points, 0.0)
    points_base = points @ rotation.transpose(-1, -2) + position.view(-1, 1, 3)
    points_yaw = points_base @ yaw_frame(projected_gravity)

    height = (-points_yaw[..., 2] - HEIGHT_OFFSET).clip(-1.0, 1.0)
    col = torch.floor((points_yaw[..., 0] - x_min) / resolution).long()
    row = torch.floor((points_yaw[..., 1] - y_min) / resolution).long()
    inside = valid & (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    cells = rows * cols
    robot = torch.arange(num_robots, device=device).view(-1, 1)
    index = torch.where(inside, robot * cells + row * cols + col, num_robots * cells).flatten()
    height = height.flatten()

    size = num_robots * cells + 1
    count = torch.zeros(size, device=device).index_add_(0, index, torch.ones_like(height))
    total = torch.zeros(size, device=device).index_add_(0, index, height)
    lowest = torch.full((size,), math.inf, device=device).scatter_reduce_(0, index, height, "amin")
    highest = torch.full((size,), -math.inf, device=device).scatter_reduce_(0, index, height, "amax")
    count, total, lowest, highest = (value[:-1].view(num_robots, rows, cols) for value in (count, total, lowest, highest))
    hit = count > 0
    channels = (
        torch.where(hit, total / count.clamp(min=1.0), 0.0),
        torch.where(hit, lowest, 0.0),
        torch.where(hit, highest, 0.0),
        torch.log1p(count) / COUNT_SCALE,
        (age / AGE_SCALE).view(-1, 1, 1).expand(num_robots, rows, cols),
    )
    return torch.stack(channels, dim=1)
