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
NO_MESSAGE_AGE = 1.0  # s, the age of a grid without a message


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


def window_input(
    grids: Tensor, dt: Tensor, valid: Tensor, mode: str, grid: dict, poses: Tensor | None = None
) -> Tensor:
    """Sensor input from the grids of the last K messages, the current one last.

    grids (N, K, 5, rows, cols); dt (N, K) seconds from each message to the current one; valid (N, K) False where there
    was no message yet. stacked: the K grids as they are, (N, 5 K, rows, cols), the network aligns them. aligned: the
    older grids moved into the current one with the base poses (N, K, 4: x, y, z, yaw, from the odometry) and merged
    into one grid (N, 5, rows, cols)."""
    age, points = CHANNELS.index("age"), CHANNELS.index("points")
    grids = grids.clone()
    grids[:, :, age] += (dt / AGE_SCALE)[..., None, None]
    empty = torch.zeros_like(grids[:, :1])
    empty[:, :, age] = NO_MESSAGE_AGE / AGE_SCALE
    grids = torch.where(valid[..., None, None, None], grids, empty)
    if mode == "stacked":
        return grids.flatten(1, 2)
    if mode != "aligned":
        raise ValueError(f"Unknown window mode: {mode}")

    num, k, channels, rows, cols = grids.shape
    resolution, (x_min, _), (y_min, _) = grid["resolution"], grid["x_range"], grid["y_range"]
    yaw = poses[:, -1, 3:4]
    shift = poses[:, :-1, :3] - poses[:, -1:, :3]  # older bases from the current one, world axes
    d_x = torch.cos(yaw) * shift[..., 0] + torch.sin(yaw) * shift[..., 1]
    d_y = -torch.sin(yaw) * shift[..., 0] + torch.cos(yaw) * shift[..., 1]
    d_z, turn = shift[..., 2], yaw - poses[:, :-1, 3]
    # centres of the current cells in the frame of each older message: R(turn) (x - d)
    x = x_min + resolution * (torch.arange(cols, device=grids.device) + 0.5)
    y = y_min + resolution * (torch.arange(rows, device=grids.device) + 0.5)
    x, y = x.view(1, 1, 1, cols) - d_x[..., None, None], y.view(1, 1, rows, 1) - d_y[..., None, None]
    cos, sin = torch.cos(turn)[..., None, None], torch.sin(turn)[..., None, None]
    col = torch.floor((cos * x - sin * y - x_min) / resolution).long()
    row = torch.floor((sin * x + cos * y - y_min) / resolution).long()
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    index = (row.clamp(0, rows - 1) * cols + col.clamp(0, cols - 1)).flatten(2)
    older = grids[:, :-1].flatten(3).gather(3, index[:, :, None].expand(-1, -1, channels, -1))
    older = older.view(num, k - 1, channels, rows, cols)
    older = torch.where(inside[:, :, None], older, empty)
    hit = older[:, :, points] > 0
    moved = (older[:, :, :points] - d_z[..., None, None, None]).clip(-1.0, 1.0)  # heights from the current base
    older[:, :, :points] = torch.where(hit[:, :, None], moved, 0.0)

    stack = torch.cat((older, grids[:, -1:]), dim=1)
    count = torch.expm1(stack[:, :, points] * COUNT_SCALE)
    hit = count > 0.5
    total = count.sum(dim=1)
    merged = torch.stack(
        (
            (count * stack[:, :, 0]).sum(dim=1) / total.clamp(min=1.0),
            torch.where(hit, stack[:, :, 1], math.inf).amin(dim=1),
            torch.where(hit, stack[:, :, 2], -math.inf).amax(dim=1),
            torch.log1p(total) / COUNT_SCALE,
            torch.where(hit, stack[:, :, age], math.inf).amin(dim=1),
        ),
        dim=1,
    )
    seen = total > 0.5
    merged[:, 1:3] = torch.where(seen[:, None], merged[:, 1:3], 0.0)
    merged[:, age] = torch.where(seen, merged[:, age], grids[:, -1, age])
    return merged
