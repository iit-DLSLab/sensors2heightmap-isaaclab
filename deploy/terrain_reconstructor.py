"""Export of a trained reconstructor to TorchScript, and its runtime on the robot (torch only).

    python deploy/terrain_reconstructor.py --checkpoint <model>.pt --output <model>.ts
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensors import lidar_grid  # noqa: E402


def decimate_depth(depth: torch.Tensor, height: int, width: int, min_valid_fraction: float = 0.5) -> torch.Tensor:
    """Median decimation of a depth image (m, 0 = invalid) to (height, width), as the robot does it."""
    factor = depth.shape[0] // height
    if factor < 1 or depth.shape[1] // width < factor:
        raise ValueError(f"A {depth.shape[1]}x{depth.shape[0]} depth image does not decimate to {width}x{height}.")
    # crop the extra rows and columns on the top left, where the right imager doesn't see
    depth = depth[depth.shape[0] - factor * height :, depth.shape[1] - factor * width :]
    size = factor * factor
    blocks = depth.reshape(height, factor, width, factor).transpose(1, 2).reshape(height, width, size)
    blocks = blocks.sort(dim=-1).values  # invalid (0) first
    n = (blocks > 0.0).sum(dim=-1, keepdim=True)
    low = blocks.gather(-1, (size - n + (n - 1).div(2, rounding_mode="floor")).clamp(max=size - 1))
    high = blocks.gather(-1, (size - n + n // 2).clamp(max=size - 1))
    return torch.where(n >= min_valid_fraction * size, 0.5 * (low + high), 0.0)[..., 0]


class TerrainReconstructor:
    def __init__(self, path: str | Path, joint_names: list[str], device: str = "cpu"):
        extra_files = {"config.json": ""}
        self.model = torch.jit.load(str(path), map_location=device, _extra_files=extra_files)
        self.config = json.loads(extra_files["config.json"])
        self.device = device
        proprio = self.config["proprio"]
        self.num_message_joints = len(joint_names)
        self.joint_order = torch.tensor([list(joint_names).index(name) for name in proprio["joint_names"]], device=device)
        self.default_joint_pos = torch.tensor(proprio["default_joint_pos"], device=device)
        self.kind = self.config["sensor"]
        if self.kind == "lidar":
            self.message_period = self.config["lidar"]["message_period"]
            position, quat_xyzw = self.config["lidar"]["mount"]
            self.rotation = lidar_grid.rotation_matrix(quat_xyzw, device)
            self.position = torch.tensor(position, device=device)
        self.reset()
        if torch.device(device).type == "cuda":
            self._warm_up()

    def _warm_up(self, steps: int = 3) -> None:
        """The first CUDA calls take ~0.2 s (cuDNN, kernels): spent here, not at the first message."""
        # the traced GRU has no flatten_parameters(): cuDNN compacts its weights at every call, cheap at this size
        warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")
        joints = torch.zeros(self.num_message_joints)
        for _ in range(steps):
            self.step(torch.zeros(3), torch.tensor((0.0, 0.0, -1.0)), joints, joints, 0.0)
        torch.cuda.synchronize(self.device)
        self.reset()

    def reset(self) -> None:
        frames = self.config["proprio"]["history"]
        self.history = torch.zeros(frames, self.config["proprio_dim"] // frames, device=self.device)
        self.hidden = torch.zeros(1, 1, self.config["hidden_dim"], device=self.device)
        self.points = torch.full((1, 1, 3), math.inf, device=self.device)
        self.message_end = None
        if self.kind == "depth":
            self.depth = torch.zeros(1, 1, *self.config["depth"]["image_size"], device=self.device)
        self.fresh = True

    def sensor_message(self, data, stamp: float) -> None:
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        if self.kind == "lidar":
            self.points = data.reshape(1, -1, 3)
            self.message_end = stamp + self.message_period
        else:
            depth = self.config["depth"]
            self.depth = decimate_depth(data, *depth["image_size"]).clip(0.0, depth["max_range"])[None, None]

    @torch.no_grad()
    def step(self, angular_velocity, projected_gravity, joint_position, joint_velocity, now: float) -> torch.Tensor:
        def tensor(value) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten()

        gravity = tensor(projected_gravity)
        frame = torch.cat((tensor(angular_velocity), gravity, tensor(joint_position)[self.joint_order] - self.default_joint_pos,
                           tensor(joint_velocity)[self.joint_order]))
        self.history = torch.cat((self.history[1:], frame[None]))
        if self.kind == "lidar":
            age = now - self.message_end if self.message_end is not None else 1.0
            sensor_input = lidar_grid.lidar_grid(
                self.points, gravity[None], torch.tensor([age], device=self.device), self.rotation, self.position,
                self.config["lidar"]["grid"],
            )
        else:
            sensor_input = self.depth
        heights, hidden = self.model(sensor_input, self.history.reshape(1, -1), self.hidden)
        if self.fresh:  # as in training, the memory starts from the second step
            self.fresh = False
            return torch.full_like(heights, self.config["flat_height"])
        self.hidden = hidden
        return heights


class _Step(torch.nn.Module):
    """One step with plain tensors in and out, for tracing."""

    def __init__(self, model: torch.nn.Module, mean: float, std: float):
        super().__init__()
        self.model, self.mean, self.std = model, mean, std

    def forward(self, sensor_input: torch.Tensor, proprio: torch.Tensor, hidden: torch.Tensor):
        output = self.model(depth_data=sensor_input[:, None], robot_info=proprio[:, None], hidden_state=hidden)
        heights = (output.refined_heightmap.reshape(-1) * self.std + self.mean).clamp(-1.0, 1.0)
        return heights, output.hidden_state


def export(checkpoint_path: str, output_path: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    from terrain_reconstruction_mlp import build_terrain_reconstructor

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config, metadata = checkpoint["model_config"], checkpoint["dataset_metadata"]
    model = build_terrain_reconstructor(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalization = checkpoint["target_normalization"]
    step = _Step(model, normalization["mean"], normalization["std"]).eval()
    hidden_dim = model.memory.hidden_size
    kind = model_config["sensor"]
    if kind == "lidar":
        input_shape = (len(metadata["lidar_grid"]["channels"]), *metadata["lidar_grid"]["shape"])
    else:
        input_shape = (1, *metadata["depth_image_size"])
    example = (torch.zeros(1, *input_shape), torch.zeros(1, model_config["proprio_dim"]), torch.zeros(1, 1, hidden_dim))
    with torch.no_grad():
        traced = torch.jit.trace(step, example, check_trace=False)
        inputs = (torch.randn_like(example[0]), torch.randn_like(example[1]), torch.randn_like(example[2]))
        difference = (traced(*inputs)[0] - step(*inputs)[0]).abs().max().item()
    config = {
        "sensor": kind,
        "sensor_model": metadata["sensor_model"],
        "robot": metadata["robot"],
        "hidden_dim": hidden_dim,
        "proprio_dim": model_config["proprio_dim"],
        "proprio": {
            "history": metadata["proprio_history"],
            "joint_names": metadata["proprio_joint_names"],
            "default_joint_pos": metadata["proprio_default_joint_pos"],
        },
        "heightmap": {**metadata["heightmap"], "shape": list(model_config["heightmap_size"])},
        "flat_height": normalization["mean"],
        "step_dt": metadata["step_dt"],
        "checkpoint": str(Path(checkpoint_path).resolve()),
    }
    if kind == "lidar":
        config["lidar"] = {"grid": metadata["lidar_grid"], "mount": metadata["sensor_mount"],
                           "message_period": metadata["lidar_message_period"]}
    else:
        config["depth"] = {"image_size": metadata["depth_image_size"], "max_range": metadata["depth_max_range"]}
    torch.jit.save(traced, output_path, _extra_files={"config.json": json.dumps(config)})
    print(f"[INFO] Exported {checkpoint_path} to {output_path} (max difference to the model {difference:.1e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a terrain reconstructor to TorchScript for the robot.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of train/terrain_reconstruction_mlp.py.")
    parser.add_argument("--output", required=True, help="TorchScript file to write.")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
