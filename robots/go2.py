"""Sensor mounts and joint order of the Unitree Go2."""

BASE_PRIM_PATH = "/World/envs/env_.*/Robot/Geometry/base"

BASE_MESHES = ["/World/envs/env_.*/Robot/Geometry/base/base_[0-9]"]
LEG_MESHES = [
    "/World/envs/env_.*/Robot/Geometry/base/[FR][LR]_hip/hip_[0-9]",
    "/World/envs/env_.*/Robot/Geometry/base/[FR][LR]_hip/[FR][LR]_thigh/thigh(_mirror)?_[0-9]",
    "/World/envs/env_.*/Robot/Geometry/base/[FR][LR]_hip/[FR][LR]_thigh/[FR][LR]_calf/(calf(_mirror)?_[0-9]|foot)",
]

# position (m) and rotation (x, y, z, w) of the sensor frame in the base frame
SENSORS = {
    # as in basic-locomotion's go2_env_cfg.py: upside down, 15 deg forward (rotation of the URDF's radar_joint)
    "unitree_l2_lidar": {
        "position": (0.31, 0.0, 0.02),
        "rotation": (0.0, 0.99134, 0.0, 0.13132),
        "meshes": LEG_MESHES,  # the L2 sits inside the base meshes
    },
    # 20 deg down, ROS optical frame
    "realsense_d435": {
        "position": (0.33, 0.0, 0.08),
        "rotation": (-0.579228, 0.579228, -0.405579, 0.405579),
        "meshes": BASE_MESHES + LEG_MESHES,
    },
}

# order of the joints in /blind_state_legged
MESSAGE_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
