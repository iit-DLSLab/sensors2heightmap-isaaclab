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
    # calibrated on the rosbag of 2026-10-30 (setup_robot.py); the URDF rotation is wrong for the real L2
    "unitree_l2_lidar": {
        "position": (0.28945, 0.0, 0.0507),
        "rotation": (-0.86610, 0.48649, -0.10267, 0.05160),
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
