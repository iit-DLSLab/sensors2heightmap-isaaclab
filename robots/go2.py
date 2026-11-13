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
    # x, y of the URDF's radar_joint; z (0.041 m on 5/11, 0.047 m on 6/11) and rotation estimated from the rosbags:
    # the floor seen by the L2 against the feet, and the L2 motion (ICP) against the IMU. Tilted 14 deg forward as
    # the URDF, but the cloud's frame is turned ~120 deg about the sensor axis from the URDF's radar link.
    "unitree_l2_lidar": {
        "position": (0.28945, 0.0, 0.045),
        "rotation": (-0.8579, 0.4977, -0.1167, 0.0520),
        "meshes": LEG_MESHES,  # the L2 sits inside the base meshes
    },
    # 20 deg down, ROS optical frame
    "realsense_d435": {
        "position": (0.33, 0.0, 0.08),
        "rotation": (-0.579228, 0.579228, -0.405579, 0.405579),
        "meshes": BASE_MESHES + LEG_MESHES,
    },
}

# legs of the URDF (m): hip joints at (+-x, +-y) from the base, thigh joint offset sideways, thigh and calf lengths
LEG_KINEMATICS = {"hip": (0.1934, 0.0465), "thigh_offset": 0.0955, "thigh": 0.213, "calf": 0.213}

# order of the joints in /blind_state_legged
MESSAGE_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
