"""Sensor mounts and joint order of the Unitree Go2."""

from pathlib import Path

BASE_PRIM_PATH = "/World/envs/env_.*/Robot/Geometry/base"

BASE_MESHES = ["/World/envs/env_.*/Robot/Geometry/base/base_[0-9]"]
LEG_MESHES = [
    "/World/envs/env_.*/Robot/Geometry/base/[FR][LR]_hip/hip_[0-9]",
    "/World/envs/env_.*/Robot/Geometry/base/[FR][LR]_hip/[FR][LR]_thigh/thigh(_mirror)?_[0-9]",
    "/World/envs/env_.*/Robot/Geometry/base/[FR][LR]_hip/[FR][LR]_thigh/[FR][LR]_calf/(calf(_mirror)?_[0-9]|foot)",
]
# This is the trunk shell, excluding the head. The stock head (x > 0.25 m) carries the L1 under the chin (base_0 and
# base_1, near the URDF radar point), but this Go2 is fitted with an L2. The shell also encloses the L2's optical centre, so rays
# originating there would hit the robot. We leave base_4 out: rays from the L2 optical centre (z = -0.008 m relative
# to the cloud origin at 0.035 m) pass beneath the trunk, and including the shell hides the floor behind x = -0.3 m.

## In a nutshell: do not change anything here unless you have an idea of what you are doing ## 

TRUNK_MESH = {"prim_expr": "/World/envs/env_.*/Robot/Geometry/base/base_4", "x_max": 0.25}

## Occlusion mask: I'm working on a better mask of the L2's rays. TODO: Come out with a better mask. 
L2_OCCLUSION_MASK = str(Path(__file__).with_name("go2_l2_occlusion.npz"))

# position (m) and rotation (x, y, z, w) of the sensor frame in the base frame
SENSORS = {
    # L2 positions are based on the user manual by Unitree; the published cloud frame was estimated from the rosbags.
    # Again, do not change anything here unless you have an idea of what you are doing.
    "unitree_l2_lidar": {
        "position": (0.28945, 0.0, 0.045),
        "rotation": (-0.8579, 0.4977, -0.1167, 0.0520),
        "meshes": LEG_MESHES,  # the trunk would stop the rays, see TRUNK_MESH
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
FOOT_RADIUS = 0.022  # m, the foot sphere: the ground is this far below the foot centre of LEG_KINEMATICS

# order of the joints in /blind_state_legged
MESSAGE_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
