""" Settings of the deploy, read by deploy/run_sensor2heightmap_ros2.py."""

## General parameters and Debugging 

# Hz at which the heightmap is published on /height_scan_markers.
PUBLISH_FREQUENCY = 16.0 # dont'change this unless you trained a new model with a different frequency
DEVICE = "cuda"
# It needs for debugging, don't change it unless you know what you are doing.
EMPTY_ZONE_X = None


## Training parameters
TEST_WARMUP_STEPS = 49
NON_FLAT_STD_THRESHOLD_M = 0.01  # a heightmap with a larger std counts as terrain
FRAME_KEYS = {"depth": "depth_data", "lidar": "lidar_grid"}
# tuples, not sets: str.startswith takes a tuple, and the noise terms are zipped in this order with their sizes
FUSION_MODULES = ("frame_reduction", "step_mlp")
PROPRIO_NOISE_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")
