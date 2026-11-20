"""Settings of the deploy, read by deploy/run_sensor2heightmap_ros2.py."""

# Hz at which the heightmap is published on /height_scan_markers.
PUBLISH_FREQUENCY = 16.0 #dont'change this unless you trained a new model with a different frequency

DEVICE = "cuda"

# It needs for debugging, don't change it unless you know what you are doing.
EMPTY_ZONE_X = None
