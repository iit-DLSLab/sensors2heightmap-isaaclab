"""Settings of the deploy, read by deploy/run_sensor2heightmap_ros2.py."""

# Hz at which the heightmap is published on /height_scan_markers.
PUBLISH_FREQUENCY = 16.0 #dont'change this unless you trained a new model with a different frequency

DEVICE = "cuda"

# It needs for debugging, don't change it unless you know what you are doing.
EMPTY_ZONE_X = None

# Relief of the published map multiplied around the floor under the stance feet (1.0 = the reconstructor's map). The
# real L2 maps draw steps 2-4 cm too low while climbing; with them the policy lifts its feet as over a map with 0.75 of
# the relief (closed-loop test, 21/11/2026). Also --relief_gain, or live: ros2 param set /Sensor2HeightmapROS2 relief_gain 1.3
RELIEF_GAIN = 1.0
