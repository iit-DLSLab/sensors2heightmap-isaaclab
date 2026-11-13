"""Settings of the deploy, read by deploy/run_sensor2heightmap_ros2.py."""

# Hz at which the heightmap is published on /height_scan_markers.
# The reconstructor always steps at the step of its training (step_dt in the .ts, 50 Hz for the example model), as its
# memory and proprio history need: this only sets how often its last prediction is sent, at most once per step.
PUBLISH_FREQUENCY = 16.0

# Device of the reconstructor: "cuda" (GPU) or "cpu". With "cuda" and a torch without CUDA (the pixi environment of
# deploy/installation installs CPU torch) the node warns and runs on the cpu.
DEVICE = "cuda"

# LiDAR grid cells to empty before the network, a band across the whole width: (x_min, x_max) in m in the base frame, or
# None. (-0.15, 0.15) is the zone under the body, where the real L2 differs from the simulated one (tests of
# 2026-11-13).
EMPTY_ZONE_X = None
