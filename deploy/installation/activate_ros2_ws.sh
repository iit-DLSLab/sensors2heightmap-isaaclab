# Sources dls2_interface once `pixi run build` has built it in deploy/ros2_ws.
# local_setup.sh holds the install path, so it works from any POSIX shell.
if [ -f "$PIXI_PROJECT_ROOT/../ros2_ws/install/local_setup.sh" ]; then
    . "$PIXI_PROJECT_ROOT/../ros2_ws/install/local_setup.sh"
fi
