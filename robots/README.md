# Robots

`robots/<robot>.py` tells the pipeline where a robot carries its sensors, what of the robot they see and the order of
its joints on the real robot. Collection, tests and deploy read it with `--robot <robot>`.

| | |
|---|---|
| `BASE_PRIM_PATH` | the body the sensors are attached to, in the simulation |
| `BASE_MESHES`, `LEG_MESHES` | the robot's visual meshes, which the sensors see |
| `SENSORS` | per sensor model: `position`, `rotation` (quaternion x, y, z, w) in the base frame, `meshes` it sees |
| `MESSAGE_JOINT_NAMES` | the joints in the order of the robot's `/blind_state_legged` |

`setup_robot.py` writes it from the robot's files, or checks it.

## Add a robot

1. Get the robot's files:
   - its URDF (plain, not xacro);
   - its USD, from basic-locomotion: `robot_model/<robot>/generated_usd/<robot>.usda`;
   - a rosbag (sqlite3) of the robot standing still on flat ground for ~10 s, with the sensors, the IMU and the joints,
     e.g. `ros2 bag record -s sqlite3 /utlidar/cloud /lowstate` on the Go2, or the depth topic, `/imu`,
     `/blind_state_legged`.
2. Find in the URDF the link of each sensor: the frame its data are in (a camera: its optical frame, z forward, x right,
   y down).
3. Run, with one `--sensor MODEL=LINK` per sensor (`MODEL`: a file of `sensors/`):

   ```bash
   python robots/setup_robot.py --urdf <robot>.urdf --usd <robot>.usda --bag <rosbag> \
     --sensor unitree_l2_lidar=radar --name_robot <robot>
   ```

4. Read the report. For each sensor and mount (the URDF's, the one fitted on the bag): the tilt of the floor (0 = the
   mount is right) and the floor's depth below the base (= the base height from the feet). The file gets the fitted
   mount.
5. Open `robots/<robot>.py` and check the lines marked `CHECK`.
6. Use it: `python train/collect_sensors2heightmap.py ... --robot <robot>`.

Without `--bag` the mounts are the URDF's; without `--usd` there are no meshes (the sensors see only the terrain).

## Check a robot

After moving a sensor, or with a new rosbag:

```bash
python robots/setup_robot.py --name_robot go2 --bag <rosbag> [--urdf go2.urdf]
```

An existing `robots/<robot>.py` is never overwritten: its mounts are checked next to the URDF's and the fitted one.
With `--urdf` the base height comes from the feet, and the fitted mount gets its height too. `--topic MODEL=TOPIC` if a
sensor is not on the default topic (`/utlidar/cloud`, `/camera/camera/depth/image_rect_raw`).

Good to know:
- a flat floor does not show the rotation about the vertical: the fit keeps the sensor leaning forward or backward, as
  in the URDF or the robot file, never sideways;
- the IMU frame is taken as the base frame;
- depth images must be 16UC1 (mm); they are decimated as on the robot, to the `CAMERA` of the sensor model.
