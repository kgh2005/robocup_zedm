# robocup_zedm

ZED Mini (ZEDM) vision package for ROS2 Humble.

## What it does
- **detection_node**: RGB → YOLO(Ultralytics) → publishes `vision_interfaces/BoundingBox`
- **refiner_node**: Depth + CameraInfo + BoundingBox → publishes `vision` / `vision_feature`

## Build
```bash
cd ~/colcon_ws
colcon build --symlink-install
source install/setup.bash
```

## Run
```bash
ros2 launch robocup_zedm robocup_zedm.launch.py
```