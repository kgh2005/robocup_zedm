# ZED MINI Stereo Vision

**Vision** package for the **ZED MINI Stereo (ZEDM)**.

## Development Environment

| Component | Version |
|---|---|
| **OS** | Ubuntu 22.04 |
| **ROS** | Humble Hawksbill |
| **Camera** | ZED MINI Stereo |

## Packages

- **robocup_zedm**: YOLO-based detection + depth-based refinement
- **vision_interfaces**: custom message definitions (`BoundingBox.msg`)

## Build

```bash
cd ~/colcon_ws
colcon build --symlink-install
source install/setup.bash

python3 -m pip install -U pip
python3 -m pip install transitions
python3 -m pip install opencv-python
python3 -m pip install numpy
python3 -m pip install ultralytics

```

## Run

```bash
ros2 launch robocup_zedm robocup_zedm.launch.py
```