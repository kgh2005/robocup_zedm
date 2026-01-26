# vision_interfaces

ROS2 interface package for custom vision messages.

## Messages
- `msg/BoundingBox.msg`

You can add more messages by placing new `.msg` files under `msg/` (e.g., `GoalPost.msg`, `LineFeature.msg`).

## Build
From workspace root:

```bash
cd ~/colcon_ws
colcon build --symlink-install
source install/setup.bash
```

## Add a new message (quick guide)
1. Create a new file in `msg/`:
   - `vision_interfaces/msg/NewMessage.msg`
2. Register it in `CMakeLists.txt` inside `rosidl_generate_interfaces(...)`
3. Build again:
   - `colcon build --symlink-install`