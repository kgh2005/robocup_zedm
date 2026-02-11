from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("robocup_zedm")

    model_path = os.path.join(pkg_share, "model", "best.pt")
    params_path = os.path.join(pkg_share, "config", "params.yaml")
    rviz_cfg = os.path.join(pkg_share, "config", "debug_image.rviz")

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("zed_wrapper"),
                "launch",
                "zed_camera.launch.py",
            ])
        ),
        launch_arguments={
            "camera_model": "zedm",
            "camera_name": "zedm",
            "publish_tf": "false",
            "publish_map_tf": "false",
            "publish_imu_tf": "false",
            "use_sim_time": "false",
        }.items(),
    )

    detection_node = Node(
        package="robocup_zedm",
        executable="detection_node",
        name="detection_node",
        output="screen",
        parameters=[
            params_path,
            {
                "model_path": model_path,
                "show_imshow": False,       # imshow 디버그
                "publish_debug": True,      # debug 이미지 publish
                "debug_topic": "/debug_image",
                "debug_rate_hz": 15.0,
            },
        ],
    )

    refiner_node = Node(
        package="robocup_zedm",
        executable="refiner_node",
        name="refiner_node",
        output="screen",
        parameters=[params_path],
    )

    pantilt_node = Node(
        package="robocup_zedm",
        executable="pantilt_node",
        name="pantilt_node",
        output="screen",
        parameters=[params_path],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
    )

    # RViz를 늦게 실행(토픽 생성 후 자동으로 붙기 쉬움)
    rviz_delayed = TimerAction(period=8.0, actions=[rviz])

    return LaunchDescription([
        zed_launch,
        detection_node,
        refiner_node,
        pantilt_node,
        rviz_delayed,
    ])
