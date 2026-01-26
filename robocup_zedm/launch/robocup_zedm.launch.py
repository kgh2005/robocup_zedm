from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('robocup_zedm')

    model_path = os.path.join(pkg_share, 'model', 'best.pt')
    params_path = os.path.join(pkg_share, 'config', 'params.yaml')

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('zed_wrapper'),
                'launch',
                'zed_camera.launch.py',
            ])
        ),
        launch_arguments={
            'camera_model': 'zedm',
            'camera_name': 'zedm',

            # TF (원하면 true로)
            'publish_tf': 'false',
            'publish_map_tf': 'false',
            'publish_imu_tf': 'false',

            'use_sim_time': 'false',
        }.items()
    )

    detection_node = Node(
        package='robocup_zedm',
        executable='detection_node',
        name='detection_node',
        output='screen',
        parameters=[{
            'model_path': model_path,
            'rgb_topic': '/zedm/zed_node/rgb/color/rect/image',
            'device': 'cuda:0',   # GPU면
            'imgsz': 960,
            'show_debug': True,
        }]
    )

    refiner_node = Node(
        package='robocup_zedm',
        executable='refiner_node',
        name='refiner_node',
        output='screen',
        parameters=[
            params_path,  # YAML 먼저
            {
                'rgb_topic': '/zedm/zed_node/rgb/color/rect/image',
                'depth_topic': '/zedm/zed_node/depth/depth_registered',
            }
        ]
    )

    return LaunchDescription([
        zed_launch,
        detection_node,
        refiner_node
    ])
