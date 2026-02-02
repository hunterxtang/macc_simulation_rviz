from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="macc_rviz",
            executable="macc_rviz_sim",
            output="screen",
            parameters=[
                {"num_robots": 4},
                {"step_hz": 10.0},
                {"use_example_structure": False},
                {"seed": 42},
            ],
        ),
    ])
