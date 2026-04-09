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
                {"step_duration_sec": 0.4},
                {"use_example_structure": False},
                # seed=-1 → fresh random seed each run.
                # Set to a specific non-negative integer for reproducibility.
                {"seed": -1},
            ],
        ),
    ])
