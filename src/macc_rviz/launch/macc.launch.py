from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    planner_arg = DeclareLaunchArgument(
        "planner",
        default_value="heuristic",
        description="Action planner: 'heuristic' (reactive FSM) or 'cbs'.",
    )
    cbs_max_t_arg = DeclareLaunchArgument(
        "cbs_max_t",
        default_value="400",
        description="CBS/A* hard time horizon per call.",
    )
    cbs_branch_limit_arg = DeclareLaunchArgument(
        "cbs_branch_limit",
        default_value="500",
        description="Max CBS branches before serial fallback. 50 is too "
                    "tight for 7x7x4 @ 4 robots; set to 0 to force fallback.",
    )
    num_robots_arg = DeclareLaunchArgument(
        "num_robots",
        default_value="4",
    )
    seed_arg = DeclareLaunchArgument(
        "seed",
        default_value="-1",
        description="-1 → fresh random each run; non-negative fixes layout.",
    )

    return LaunchDescription([
        planner_arg,
        cbs_max_t_arg,
        cbs_branch_limit_arg,
        num_robots_arg,
        seed_arg,
        Node(
            package="macc_rviz",
            executable="macc_rviz_sim",
            output="screen",
            parameters=[
                {"num_robots": LaunchConfiguration("num_robots")},
                {"step_duration_sec": 0.4},
                {"use_example_structure": False},
                {"seed": LaunchConfiguration("seed")},
                {"planner": LaunchConfiguration("planner")},
                {"cbs_max_t": LaunchConfiguration("cbs_max_t")},
                {"cbs_branch_limit": LaunchConfiguration("cbs_branch_limit")},
            ],
        ),
    ])
