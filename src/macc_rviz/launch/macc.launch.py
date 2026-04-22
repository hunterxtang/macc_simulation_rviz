from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    planner_arg = DeclareLaunchArgument(
        "planner",
        default_value="milp",
        description="Action planner: 'heuristic' (reactive FSM), 'cbs', or 'milp'.",
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
    milp_per_t_time_limit_arg = DeclareLaunchArgument(
        "milp_per_t_time_limit",
        default_value="60.0",
        description="Gurobi TimeLimit (seconds) for each T in the MILP sweep.",
    )
    milp_total_time_limit_arg = DeclareLaunchArgument(
        "milp_total_time_limit",
        default_value="60.0",
        description="Soft total budget (seconds) per MILP group solve.",
    )
    milp_mip_gap_arg = DeclareLaunchArgument(
        "milp_mip_gap",
        default_value="0.0",
        description="Gurobi MIPGap for MILP solves (0.0 = prove optimal).",
    )
    milp_T_max_arg = DeclareLaunchArgument(
        "milp_T_max",
        default_value="40",
        description="Hard upper bound on MILP horizon sweep per group.",
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
    use_example_structure_arg = DeclareLaunchArgument(
        "use_example_structure",
        default_value="true",
        description="Use the hand-built 13-block example structure instead of a random one.",
    )
    grid_x_arg = DeclareLaunchArgument(
        "grid_x",
        default_value="5",
        description="Random-structure grid X extent (only used when use_example_structure=false).",
    )
    grid_y_arg = DeclareLaunchArgument(
        "grid_y",
        default_value="5",
        description="Random-structure grid Y extent (only used when use_example_structure=false).",
    )
    grid_z_arg = DeclareLaunchArgument(
        "grid_z",
        default_value="3",
        description="Random-structure grid Z extent (only used when use_example_structure=false).",
    )
    density_arg = DeclareLaunchArgument(
        "density",
        default_value="0.25",
        description="Per-column height probability; expected blocks ≈ grid_x*grid_y*grid_z*density.",
    )

    # LaunchConfiguration always resolves to a string. For non-string node
    # parameters, the resolved string is silently rejected by ROS2's
    # parameter system (type mismatch) and the node falls back to its
    # declared default. Wrapping with ParameterValue(value_type=...) coerces
    # the string before dispatch so the override actually lands.
    def _typed(name, value_type):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription([
        planner_arg,
        cbs_max_t_arg,
        cbs_branch_limit_arg,
        milp_per_t_time_limit_arg,
        milp_total_time_limit_arg,
        milp_mip_gap_arg,
        milp_T_max_arg,
        num_robots_arg,
        seed_arg,
        use_example_structure_arg,
        grid_x_arg,
        grid_y_arg,
        grid_z_arg,
        density_arg,
        Node(
            package="macc_rviz",
            executable="macc_rviz_sim",
            output="screen",
            parameters=[
                {"num_robots": _typed("num_robots", int)},
                {"step_duration_sec": 0.4},
                {"use_example_structure": _typed("use_example_structure", bool)},
                {"seed": _typed("seed", int)},
                {"grid_x": _typed("grid_x", int)},
                {"grid_y": _typed("grid_y", int)},
                {"grid_z": _typed("grid_z", int)},
                {"density": _typed("density", float)},
                {"planner": LaunchConfiguration("planner")},
                {"cbs_max_t": _typed("cbs_max_t", int)},
                {"cbs_branch_limit": _typed("cbs_branch_limit", int)},
                {"milp_per_t_time_limit": _typed("milp_per_t_time_limit", float)},
                {"milp_total_time_limit": _typed("milp_total_time_limit", float)},
                {"milp_mip_gap": _typed("milp_mip_gap", float)},
                {"milp_T_max": _typed("milp_T_max", int)},
            ],
        ),
    ])
