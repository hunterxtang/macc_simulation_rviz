# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See the workspace-root `~/ros2_ws/CLAUDE.md` for the full reference (architecture, pipeline, conventions). This file covers package-level build/test commands only.

## Build and Run

```bash
# From workspace root
colcon build --packages-select macc_rviz
source ~/ros2_ws/install/setup.bash

ros2 launch macc_rviz macc.launch.py
ros2 run macc_rviz macc_rviz_sim --ros-args -p seed:=42 -p num_robots:=6

# Standalone (no ROS)
python3 macc_rviz/main.py [--seed 42]
```

## Tests

```bash
colcon test --packages-select macc_rviz
colcon test-result --verbose
python3 -m pytest test/test_flake8.py -v
```

Style/lint only (flake8, pep257, copyright) — no functional unit tests.
