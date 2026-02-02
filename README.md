# Multi-Agent Collective Construction (MACC) – RViz Simulation

This project is a Python + ROS 2 (RViz) simulation of **multi-agent collective construction** inspired by the paper:

> **Multi-agent Collective Construction using 3D Decomposition**  
> Akshaya Kesarimangalam Srinivasan et al., 2023  
> https://arxiv.org/abs/2309.00985

The goal is to visualize how multiple simple robots can collaboratively build a 3D voxel structure by decomposing it into independent substructures and executing a parallelizable construction schedule.

## Overview

The system:
1. Takes a **3D voxel structure** as input
2. **Decomposes** it into valid substructures using a shadow-region–based algorithm
3. Computes **dependency constraints** between substructures
4. Identifies which substructures can be built **in parallel**
5. Simulates multiple robots that:
   - Move on a voxel grid
   - Climb up/down one block at a time
   - Carry one block at a time
   - Place blocks while respecting structural validity
6. Visualizes the entire process in **RViz** using ROS 2 MarkerArray messages

This project focuses on **visualization and system-level behavior**, not MILP optimization or low-level control.

## Technologies Used

- **Python 3**
- **ROS 2 (Jazzy)**
- **RViz2**
- NumPy
- Matplotlib (for earlier non-ROS visualization)

## Key Features

- Structural decomposition using shadow regions  
- Dependency graph + parallel build grouping  
- Multi-robot simulation (configurable robot count)  
- Incremental voxel construction (no floating blocks)  
- Real-time RViz visualization of robots and blocks  
