import matplotlib.pyplot as plt
import numpy as np
import time


def set_axes_equal(ax):
    """
    Forces equal scaling on 3D axes so voxels appear cubic.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range]) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])


def show_structure_voxels(structure, title="Structure"):
    """
    Displays the full 3D structure using solid cubic voxels.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(structure, facecolors='skyblue', edgecolor='k', alpha=0.9)
    ax.set_title(title)
    ax.view_init(elev=25, azim=45)
    set_axes_equal(ax)
    plt.show()


def show_substructures(substructures):
    """
    Displays each decomposed substructure in a distinct color.
    """
    colors = ['red', 'green', 'blue', 'orange', 'purple',
              'cyan', 'magenta', 'yellow', 'lime', 'navy']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, s in enumerate(substructures):
        color = colors[i % len(colors)]
        ax.voxels(s, facecolors=color, edgecolor='k', alpha=0.9)

    ax.set_title("Substructures (Decomposed)")
    ax.view_init(elev=25, azim=45)
    set_axes_equal(ax)
    plt.show()


def animate_build(structure, substructures, groups, delay=0.08):
    """
    Animates the construction process, substructure-by-substructure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=45)
    plt.ion()
    plt.show()

    colors = ['red', 'green', 'blue', 'orange', 'purple',
              'cyan', 'magenta', 'yellow', 'lime', 'navy']

    built = np.zeros_like(structure)
    print("\n=== STARTING VISUAL BUILD ANIMATION ===")

    for g_idx, group in enumerate(groups):
        print(f"\nParallel Layer {g_idx + 1}: Substructures {group}")
        for sub_idx in group:
            sub = substructures[sub_idx]
            color = colors[sub_idx % len(colors)]
            z, y, x = np.where(sub == 1)

            for zi, yi, xi in zip(z, y, x):
                built[zi, yi, xi] = 1
                ax.cla()
                ax.voxels(built, facecolors='lightgrey', edgecolor='k', alpha=0.3)
                ax.voxels(sub * (built == 1), facecolors=color, edgecolor='k', alpha=1.0)
                ax.set_title(f"Building Substructure {sub_idx + 1} (Layer {g_idx + 1})")
                ax.view_init(elev=25, azim=45)
                set_axes_equal(ax)
                plt.draw()
                plt.pause(delay)
        plt.pause(0.5)

    print("\n=== BUILD COMPLETE ===")
    plt.ioff()
    plt.show()


def show_construction_process(structure, substructures, groups):
    """
    Wrapper: shows the target, decomposed, and animated build in sequence.
    """
    show_structure_voxels(structure, "Target Structure")
    show_substructures(substructures)
    animate_build(structure, substructures, groups, delay=0.08)
