from macc_rviz.structure_utils import create_random_structure, create_example_structure
from macc_rviz.decomposition import decompose_structure, compute_dependencies, order_substructures
from macc_rviz.parallel import find_parallel_groups
from macc_rviz.visualization import show_construction_process


def main():
    print("=== MULTI-AGENT COLLECTIVE CONSTRUCTION DEMO ===")

    # --- choose which structure to visualize ---
    # Option 1: Simple tower (1 substructure)
    # structure = create_example_structure()

    # Option 2: Random 3D structure (multiple substructures, better for visualization)
    structure = create_random_structure(x=7, y=7, z=4, density=0.4, seed=42)

    # --- decomposition phase ---
    print("\nDecomposing structure...")
    substructures = decompose_structure(structure)
    print(f"Found {len(substructures)} substructures.")

    # --- compute dependencies and build order ---
    order, deps = order_substructures(substructures)
    print(f"\nSequential build order: {order}")

    # --- compute parallel groups ---
    groups = find_parallel_groups(substructures, deps)
    print(f"Parallel build groups: {groups}")

    # --- visualize everything ---
    show_construction_process(structure, substructures, groups)

    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
