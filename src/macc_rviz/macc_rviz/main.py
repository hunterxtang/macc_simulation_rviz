import argparse

from macc_rviz.structure_utils import create_small_random_structure
from macc_rviz.decomposition import decompose_structure, order_substructures
from macc_rviz.parallel import find_parallel_groups
from macc_rviz.visualization import show_construction_process


def main():
    parser = argparse.ArgumentParser(
        description="MACC standalone matplotlib visualisation"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for the random structure (omit for a new structure each run)"
    )
    args = parser.parse_args()

    print("=== MULTI-AGENT COLLECTIVE CONSTRUCTION DEMO ===")
    print(f"Seed: {args.seed if args.seed is not None else 'random'}")

    structure = create_small_random_structure(seed=args.seed)
    print(f"Structure shape (Z,Y,X): {structure.shape}  blocks: {int(structure.sum())}")

    print("\nDecomposing structure...")
    substructures = decompose_structure(structure)
    print(f"Found {len(substructures)} substructures.")

    order, deps = order_substructures(substructures)
    print(f"Sequential build order: {order}")

    groups = find_parallel_groups(substructures, deps)
    print(f"Parallel build groups:  {groups}")

    show_construction_process(structure, substructures, groups)

    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
