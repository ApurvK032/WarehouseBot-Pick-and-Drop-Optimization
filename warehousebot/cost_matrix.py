"""
Cost matrix generation for WarehouseBot.
Computes shortest-path travel costs between all stops using A* pathfinding.
"""

import math
from typing import List, Tuple, Dict

from .astar import a_star
from .warehouse import make_static_demo, build_stops


def compute_cost_matrix(
    grid: List[List[int]],
    coords: List[Tuple[int, int]],
    heuristic: str = "manhattan"
) -> Dict:
    """
    Compute shortest-path cost matrix between all stop coordinates.
    
    Args:
        grid: 2D grid (0=free, 1=obstacle)
        coords: List of (r, c) coordinates for all stops
        heuristic: "manhattan" or "euclidean"
    
    Returns:
        dict with keys:
            - "matrix": NxN 2D list of costs (ints or math.inf)
            - "expanded_total": total nodes expanded across all A* calls
            - "astar_calls": number of A* runs performed
    
    Notes:
        - Diagonal entries are 0
        - Matrix is symmetric (undirected graph)
        - Only computes for i < j, then mirrors to j,i
    """
    n = len(coords)
    
    # Initialize matrix with zeros on diagonal
    matrix = [[0 if i == j else math.inf for j in range(n)] for i in range(n)]
    
    expanded_total = 0
    astar_calls = 0
    
    # Compute only upper triangle (i < j), then mirror
    for i in range(n):
        for j in range(i + 1, n):
            # Run A* from coords[i] to coords[j]
            result = a_star(grid, coords[i], coords[j], heuristic=heuristic)
            
            cost = result["cost"]
            expanded = result["expanded"]
            
            # Store cost in both positions (symmetric)
            matrix[i][j] = cost
            matrix[j][i] = cost
            
            expanded_total += expanded
            astar_calls += 1
    
    return {
        "matrix": matrix,
        "expanded_total": expanded_total,
        "astar_calls": astar_calls
    }


def pretty_print_matrix(names: List[str], matrix: List[List[float]]) -> None:
    """
    Print a readable table-like matrix with row/column labels.
    
    Args:
        names: List of stop names (labels)
        matrix: NxN cost matrix
    """
    n = len(matrix)
    
    if n == 0:
        print("Empty matrix")
        return
    
    # Truncate names to max 8 characters for readability
    max_name_len = 8
    truncated_names = [name[:max_name_len].ljust(max_name_len) for name in names]
    
    # Determine column width based on max value in matrix
    max_val = max(
        (val for row in matrix for val in row if val != math.inf),
        default=0
    )
    has_inf = any(val == math.inf for row in matrix for val in row)
    
    # Column width: at least 5, or enough for the largest number
    col_width = max(5, len(str(int(max_val))) + 1, 3 if has_inf else 0)
    
    # Print header row
    print(" " * (max_name_len + 2), end="")
    for name in truncated_names:
        print(name[:col_width].center(col_width), end=" ")
    print()
    
    # Print separator
    print(" " * (max_name_len + 2) + "-" * (n * (col_width + 1)))
    
    # Print matrix rows
    for i, row_name in enumerate(truncated_names):
        print(f"{row_name} |", end=" ")
        for j in range(n):
            val = matrix[i][j]
            if val == math.inf:
                cell = "inf"
            elif val == int(val):
                cell = str(int(val))
            else:
                cell = f"{val:.1f}"
            print(cell.rjust(col_width), end=" ")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("COST MATRIX GENERATION - STATIC DEMO")
    print("=" * 70)
    
    # Build static demo warehouse
    demo = make_static_demo()
    stops = build_stops(demo["start"], demo["parcels"])
    
    n_stops = len(stops["names"])
    print(f"\nNumber of stops: {n_stops}")
    print(f"Stop names: {stops['names']}")
    print()
    
    # Compute cost matrix with Manhattan heuristic
    print("-" * 70)
    print("MANHATTAN HEURISTIC")
    print("-" * 70)
    
    result_manhattan = compute_cost_matrix(
        demo["grid"],
        stops["coords"],
        heuristic="manhattan"
    )
    
    print(f"\nA* calls made: {result_manhattan['astar_calls']}")
    print(f"Total nodes expanded: {result_manhattan['expanded_total']}")
    print(f"Average expanded per call: {result_manhattan['expanded_total'] / result_manhattan['astar_calls']:.1f}")
    print()
    
    print("Cost matrix:")
    pretty_print_matrix(stops["names"], result_manhattan["matrix"])
    
    # Check for unreachable pairs
    has_unreachable = any(
        val == math.inf
        for i, row in enumerate(result_manhattan["matrix"])
        for j, val in enumerate(row)
        if i != j
    )
    
    if has_unreachable:
        print("\n⚠️  WARNING: Some stop pairs are unreachable (cost = inf)")
        for i in range(n_stops):
            for j in range(i + 1, n_stops):
                if result_manhattan["matrix"][i][j] == math.inf:
                    print(f"   {stops['names'][i]} -> {stops['names'][j]}: unreachable")
    else:
        print("\n✓ All stop pairs are reachable")
    
    # Compute cost matrix with Euclidean heuristic
    print("\n" + "-" * 70)
    print("EUCLIDEAN HEURISTIC")
    print("-" * 70)
    
    result_euclidean = compute_cost_matrix(
        demo["grid"],
        stops["coords"],
        heuristic="euclidean"
    )
    
    print(f"\nA* calls made: {result_euclidean['astar_calls']}")
    print(f"Total nodes expanded: {result_euclidean['expanded_total']}")
    print(f"Average expanded per call: {result_euclidean['expanded_total'] / result_euclidean['astar_calls']:.1f}")
    print()
    
    print("Cost matrix:")
    pretty_print_matrix(stops["names"], result_euclidean["matrix"])
    
    # Compare heuristics
    print("\n" + "=" * 70)
    print("HEURISTIC COMPARISON")
    print("=" * 70)
    
    print(f"\nManhattan - Total expanded: {result_manhattan['expanded_total']}")
    print(f"Euclidean - Total expanded: {result_euclidean['expanded_total']}")
    
    if result_euclidean['expanded_total'] < result_manhattan['expanded_total']:
        improvement = result_manhattan['expanded_total'] - result_euclidean['expanded_total']
        percent = (improvement / result_manhattan['expanded_total']) * 100
        print(f"Euclidean expanded {improvement} fewer nodes ({percent:.1f}% reduction)")
    elif result_manhattan['expanded_total'] < result_euclidean['expanded_total']:
        improvement = result_euclidean['expanded_total'] - result_manhattan['expanded_total']
        percent = (improvement / result_euclidean['expanded_total']) * 100
        print(f"Manhattan expanded {improvement} fewer nodes ({percent:.1f}% reduction)")
    else:
        print("Both heuristics expanded the same number of nodes")
    
    # Verify matrices are identical in costs (they should be, only expanded counts differ)
    costs_match = all(
        result_manhattan["matrix"][i][j] == result_euclidean["matrix"][i][j]
        for i in range(n_stops)
        for j in range(n_stops)
    )
    print(f"\nCost matrices identical: {costs_match} (expected: True)")
