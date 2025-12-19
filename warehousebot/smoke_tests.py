"""
Smoke tests for WarehouseBot core functionality.
Uses stdlib only - no external test frameworks.
"""

import math

from .astar import a_star
from .warehouse import make_static_demo, build_stops
from .cost_matrix import compute_cost_matrix
from .sequencing import greedy_route, hill_climb, simulated_annealing, is_route_feasible


def test_astar_edge_cases():
    """Test A* edge cases: start==goal and unreachable goal."""
    print("Running test_astar_edge_cases...", end=" ")
    
    # Create a simple grid
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    # Test 1: start == goal should return cost 0 and path length 1
    result = a_star(grid, (0, 0), (0, 0), heuristic="manhattan")
    assert result["found"], "A* should find path when start == goal"
    assert result["cost"] == 0, f"Expected cost 0 when start==goal, got {result['cost']}"
    assert len(result["path"]) == 1, f"Expected path length 1 when start==goal, got {len(result['path'])}"
    assert result["path"][0] == (0, 0), "Path should contain only the start/goal position"
    
    # Test 2: goal on obstacle should return found=False and cost=inf
    obstacle_pos = (1, 1)  # This is an obstacle (value 1)
    result = a_star(grid, (0, 0), obstacle_pos, heuristic="manhattan")
    assert not result["found"], "A* should not find path to obstacle"
    assert result["cost"] == math.inf, f"Expected cost inf for obstacle, got {result['cost']}"
    
    print("PASS")


def test_cost_matrix_symmetry():
    """Test cost matrix properties: diagonal zeros and symmetry."""
    print("Running test_cost_matrix_symmetry...", end=" ")
    
    # Build demo warehouse
    demo = make_static_demo()
    stops = build_stops(demo["start"], demo["parcels"])
    
    # Compute cost matrix
    result = compute_cost_matrix(demo["grid"], stops["coords"], heuristic="manhattan")
    matrix = result["matrix"]
    
    n = len(matrix)
    
    # Test 1: Diagonal should be all zeros
    for i in range(n):
        assert matrix[i][i] == 0, f"Diagonal element matrix[{i}][{i}] should be 0, got {matrix[i][i]}"
    
    # Test 2: Matrix should be symmetric
    for i in range(n):
        for j in range(n):
            assert matrix[i][j] == matrix[j][i], \
                f"Matrix not symmetric: matrix[{i}][{j}]={matrix[i][j]} != matrix[{j}][{i}]={matrix[j][i]}"
    
    print("PASS")


def test_sequencing_feasibility():
    """Test that sequencing algorithms produce feasible routes."""
    print("Running test_sequencing_feasibility...", end=" ")
    
    # Build demo warehouse
    demo = make_static_demo()
    stops = build_stops(demo["start"], demo["parcels"])
    
    # Compute cost matrix
    result = compute_cost_matrix(demo["grid"], stops["coords"], heuristic="manhattan")
    matrix = result["matrix"]
    precedence = stops["precedence"]
    
    # Test 1: Greedy route should be feasible
    greedy = greedy_route(matrix, precedence)
    assert is_route_feasible(greedy, precedence), \
        f"Greedy route is not feasible: {greedy}"
    
    # Test 2: Hill climb route should be feasible
    hc_result = hill_climb(
        matrix,
        precedence,
        max_iters=1000,
        neighbors_per_iter=30,
        seed=42,
        restarts=5
    )
    hill_route = hc_result["best_route"]
    assert is_route_feasible(hill_route, precedence), \
        f"Hill climbing route is not feasible: {hill_route}"
    
    # Test 3: Simulated annealing route should be feasible
    sa_result = simulated_annealing(
        matrix,
        precedence,
        max_iters=2000,
        t0=50.0,
        alpha=0.995,
        seed=42
    )
    sa_route = sa_result["best_route"]
    assert is_route_feasible(sa_route, precedence), \
        f"Simulated annealing route is not feasible: {sa_route}"
    
    print("PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("WAREHOUSEBOT SMOKE TESTS")
    print("=" * 60)
    print()
    
    try:
        test_astar_edge_cases()
        test_cost_matrix_symmetry()
        test_sequencing_feasibility()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        raise
    except Exception as e:
        print()
        print("=" * 60)
        print("UNEXPECTED ERROR ✗")
        print("=" * 60)
        print(f"Error: {e}")
        raise
