"""
Task sequencing for WarehouseBot.
Optimizes the order of pickups and drops while respecting precedence constraints.
"""

import math
import random
from typing import List, Tuple, Dict, Optional

from .warehouse import make_static_demo, build_stops
from .cost_matrix import compute_cost_matrix


def route_cost(route: List[int], cost_matrix: List[List[float]]) -> float:
    """
    Calculate total cost of a route.
    
    Args:
        route: List of stop indices
        cost_matrix: NxN cost matrix
    
    Returns:
        Total travel cost (math.inf if any segment is unreachable)
    """
    total = 0.0
    for i in range(len(route) - 1):
        segment_cost = cost_matrix[route[i]][route[i + 1]]
        if segment_cost == math.inf:
            return math.inf
        total += segment_cost
    return total


def is_route_feasible(route: List[int], precedence: List[Tuple[int, int]]) -> bool:
    """
    Check if route is feasible (respects precedence constraints).
    
    Args:
        route: List of stop indices
        precedence: List of (pickup_idx, drop_idx) pairs
    
    Returns:
        True if route contains all stops exactly once, starts with 0,
        and respects all precedence constraints
    """
    # Must contain all stops exactly once
    if len(route) != len(set(route)):
        return False
    
    # Must start with 0 (START)
    if not route or route[0] != 0:
        return False
    
    # Build position map: stop_idx -> position in route
    position = {stop: pos for pos, stop in enumerate(route)}
    
    # Check that all expected stops are present
    n_stops = len(route)
    if set(route) != set(range(n_stops)):
        return False
    
    # Check precedence constraints: pickup must come before drop
    for pickup_idx, drop_idx in precedence:
        if position[pickup_idx] >= position[drop_idx]:
            return False
    
    return True


def greedy_route(
    cost_matrix: List[List[float]],
    precedence: List[Tuple[int, int]]
) -> List[int]:
    """
    Build a feasible route using greedy nearest-neighbor heuristic.
    
    Args:
        cost_matrix: NxN cost matrix
        precedence: List of (pickup_idx, drop_idx) pairs
    
    Returns:
        Feasible route starting at 0
    """
    n_stops = len(cost_matrix)
    
    # Build dependency map: drop -> pickup
    drop_needs_pickup = {}
    for pickup_idx, drop_idx in precedence:
        drop_needs_pickup[drop_idx] = pickup_idx
    
    route = [0]  # Start at 0
    visited = {0}
    
    while len(route) < n_stops:
        current = route[-1]
        
        # Find all feasible next stops
        feasible = []
        for candidate in range(n_stops):
            if candidate in visited:
                continue
            
            # If it's a drop, check if pickup has been visited
            if candidate in drop_needs_pickup:
                required_pickup = drop_needs_pickup[candidate]
                if required_pickup not in visited:
                    continue
            
            feasible.append(candidate)
        
        if not feasible:
            # Fail loudly on impossible cases
            raise RuntimeError("Greedy failed: no feasible next stop (possible inf edge or precedence dead-end)")
        
        # Choose feasible stop with minimum cost
        # Tie-break by smallest index
        best_stop = min(feasible, key=lambda s: (cost_matrix[current][s], s))
        
        route.append(best_stop)
        visited.add(best_stop)
    
    return route


def random_feasible_route(
    n_stops: int,
    precedence: List[Tuple[int, int]],
    seed: Optional[int] = None
) -> List[int]:
    """
    Generate a random feasible route.
    
    Args:
        n_stops: Number of stops
        precedence: List of (pickup_idx, drop_idx) pairs
        seed: Random seed
    
    Returns:
        Random feasible route starting at 0
    """
    rng = random.Random(seed)
    
    # Start with base route [0, 1, 2, 3, ...]
    route = list(range(n_stops))
    
    # Perform many random swaps, only accepting feasible ones
    max_swaps = 1000
    for _ in range(max_swaps):
        # Pick two random positions (excluding position 0 which must be START)
        if n_stops <= 2:
            break
        
        i = rng.randint(1, n_stops - 1)
        j = rng.randint(1, n_stops - 1)
        
        if i == j:
            continue
        
        # Try swap
        route[i], route[j] = route[j], route[i]
        
        # Check if still feasible
        if not is_route_feasible(route, precedence):
            # Undo swap
            route[i], route[j] = route[j], route[i]
    
    return route


def _insertion_neighbor(route: List[int], rng: random.Random) -> List[int]:
    """
    Generate neighbor by removing an element and inserting it elsewhere.
    
    Args:
        route: Current route
        rng: Random number generator
    
    Returns:
        New route with one element moved
    """
    n = len(route)
    if n <= 2:
        return route[:]
    
    # Choose two different indices from [1, n-1]
    i = rng.randint(1, n - 1)
    j = rng.randint(1, n - 1)
    
    while i == j:
        j = rng.randint(1, n - 1)
    
    # Remove element at i and insert at j
    new_route = route[:]
    element = new_route.pop(i)
    new_route.insert(j, element)
    
    return new_route


def random_valid_neighbor(
    route: List[int],
    precedence: List[Tuple[int, int]],
    rng: random.Random
) -> List[int]:
    """
    Generate a valid neighboring route by swapping two positions.
    
    Args:
        route: Current route
        precedence: List of (pickup_idx, drop_idx) pairs
        rng: Random number generator
    
    Returns:
        New feasible route (or original if couldn't find valid move)
    """
    n = len(route)
    
    # Can only modify positions 1 to n-1 (position 0 is always START)
    if n <= 2:
        return route[:]
    
    # Phase 1: Try swap-based neighbors (up to 80 attempts)
    max_swap_attempts = 80
    for _ in range(max_swap_attempts):
        # Pick two different positions to swap (excluding position 0)
        i = rng.randint(1, n - 1)
        j = rng.randint(1, n - 1)
        
        if i == j:
            continue
        
        # Create new route with swap
        new_route = route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        
        # Check if feasible
        if is_route_feasible(new_route, precedence):
            return new_route
    
    # Phase 2: Try insertion-based neighbors (up to 80 attempts)
    max_insertion_attempts = 80
    for _ in range(max_insertion_attempts):
        new_route = _insertion_neighbor(route, rng)
        
        # Check if feasible
        if is_route_feasible(new_route, precedence):
            return new_route
    
    # Couldn't find valid move, return original
    return route[:]


def hill_climb(
    cost_matrix: List[List[float]],
    precedence: List[Tuple[int, int]],
    start_route: Optional[List[int]] = None,
    max_iters: int = 2000,
    neighbors_per_iter: int = 50,
    seed: Optional[int] = None,
    restarts: int = 10,
    restart_neighbors_per_iter: Optional[int] = None
) -> Dict:
    """
    Hill climbing optimization for route sequencing with random restarts.
    
    Args:
        cost_matrix: NxN cost matrix
        precedence: List of (pickup_idx, drop_idx) pairs
        start_route: Initial route (uses greedy if None)
        max_iters: Maximum iterations per restart
        neighbors_per_iter: Number of neighbors to sample per iteration
        seed: Random seed
        restarts: Number of times to restart with different initial routes
        restart_neighbors_per_iter: Neighbors per iter for restarts (uses neighbors_per_iter if None)
    
    Returns:
        dict with keys: "best_route", "best_cost", "start_cost", "iters", "restarts", "best_restart"
    """
    rng = random.Random(seed)
    n_stops = len(cost_matrix)
    
    if restart_neighbors_per_iter is None:
        restart_neighbors_per_iter = neighbors_per_iter
    
    overall_best_route = None
    overall_best_cost = math.inf
    overall_start_cost = None
    best_restart_idx = 0
    iters_per_restart = []  # Track iterations for each restart
    
    for restart_idx in range(restarts):
        # First restart uses provided start_route or greedy
        # Subsequent restarts use random feasible routes
        if restart_idx == 0:
            if start_route is None:
                current_route = greedy_route(cost_matrix, precedence)
            else:
                current_route = start_route[:]
        else:
            # Generate random feasible route for restart
            current_route = random_feasible_route(n_stops, precedence, seed=rng.randint(0, 1000000))
        
        current_cost = route_cost(current_route, cost_matrix)
        
        # Track start cost from first restart only
        if restart_idx == 0:
            overall_start_cost = current_cost
        
        best_route = current_route[:]
        best_cost = current_cost
        
        # Hill climbing loop
        actual_iters = 0  # Track actual iterations for this restart
        for iteration in range(max_iters):
            actual_iters = iteration + 1  # Track iteration count
            
            # Sample neighbors
            improved = False
            best_neighbor_route = None
            best_neighbor_cost = current_cost
            
            neighbors_to_sample = restart_neighbors_per_iter if restart_idx > 0 else neighbors_per_iter
            
            for _ in range(neighbors_to_sample):
                neighbor = random_valid_neighbor(current_route, precedence, rng)
                neighbor_cost = route_cost(neighbor, cost_matrix)
                
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor_route = neighbor
                    best_neighbor_cost = neighbor_cost
                    improved = True
            
            # If we found an improvement, move to it
            if improved:
                current_route = best_neighbor_route
                current_cost = best_neighbor_cost
                
                # Update best if needed
                if current_cost < best_cost:
                    best_route = current_route[:]
                    best_cost = current_cost
            else:
                # No improvement found, stop this restart
                break
        
        iters_per_restart.append(actual_iters)
        
        # Update overall best
        if best_cost < overall_best_cost:
            overall_best_route = best_route[:]
            overall_best_cost = best_cost
            best_restart_idx = restart_idx
    
    return {
        "best_route": overall_best_route,
        "best_cost": overall_best_cost,
        "start_cost": overall_start_cost,
        "iters": iters_per_restart[best_restart_idx],  # Actual iters for best restart
        "iters_per_restart": iters_per_restart,  # All iterations per restart
        "restarts": restarts,
        "best_restart": best_restart_idx
    }


def simulated_annealing(
    cost_matrix: List[List[float]],
    precedence: List[Tuple[int, int]],
    start_route: Optional[List[int]] = None,
    max_iters: int = 5000,
    t0: float = 50.0,
    alpha: float = 0.995,
    seed: Optional[int] = None
) -> Dict:
    """
    Simulated annealing optimization for route sequencing.
    
    Args:
        cost_matrix: NxN cost matrix
        precedence: List of (pickup_idx, drop_idx) pairs
        start_route: Initial route (uses greedy if None)
        max_iters: Maximum iterations
        t0: Initial temperature
        alpha: Cooling rate (T_new = alpha * T_old)
        seed: Random seed
    
    Returns:
        dict with keys: "best_route", "best_cost", "start_cost", "iters", "accepted"
    """
    rng = random.Random(seed)
    
    # Initialize with greedy route if not provided
    if start_route is None:
        current_route = greedy_route(cost_matrix, precedence)
    else:
        current_route = start_route[:]
    
    current_cost = route_cost(current_route, cost_matrix)
    start_cost = current_cost
    
    best_route = current_route[:]
    best_cost = current_cost
    
    accepted = 0
    
    for iteration in range(max_iters):
        # Temperature schedule
        temperature = t0 * (alpha ** iteration)
        
        # Generate neighbor
        neighbor = random_valid_neighbor(current_route, precedence, rng)
        neighbor_cost = route_cost(neighbor, cost_matrix)
        
        # Calculate cost difference
        delta = neighbor_cost - current_cost
        
        # Acceptance criterion
        if delta < 0:
            # Better solution, always accept
            accept = True
        else:
            # Worse solution, accept with probability
            if temperature > 1e-10:
                acceptance_prob = math.exp(-delta / temperature)
                accept = rng.random() < acceptance_prob
            else:
                accept = False
        
        if accept:
            current_route = neighbor
            current_cost = neighbor_cost
            accepted += 1
            
            # Update best if needed
            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost
    
    return {
        "best_route": best_route,
        "best_cost": best_cost,
        "start_cost": start_cost,
        "iters": max_iters,
        "accepted": accepted
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TASK SEQUENCING - STATIC DEMO")
    print("=" * 70)
    
    # Build warehouse and compute cost matrix
    demo = make_static_demo()
    stops = build_stops(demo["start"], demo["parcels"])
    
    print(f"\nNumber of stops: {len(stops['names'])}")
    print(f"Stop names: {stops['names']}")
    print(f"Precedence constraints: {len(stops['precedence'])}")
    for pickup_idx, drop_idx in stops['precedence']:
        print(f"  {stops['names'][pickup_idx]} must come before {stops['names'][drop_idx]}")
    
    print("\nComputing cost matrix...")
    cost_result = compute_cost_matrix(
        demo["grid"],
        stops["coords"],
        heuristic="manhattan"
    )
    cost_matrix = cost_result["matrix"]
    
    print(f"Cost matrix computed: {len(cost_matrix)}x{len(cost_matrix)} matrix")
    
    # Greedy route
    print("\n" + "-" * 70)
    print("GREEDY NEAREST-NEIGHBOR ROUTE")
    print("-" * 70)
    
    greedy = greedy_route(cost_matrix, stops["precedence"])
    greedy_cost = route_cost(greedy, cost_matrix)
    greedy_feasible = is_route_feasible(greedy, stops["precedence"])
    
    print(f"\nRoute indices: {greedy}")
    print(f"Route names: {[stops['names'][i] for i in greedy]}")
    print(f"Total cost: {greedy_cost}")
    print(f"Feasible: {greedy_feasible}")
    
    # Hill climbing with restarts
    print("\n" + "-" * 70)
    print("HILL CLIMBING OPTIMIZATION (WITH RESTARTS)")
    print("-" * 70)
    
    hc_result = hill_climb(
        cost_matrix,
        stops["precedence"],
        max_iters=2000,
        neighbors_per_iter=50,
        seed=42,
        restarts=10
    )
    
    hc_route = hc_result["best_route"]
    hc_cost = hc_result["best_cost"]
    hc_feasible = is_route_feasible(hc_route, stops["precedence"])
    
    print(f"\nRestarts: {hc_result['restarts']}")
    print(f"Best found at restart: {hc_result['best_restart']}")
    print(f"Start cost (greedy): {hc_result['start_cost']}")
    print(f"Best cost: {hc_cost}")
    
    if greedy_cost > 0:
        improvement = ((greedy_cost - hc_cost) / greedy_cost) * 100
        print(f"Improvement over greedy: {improvement:.2f}%")
    
    print(f"\nRoute indices: {hc_route}")
    print(f"Route names: {[stops['names'][i] for i in hc_route]}")
    print(f"Feasible: {hc_feasible}")
    
    # Simulated annealing
    print("\n" + "-" * 70)
    print("SIMULATED ANNEALING OPTIMIZATION")
    print("-" * 70)
    
    sa_result = simulated_annealing(
        cost_matrix,
        stops["precedence"],
        max_iters=5000,
        t0=50.0,
        alpha=0.995,
        seed=42
    )
    
    sa_route = sa_result["best_route"]
    sa_cost = sa_result["best_cost"]
    sa_feasible = is_route_feasible(sa_route, stops["precedence"])
    
    print(f"\nIterations: {sa_result['iters']}")
    print(f"Accepted moves: {sa_result['accepted']}")
    print(f"Start cost: {sa_result['start_cost']}")
    print(f"Best cost: {sa_cost}")
    
    if greedy_cost > 0:
        improvement = ((greedy_cost - sa_cost) / greedy_cost) * 100
        print(f"Improvement over greedy: {improvement:.2f}%")
    
    print(f"\nRoute indices: {sa_route}")
    print(f"Route names: {[stops['names'][i] for i in sa_route]}")
    print(f"Feasible: {sa_feasible}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"\nGreedy:              Cost = {greedy_cost:.1f}")
    print(f"Hill Climbing:       Cost = {hc_cost:.1f} ({((greedy_cost - hc_cost) / greedy_cost * 100):.2f}% better)")
    print(f"Simulated Annealing: Cost = {sa_cost:.1f} ({((greedy_cost - sa_cost) / greedy_cost * 100):.2f}% better)")
    
    print(f"\nAll routes feasible: {greedy_feasible and hc_feasible and sa_feasible}")
