"""
Warehouse map definitions for WarehouseBot.
Defines grid layouts, start positions, and pickup/drop parcel pairs.
"""

import random
from collections import deque
from typing import List, Tuple, Dict, Set, Optional


def make_static_demo() -> Dict:
    """
    Create a static demo warehouse instance.
    
    Returns:
        dict with keys:
            - "grid": 2D grid (0=free, 1=obstacle)
            - "start": (r, c) starting position
            - "parcels": list of dicts with "pickup" and "drop" keys
    """
    # Small 8x8 grid with some shelf obstacles
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    start = (0, 0)
    
    parcels = [
        {"pickup": (3, 1), "drop": (7, 7)},
        {"pickup": (0, 4), "drop": (5, 5)},
        {"pickup": (6, 3), "drop": (2, 7)},
    ]
    
    return {
        "grid": grid,
        "start": start,
        "parcels": parcels
    }


def build_stops(start: Tuple[int, int], parcels: List[Dict]) -> Dict:
    """
    Build a standardized stop list for routing.
    
    Args:
        start: Starting position (r, c)
        parcels: List of parcel dicts with "pickup" and "drop" keys
    
    Returns:
        dict with keys:
            - "names": list of stop names ["START", "P0", "D0", "P1", "D1", ...]
            - "coords": list of coordinates aligned with names
            - "precedence": list of (pickup_idx, drop_idx) tuples indicating
                           each parcel's pickup must come before its drop
    """
    names = ["START"]
    coords = [start]
    precedence = []
    
    for i, parcel in enumerate(parcels):
        pickup_idx = len(names)
        names.append(f"P{i}")
        coords.append(parcel["pickup"])
        
        drop_idx = len(names)
        names.append(f"D{i}")
        coords.append(parcel["drop"])
        
        # Record that pickup must come before drop
        precedence.append((pickup_idx, drop_idx))
    
    return {
        "names": names,
        "coords": coords,
        "precedence": precedence
    }


def is_shelf_adjacent(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
    """
    Check if a cell is free and adjacent to at least one obstacle (shelf).
    
    Args:
        grid: 2D grid (0=free, 1=obstacle)
        cell: (r, c) cell to check
    
    Returns:
        True if cell is free and has at least one obstacle neighbor
    """
    if not grid or not grid[0]:
        return False
    
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    
    # Check if cell is in bounds and free
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    if grid[r][c] != 0:
        return False
    
    # Check if any of the 4 neighbors is an obstacle
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 1:  # Found adjacent obstacle
                return True
    
    return False


def reachable_from_start(grid: List[List[int]], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """
    Find all free cells reachable from start using 4-neighbor BFS.
    
    Args:
        grid: 2D grid (0=free, 1=obstacle)
        start: Starting position (r, c)
    
    Returns:
        Set of all reachable (r, c) coordinates
    """
    if not grid or not grid[0]:
        return set()
    
    rows, cols = len(grid), len(grid[0])
    
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols
    
    def is_free(r: int, c: int) -> bool:
        return in_bounds(r, c) and grid[r][c] == 0
    
    if not is_free(*start):
        return set()
    
    reachable = set()
    queue = deque([start])
    reachable.add(start)
    
    while queue:
        r, c = queue.popleft()
        
        # Explore 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            
            if is_free(nr, nc) and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    
    return reachable


def generate_random_warehouse(
    rows: int,
    cols: int,
    obstacle_prob: float,
    n_parcels: int,
    seed: Optional[int] = None,
    max_tries: int = 200
) -> Dict:
    """
    Generate a random feasible warehouse instance.
    
    Args:
        rows: Number of grid rows
        cols: Number of grid columns
        obstacle_prob: Probability of each cell being an obstacle (0.0 to 1.0)
        n_parcels: Number of parcels to generate
        seed: Random seed for reproducibility
        max_tries: Maximum number of retry attempts
    
    Returns:
        dict with keys "grid", "start", "parcels" (same format as make_static_demo)
    
    Raises:
        RuntimeError: If unable to generate feasible warehouse after max_tries attempts
    """
    rng = random.Random(seed)
    
    for attempt in range(max_tries):
        # Generate random grid with obstacles
        grid = [
            [1 if rng.random() < obstacle_prob else 0 for _ in range(cols)]
            for _ in range(rows)
        ]
        
        # Get all free cells
        free_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if grid[r][c] == 0
        ]
        
        # Need at least 1 (start) + 2*n_parcels (pickups + drops) free cells
        required_cells = 1 + 2 * n_parcels
        if len(free_cells) < required_cells:
            continue
        
        # Randomly select start (any free cell)
        start = rng.choice(free_cells)
        
        # Check reachability from start
        reachable = reachable_from_start(grid, start)
        
        # Filter to only reachable free cells for parcels
        reachable_free_cells = [cell for cell in free_cells if cell in reachable and cell != start]
        
        if len(reachable_free_cells) < 2 * n_parcels:
            continue
        
        # Prefer shelf-adjacent cells for parcel locations (more realistic)
        shelf_adjacent_cells = [cell for cell in reachable_free_cells if is_shelf_adjacent(grid, cell)]
        
        # Select parcel locations: prefer shelf-adjacent, fall back to any reachable if needed
        if len(shelf_adjacent_cells) >= 2 * n_parcels:
            # Enough shelf-adjacent cells available
            parcel_cells = rng.sample(shelf_adjacent_cells, 2 * n_parcels)
        else:
            # Not enough shelf-adjacent cells, use what we have plus random reachable cells
            parcel_cells = shelf_adjacent_cells[:]
            remaining_needed = 2 * n_parcels - len(shelf_adjacent_cells)
            non_shelf_cells = [cell for cell in reachable_free_cells if cell not in shelf_adjacent_cells]
            parcel_cells.extend(rng.sample(non_shelf_cells, remaining_needed))
        
        # Build parcels list
        parcels = []
        for i in range(n_parcels):
            pickup = parcel_cells[2 * i]
            drop = parcel_cells[2 * i + 1]
            parcels.append({"pickup": pickup, "drop": drop})
        
        return {
            "grid": grid,
            "start": start,
            "parcels": parcels
        }
    
    # Failed to generate feasible warehouse
    raise RuntimeError(
        f"Failed to generate feasible warehouse after {max_tries} attempts. "
        f"Try reducing obstacle_prob or grid size, or increasing max_tries."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("STATIC DEMO WAREHOUSE")
    print("=" * 60)
    
    demo = make_static_demo()
    
    print("\nGrid (0=free, 1=obstacle):")
    for row in demo["grid"]:
        print("  ", row)
    
    print(f"\nStart position: {demo['start']}")
    
    print(f"\nParcels ({len(demo['parcels'])} total):")
    for i, parcel in enumerate(demo["parcels"]):
        print(f"  Parcel {i}: pickup={parcel['pickup']}, drop={parcel['drop']}")
    
    print("\n" + "-" * 60)
    print("STOP LIST")
    print("-" * 60)
    
    stops = build_stops(demo["start"], demo["parcels"])
    
    print(f"\nStop names: {stops['names']}")
    print(f"\nStop coordinates:")
    for name, coord in zip(stops['names'], stops['coords']):
        print(f"  {name}: {coord}")
    
    print(f"\nPrecedence constraints (pickup must come before drop):")
    for pickup_idx, drop_idx in stops['precedence']:
        pickup_name = stops['names'][pickup_idx]
        drop_name = stops['names'][drop_idx]
        print(f"  {pickup_name} (index {pickup_idx}) -> {drop_name} (index {drop_idx})")
    
    print("\n" + "=" * 60)
    print("RANDOM WAREHOUSE GENERATION")
    print("=" * 60)
    
    rnd = generate_random_warehouse(
        rows=10,
        cols=10,
        obstacle_prob=0.2,
        n_parcels=3,
        seed=42
    )
    
    print("\nGrid (0=free, 1=obstacle):")
    for row in rnd["grid"]:
        print("  ", [str(x) for x in row])
    
    print(f"\nStart position: {rnd['start']}")
    
    print(f"\nParcels ({len(rnd['parcels'])} total):")
    for i, parcel in enumerate(rnd["parcels"]):
        print(f"  Parcel {i}: pickup={parcel['pickup']}, drop={parcel['drop']}")
    
    # Verify all locations are reachable
    reachable = reachable_from_start(rnd["grid"], rnd["start"])
    print(f"\nReachability check:")
    print(f"  Total reachable cells: {len(reachable)}")
    print(f"  Start reachable: {rnd['start'] in reachable}")
    all_parcels_reachable = all(
        p["pickup"] in reachable and p["drop"] in reachable
        for p in rnd["parcels"]
    )
    print(f"  All parcels reachable: {all_parcels_reachable}")
