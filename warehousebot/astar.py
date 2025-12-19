"""
A* pathfinding implementation for WarehouseBot.
Supports Manhattan and Euclidean heuristics on 2D grid with obstacles.
"""

import heapq
import math
from typing import Tuple, List, Dict, Callable


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Compute Euclidean distance between two grid cells."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def in_bounds(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
    """Check if cell is within grid bounds."""
    r, c = cell
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) if grid else False


def passable(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
    """Check if cell is passable (in bounds and not obstacle)."""
    return in_bounds(grid, cell) and grid[cell[0]][cell[1]] == 0


def neighbors(grid: List[List[int]], cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get valid 4-neighborhood neighbors (up, down, left, right)."""
    r, c = cell
    candidates = [
        (r - 1, c),  # up
        (r + 1, c),  # down
        (r, c - 1),  # left
        (r, c + 1),  # right
    ]
    return [n for n in candidates if passable(grid, n)]


def a_star(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: str = "manhattan"
) -> Dict:
    """
    A* pathfinding on a 2D grid.
    
    Args:
        grid: 2D grid where 0 = free, 1 = obstacle
        start: Starting cell (row, col)
        goal: Goal cell (row, col)
        heuristic: "manhattan" or "euclidean"
    
    Returns:
        dict with keys:
            - "path": list of (r, c) from start to goal (inclusive) or []
            - "cost": number of steps or math.inf if unreachable
            - "expanded": count of nodes popped from priority queue
            - "found": True if path found
            
    Edge cases:
        - If start == goal and valid: returns path [start], cost 0, expanded 0
        - If start/goal invalid (out of bounds or obstacle): returns found False, 
          cost math.inf, path [], expanded 0
    """
    # Validate heuristic
    if heuristic not in {"manhattan", "euclidean"}:
        raise ValueError(f"Invalid heuristic: {heuristic}. Must be 'manhattan' or 'euclidean'.")
    
    # Select heuristic function
    h_func: Callable = manhattan if heuristic == "manhattan" else euclidean
    
    # Check if start/goal are valid
    if not passable(grid, start) or not passable(grid, goal):
        return {
            "path": [],
            "cost": math.inf,
            "expanded": 0,
            "found": False
        }
    
    # Special case: start == goal
    # Documented behavior: expanded = 0 (no nodes popped from queue)
    if start == goal:
        return {
            "path": [start],
            "cost": 0,
            "expanded": 0,
            "found": True
        }
    
    # Initialize A* data structures
    counter = 0  # For tie-breaking to ensure deterministic behavior
    g_score = {start: 0}
    came_from = {}
    
    # Priority queue: (f, g, counter, cell)
    # Primary ordering: lowest f-score
    # Tie-break: lowest g-score (prefer nodes closer to start)
    # Secondary tie-break: counter for stability
    h_start = h_func(start, goal)
    open_set = [(h_start, 0, counter, start)]
    counter += 1
    
    expanded = 0
    
    while open_set:
        # Pop node with lowest f-score
        f, g, _, current = heapq.heappop(open_set)
        expanded += 1
        
        # Skip if we've already processed this node with a better path
        # (handles duplicate entries in priority queue)
        if g > g_score.get(current, math.inf):
            continue
        
        # Check if we reached the goal
        if current == goal:
            # Reconstruct path from goal back to start
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            
            return {
                "path": path,
                "cost": g_score[goal],
                "expanded": expanded,
                "found": True
            }
        
        # Explore neighbors
        for neighbor in neighbors(grid, current):
            tentative_g = g_score[current] + 1  # Each move costs 1
            
            if tentative_g < g_score.get(neighbor, math.inf):
                # This path to neighbor is better than any previous one
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h_func(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor))
                counter += 1
    
    # No path found
    return {
        "path": [],
        "cost": math.inf,
        "expanded": expanded,
        "found": False
    }


if __name__ == "__main__":
    # Define a small grid with obstacles
    # 0 = free, 1 = obstacle
    test_grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
    ]
    
    start = (0, 0)
    goal = (4, 4)
    
    print("Grid (0=free, 1=obstacle):")
    for row in test_grid:
        print(row)
    print()
    
    # Test with Manhattan heuristic
    print("A* with Manhattan heuristic:")
    result_manhattan = a_star(test_grid, start, goal, heuristic="manhattan")
    print(f"  Found: {result_manhattan['found']}")
    print(f"  Cost: {result_manhattan['cost']}")
    print(f"  Expanded: {result_manhattan['expanded']}")
    print(f"  Path: {result_manhattan['path']}")
    print()
    
    # Test with Euclidean heuristic
    print("A* with Euclidean heuristic:")
    result_euclidean = a_star(test_grid, start, goal, heuristic="euclidean")
    print(f"  Found: {result_euclidean['found']}")
    print(f"  Cost: {result_euclidean['cost']}")
    print(f"  Expanded: {result_euclidean['expanded']}")
    print(f"  Path: {result_euclidean['path']}")
    print()
    
    # Test edge case: unreachable goal (obstacle cell)
    unreachable_goal = (0, 3)  # This is an obstacle (value 1)
    print("A* with unreachable goal (0, 3) [obstacle cell]:")
    result_unreachable = a_star(test_grid, start, unreachable_goal, heuristic="manhattan")
    print(f"  Found: {result_unreachable['found']}")
    print(f"  Cost: {result_unreachable['cost']}")
    print(f"  Expanded: {result_unreachable['expanded']}")
    print(f"  Path: {result_unreachable['path']}")
    print()
    
    # Test edge case: start == goal
    print("A* with start == goal (2, 2):")
    result_same = a_star(test_grid, (2, 2), (2, 2), heuristic="manhattan")
    print(f"  Found: {result_same['found']}")
    print(f"  Cost: {result_same['cost']}")
    print(f"  Expanded: {result_same['expanded']}")
    print(f"  Path: {result_same['path']}")
