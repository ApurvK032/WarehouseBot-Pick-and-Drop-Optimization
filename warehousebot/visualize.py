"""
Visualization for WarehouseBot using matplotlib.
Displays warehouse grids, routes, and performance comparisons.
"""

import argparse
import math
import sys
import textwrap
import time
from typing import List, Tuple, Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .astar import a_star
from .warehouse import make_static_demo, build_stops, generate_random_warehouse
from .cost_matrix import compute_cost_matrix
from .sequencing import greedy_route, hill_climb, simulated_annealing, route_cost


def plot_grid(ax, grid: List[List[int]]) -> None:
    """
    Plot the warehouse grid with obstacles and visible gridlines.
    
    Args:
        ax: Matplotlib axes object
        grid: 2D grid (0=free, 1=obstacle)
    """
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    
    # Use proper extent so integer coordinates are at cell centers
    # extent = [left, right, bottom, top] in data coordinates
    ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1, origin='upper',
              interpolation='none', extent=[-0.5, cols-0.5, rows-0.5, -0.5])
    ax.set_aspect('equal')
    
    # Add visible gridlines for clarity
    # Set minor ticks for gridlines at cell boundaries
    ax.set_xticks([i - 0.5 for i in range(cols + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(rows + 1)], minor=True)
    
    # Enable grid with thin lines
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    
    # Remove major ticks but keep labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')


def route_to_cell_path(
    grid: List[List[int]],
    coords: List[Tuple[int, int]],
    route: List[int],
    heuristic: str = "manhattan"
) -> Dict:
    """
    Convert a route (sequence of stop indices) to a continuous cell path.
    
    Args:
        grid: 2D warehouse grid
        coords: List of stop coordinates
        route: List of stop indices
        heuristic: "manhattan" or "euclidean"
    
    Returns:
        dict with keys:
            - "cell_path": continuous list of (r, c) cells
            - "cost": total path cost
            - "expanded_total": total nodes expanded
            - "found": True if all segments found
    """
    cell_path = []
    total_cost = 0.0
    expanded_total = 0
    all_found = True
    
    for i in range(len(route) - 1):
        start_idx = route[i]
        goal_idx = route[i + 1]
        
        start_coord = coords[start_idx]
        goal_coord = coords[goal_idx]
        
        # Run A* for this segment
        result = a_star(grid, start_coord, goal_coord, heuristic=heuristic)
        
        if not result["found"]:
            all_found = False
            continue
        
        segment_path = result["path"]
        total_cost += result["cost"]
        expanded_total += result["expanded"]
        
        # Append to cell path, avoiding duplicate connecting cells
        if i == 0:
            # First segment, add all cells
            cell_path.extend(segment_path)
        else:
            # Skip first cell of segment (it's the last cell of previous segment)
            cell_path.extend(segment_path[1:])
    
    return {
        "cell_path": cell_path,
        "cost": total_cost,
        "expanded_total": expanded_total,
        "found": all_found
    }


def plot_route_overlay(
    ax,
    cell_path: List[Tuple[int, int]],
    color: str = "red"
) -> None:
    """
    Overlay a route path on the grid with directional arrows.
    
    Args:
        ax: Matplotlib axes object
        cell_path: List of (r, c) cells forming the path
        color: Line color
    """
    if not cell_path:
        return
    
    # Convert (r, c) to plot coordinates (x, y)
    # With proper extent, integers are at cell centers: x=c, y=r
    x_coords = [c for r, c in cell_path]
    y_coords = [r for r, c in cell_path]
    
    # Plot path as line
    ax.plot(x_coords, y_coords, color=color, linewidth=2, marker='o',
            markersize=3, alpha=0.7, label='Route')
    
    # Add directional arrows along the path (every 8th segment to reduce clutter)
    arrow_interval = 8
    for i in range(0, len(cell_path) - 1, arrow_interval):
        # Get direction vector
        dx = x_coords[i + 1] - x_coords[i]
        dy = y_coords[i + 1] - y_coords[i]
        
        # Normalize and scale for arrow
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            # Position arrow slightly along the segment
            mid_x = x_coords[i] + 0.6 * dx
            mid_y = y_coords[i] + 0.6 * dy
            
            # Scale arrow to be visible but not too large
            scale = 0.3
            ax.arrow(mid_x, mid_y, dx * scale, dy * scale,
                    head_width=0.3, head_length=0.2, fc=color, ec=color, alpha=0.8)
    
    # Mark start with distinct marker
    if cell_path:
        ax.plot(x_coords[0], y_coords[0], color='green', marker='o',
                markersize=12, markeredgecolor='darkgreen', markeredgewidth=2.5,
                label='Start', zorder=10)
        
        # Mark end with distinct marker
        ax.plot(x_coords[-1], y_coords[-1], color='blue', marker='s',
                markersize=12, markeredgecolor='darkblue', markeredgewidth=2.5,
                label='End', zorder=10)


def plot_stops(
    ax,
    grid: List[List[int]],
    names: List[str],
    coords: List[Tuple[int, int]],
    route: Optional[List[int]] = None
) -> None:
    """
    Annotate stops on the grid with step order numbers.
    
    Args:
        ax: Matplotlib axes object
        grid: 2D grid for safety checks
        names: List of stop names
        coords: List of stop coordinates
        route: Optional route to show visit order
    """
    # Safety check: ensure all stops are on free cells
    for i, (name, (r, c)) in enumerate(zip(names, coords)):
        if grid[r][c] != 0:
            raise ValueError(f"Stop {name} at ({r}, {c}) is on an obstacle (grid[{r}][{c}] = {grid[r][c]})")
    
    # If route provided, show step order
    if route:
        # Build visit order map: stop_idx -> step number
        visit_order = {stop_idx: step for step, stop_idx in enumerate(route)}
        
        for stop_idx, (name, (r, c)) in enumerate(zip(names, coords)):
            if stop_idx in visit_order:
                step = visit_order[stop_idx]
                if step == 0:
                    # START - no number
                    label = name
                else:
                    # Other stops - show step number
                    label = f"{step}: {name}"
            else:
                label = name
            
            # Place text with slight offset for better readability
            # With proper extent, x=c, y=r (no +0.5 shift needed)
            ax.text(c + 0.7, r + 0.3, label, fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black',
                              linewidth=0.5, alpha=0.9),
                    weight='bold', zorder=5)
    else:
        # No route - just show names
        for name, (r, c) in zip(names, coords):
            ax.text(c + 0.7, r + 0.3, name, fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black',
                              linewidth=0.5, alpha=0.9),
                    weight='bold', zorder=5)


def route_has_interleaved_drops(route: List[int], names: List[str]) -> bool:
    """
    Check if route has interleaved drops (drop before last pickup).
    
    Args:
        route: List of stop indices in visit order
        names: List of stop names
    
    Returns:
        True if there exists a drop that occurs before the last pickup
    """
    # Find positions of pickups and drops
    pickup_positions = []
    drop_positions = []
    
    for pos, stop_idx in enumerate(route):
        name = names[stop_idx]
        if name.startswith('P') and name != 'START':
            pickup_positions.append(pos)
        elif name.startswith('D'):
            drop_positions.append(pos)
    
    # No pickups or drops means no interleaving
    if not pickup_positions or not drop_positions:
        return False
    
    # Find last pickup position and first drop position
    last_pickup_pos = max(pickup_positions)
    first_drop_pos = min(drop_positions)
    
    # Interleaving occurs if first drop comes before last pickup
    return first_drop_pos < last_pickup_pos


def compute_routes_for_instance(
    grid: List[List[int]],
    stops: Dict,
    heuristic: str = "manhattan",
    seed: int = 0
) -> Dict:
    """
    Compute routes for all algorithms on a single instance.
    
    Args:
        grid: Warehouse grid
        stops: Stops dict from build_stops
        heuristic: Heuristic to use
        seed: Random seed
    
    Returns:
        Dict with routes, costs, and best algorithm choice
    """
    # Compute cost matrix once
    cost_result = compute_cost_matrix(grid, stops["coords"], heuristic=heuristic)
    cost_matrix = cost_result["matrix"]
    
    # Greedy
    greedy = greedy_route(cost_matrix, stops["precedence"])
    greedy_cost = route_cost(greedy, cost_matrix)
    
    # Hill climbing
    hc_result = hill_climb(cost_matrix, stops["precedence"], 
                           max_iters=2000, neighbors_per_iter=50, 
                           seed=seed, restarts=10)
    hc_route = hc_result["best_route"]
    hc_cost = hc_result["best_cost"]
    
    # Simulated annealing
    sa_result = simulated_annealing(cost_matrix, stops["precedence"],
                                    max_iters=5000, t0=50.0, alpha=0.995, 
                                    seed=seed)
    sa_route = sa_result["best_route"]
    sa_cost = sa_result["best_cost"]
    
    # Determine best (prefer hill on tie)
    best_algo = "greedy"
    best_route = greedy
    best_cost = greedy_cost
    
    if hc_cost < best_cost:
        best_algo = "hill"
        best_route = hc_route
        best_cost = hc_cost
    
    if sa_cost < best_cost:
        best_algo = "sa"
        best_route = sa_route
        best_cost = sa_cost
    elif sa_cost == best_cost and best_algo == "greedy":
        # Prefer hill over sa on tie
        if hc_cost == sa_cost:
            best_algo = "hill"
            best_route = hc_route
            best_cost = hc_cost
    
    return {
        "greedy": {"route": greedy, "cost": greedy_cost},
        "hill": {"route": hc_route, "cost": hc_cost},
        "sa": {"route": sa_route, "cost": sa_cost},
        "best_algo": best_algo,
        "best_route": best_route,
        "best_cost": best_cost,
    }


def plot_random_instance(
    rows: int,
    cols: int,
    obstacle_prob: float,
    n_parcels: int,
    seed: int,
    heuristic: str = "manhattan",
    algo: str = "best",
    save_path: Optional[str] = None
) -> Dict:
    """
    Visualize a random warehouse instance.
    
    Args:
        rows: Grid rows
        cols: Grid columns
        obstacle_prob: Obstacle probability
        n_parcels: Number of parcels
        seed: Random seed
        heuristic: Heuristic to use
        algo: Algorithm to use ("greedy", "hill", "sa", or "best")
        save_path: Path to save figure (or None to show)
    
    Returns:
        Dict summary with seed, costs, algo, interleaving status, route names
    """
    print(f"Generating random instance: {rows}x{cols}, {n_parcels} parcels, seed={seed}")
    
    # Generate warehouse
    warehouse = generate_random_warehouse(rows, cols, obstacle_prob, n_parcels, seed=seed)
    grid = warehouse["grid"]
    start = warehouse["start"]
    parcels = warehouse["parcels"]
    
    # Build stops
    stops = build_stops(start, parcels)
    
    # Compute routes
    routes_result = compute_routes_for_instance(grid, stops, heuristic=heuristic, seed=seed)
    
    # Select route based on algo parameter
    if algo == "best":
        chosen_route = routes_result["best_route"]
        chosen_cost = routes_result["best_cost"]
        chosen_algo = routes_result["best_algo"]
    elif algo == "greedy":
        chosen_route = routes_result["greedy"]["route"]
        chosen_cost = routes_result["greedy"]["cost"]
        chosen_algo = "greedy"
    elif algo == "hill":
        chosen_route = routes_result["hill"]["route"]
        chosen_cost = routes_result["hill"]["cost"]
        chosen_algo = "hill"
    elif algo == "sa":
        chosen_route = routes_result["sa"]["route"]
        chosen_cost = routes_result["sa"]["cost"]
        chosen_algo = "sa"
    else:
        raise ValueError(f"Unknown algo: {algo}")
    
    # Check for interleaving
    has_interleaving = route_has_interleaved_drops(chosen_route, stops["names"])
    
    # Get route names
    route_names = [stops["names"][i] for i in chosen_route]
    
    # Convert route to cell path
    path_result = route_to_cell_path(grid, stops["coords"], chosen_route, heuristic=heuristic)
    
    # Create figure with improved layout using GridSpec
    # Outer: 1 row x 2 columns (left: grid, right: panel)
    # Right panel: 2 rows with height ratio [3, 1] for 75/25 split
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1])
    
    # Left column: Grid with route
    ax_grid = fig.add_subplot(gs[0, 0])
    plot_grid(ax_grid, grid)
    plot_route_overlay(ax_grid, path_result["cell_path"], color='red')
    plot_stops(ax_grid, grid, stops["names"], stops["coords"], route=chosen_route)
    
    interleave_str = "INTERLEAVED" if has_interleaving else "Sequential"
    ax_grid.set_title(f'Random Warehouse ({rows}x{cols}, seed={seed})\n'
                      f'Algorithm: {chosen_algo.upper()}, Cost: {chosen_cost:.1f}, {interleave_str}',
                      fontsize=11, weight='bold')
    ax_grid.legend(loc='upper right', fontsize=8)
    
    # Right panel: Create subgridspec with 2 rows [3, 1] for 75/25 split
    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Top-right: Cost comparison (75% of right panel)
    ax_cost = fig.add_subplot(gs_right[0, 0])
    algorithms = ['Greedy', 'Hill\nClimb', 'SA']
    costs = [routes_result["greedy"]["cost"], 
             routes_result["hill"]["cost"], 
             routes_result["sa"]["cost"]]
    
    bars_cost = ax_cost.bar(algorithms, costs, 
                             color=['steelblue', 'coral', 'lightgreen'],
                             alpha=0.8, edgecolor='black', linewidth=1.2)
    ax_cost.set_ylabel('Path Cost', fontsize=10, weight='bold')
    ax_cost.set_title('Cost Comparison', fontsize=10, weight='bold')
    ax_cost.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set tight y-limits so bars fill the panel nicely
    max_cost = max(costs)
    ax_cost.set_ylim(0, max_cost * 1.15)
    
    # Highlight chosen algorithm
    highlight_idx = {"greedy": 0, "hill": 1, "sa": 2}.get(chosen_algo, -1)
    if highlight_idx >= 0:
        bars_cost[highlight_idx].set_edgecolor('red')
        bars_cost[highlight_idx].set_linewidth(2.5)
    
    # Add value labels
    for bar in bars_cost:
        height = bar.get_height()
        ax_cost.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Bottom-right: Route info box (25% of right panel)
    ax_info = fig.add_subplot(gs_right[1, 0])
    ax_info.axis('off')
    
    # Wrap route sequence text
    route_str = " → ".join(route_names)
    wrapped_route = textwrap.fill(route_str, width=60)
    
    info_text = f"Route Sequence:\n{wrapped_route}\n\n"
    info_text += f"Stops: {len(chosen_route)}\n"
    info_text += f"Parcels: {n_parcels}\n"
    info_text += f"Algorithm: {chosen_algo.upper()}\n"
    info_text += f"Interleaved: {'YES' if has_interleaving else 'NO'}\n"
    info_text += f"Cost: {chosen_cost:.1f}"
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'WarehouseBot Random Instance', fontsize=13, weight='bold')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        print("Displaying visualization...")
        plt.show()
    
    plt.close()
    
    return {
        "seed": seed,
        "rows": rows,
        "cols": cols,
        "n_parcels": n_parcels,
        "algo": chosen_algo,
        "cost": chosen_cost,
        "has_interleaving": has_interleaving,
        "route_names": route_names,
    }


def find_and_save_interleaving_example(
    rows: int,
    cols: int,
    obstacle_prob: float,
    n_parcels: int,
    start_seed: int,
    max_tries: int,
    heuristic: str = "manhattan",
    algo: str = "best",
    save_path: str = "interleaving.png"
) -> Dict:
    """
    Search for an instance with interleaved drops and save visualization.
    
    Args:
        rows: Grid rows
        cols: Grid columns
        obstacle_prob: Obstacle probability
        n_parcels: Number of parcels
        start_seed: Starting seed
        max_tries: Maximum number of seeds to try
        heuristic: Heuristic to use
        algo: Algorithm to use
        save_path: Where to save the figure
    
    Returns:
        Summary dict of found/last instance
    """
    print(f"\nSearching for interleaving example...")
    print(f"Config: {rows}x{cols}, {n_parcels} parcels, algo={algo}")
    print(f"Trying seeds {start_seed} to {start_seed + max_tries - 1}...\n")
    
    last_result = None
    
    for i in range(max_tries):
        seed = start_seed + i
        
        try:
            # Generate instance
            warehouse = generate_random_warehouse(rows, cols, obstacle_prob, n_parcels, seed=seed)
            grid = warehouse["grid"]
            start = warehouse["start"]
            parcels = warehouse["parcels"]
            
            # Build stops
            stops = build_stops(start, parcels)
            
            # Compute routes
            routes_result = compute_routes_for_instance(grid, stops, heuristic=heuristic, seed=seed)
            
            # Select route
            if algo == "best":
                chosen_route = routes_result["best_route"]
            elif algo == "greedy":
                chosen_route = routes_result["greedy"]["route"]
            elif algo == "hill":
                chosen_route = routes_result["hill"]["route"]
            elif algo == "sa":
                chosen_route = routes_result["sa"]["route"]
            else:
                chosen_route = routes_result["best_route"]
            
            # Check for interleaving
            has_interleaving = route_has_interleaved_drops(chosen_route, stops["names"])
            
            if has_interleaving:
                # Found one! Visualize it
                print(f"✓ Found interleaving at seed {seed}!")
                route_names = [stops["names"][idx] for idx in chosen_route]
                print(f"  Route: {' → '.join(route_names)}")
                
                result = plot_random_instance(rows, cols, obstacle_prob, n_parcels, 
                                             seed, heuristic=heuristic, algo=algo, 
                                             save_path=save_path)
                
                print(f"\nSuccess! Saved interleaving example to {save_path}")
                return result
            
            # Save as last result in case we don't find any
            if i == max_tries - 1:
                last_result = (seed, routes_result, stops)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Tried {i + 1}/{max_tries} seeds...")
        
        except RuntimeError:
            # Failed to generate warehouse, skip
            continue
    
    # No interleaving found
    print(f"\n⚠️  No interleaving found in {max_tries} attempts.")
    print(f"Saving last attempted instance as fallback...")
    
    if last_result:
        seed, routes_result, stops = last_result
        result = plot_random_instance(rows, cols, obstacle_prob, n_parcels, 
                                     seed, heuristic=heuristic, algo=algo, 
                                     save_path=save_path)
        print(f"Saved fallback to {save_path}")
        return result
    else:
        print("ERROR: Could not generate any valid instances")
        return {}


def plot_static_comparison(save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive visualization comparing all algorithms on static demo.
    
    Args:
        save_path: If provided, save figure to this path instead of showing
    """
    print("Generating static demo comparison visualization...")
    
    # Load demo warehouse
    demo = make_static_demo()
    grid = demo["grid"]
    start = demo["start"]
    parcels = demo["parcels"]
    
    # Build stops
    stops = build_stops(start, parcels)
    
    # Compute cost matrix (once)
    print("Computing cost matrix...")
    t0 = time.perf_counter()
    cost_result = compute_cost_matrix(grid, stops["coords"], heuristic="manhattan")
    cost_matrix_time = time.perf_counter() - t0
    cost_matrix = cost_result["matrix"]
    
    # Run greedy
    print("Running greedy...")
    t0 = time.perf_counter()
    greedy = greedy_route(cost_matrix, stops["precedence"])
    greedy_time = time.perf_counter() - t0
    greedy_cost = route_cost(greedy, cost_matrix)
    
    # Run hill climbing
    print("Running hill climbing...")
    t0 = time.perf_counter()
    hc_result = hill_climb(cost_matrix, stops["precedence"], max_iters=2000,
                           neighbors_per_iter=50, seed=42, restarts=10)
    hc_time = time.perf_counter() - t0
    hc_route = hc_result["best_route"]
    hc_cost = hc_result["best_cost"]
    
    # Run simulated annealing
    print("Running simulated annealing...")
    t0 = time.perf_counter()
    sa_result = simulated_annealing(cost_matrix, stops["precedence"],
                                    max_iters=5000, t0=50.0, alpha=0.995, seed=42)
    sa_time = time.perf_counter() - t0
    sa_route = sa_result["best_route"]
    sa_cost = sa_result["best_cost"]
    
    # Choose best route to display (prefer hill climb on tie)
    if hc_cost <= sa_cost:
        best_route = hc_route
        best_cost = hc_cost
        best_name = "Hill Climbing"
    else:
        best_route = sa_route
        best_cost = sa_cost
        best_name = "Simulated Annealing"
    
    print(f"Best route: {best_name} (cost = {best_cost})")
    
    # Convert best route to cell path
    print("Computing cell path for visualization...")
    path_result = route_to_cell_path(grid, stops["coords"], best_route, heuristic="manhattan")
    
    # Create figure with improved layout using GridSpec
    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[1, 1, 1])
    
    # Left column: Grid with route (spans all 3 rows)
    ax_grid = fig.add_subplot(gs[:, 0])
    plot_grid(ax_grid, grid)
    plot_route_overlay(ax_grid, path_result["cell_path"], color='red')
    plot_stops(ax_grid, grid, stops["names"], stops["coords"], route=best_route)
    ax_grid.set_title(f'Warehouse Grid - Best: {best_name}\nCost: {best_cost:.1f}', 
                      fontsize=11, weight='bold')
    ax_grid.legend(loc='upper right', fontsize=8)
    
    # Right column, row 1: Cost comparison
    ax_cost = fig.add_subplot(gs[0, 1])
    algorithms = ['Greedy', 'Hill\nClimb', 'Simulated\nAnneal']
    costs = [greedy_cost, hc_cost, sa_cost]
    
    bars_cost = ax_cost.bar(algorithms, costs, color=['steelblue', 'coral', 'lightgreen'],
                             alpha=0.8, edgecolor='black', linewidth=1.2)
    ax_cost.set_ylabel('Path Cost', fontsize=10, weight='bold')
    ax_cost.set_title('Cost Comparison', fontsize=10, weight='bold')
    ax_cost.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on cost bars
    for bar in bars_cost:
        height = bar.get_height()
        ax_cost.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Right column, row 2: Runtime comparison (log scale)
    ax_runtime = fig.add_subplot(gs[1, 1])
    times_ms = [greedy_time * 1000, hc_time * 1000, sa_time * 1000]
    
    bars_time = ax_runtime.bar(algorithms, times_ms, color=['steelblue', 'coral', 'lightgreen'],
                                alpha=0.8, edgecolor='black', linewidth=1.2)
    ax_runtime.set_ylabel('Runtime (ms, log)', fontsize=10, weight='bold')
    ax_runtime.set_title('Runtime Comparison', fontsize=10, weight='bold')
    ax_runtime.set_yscale('log')
    ax_runtime.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    # Set y-axis limits
    ax_runtime.set_ylim(bottom=max(0.01, min(times_ms) * 0.5))
    
    # Add value labels on runtime bars
    for bar in bars_time:
        height = bar.get_height()
        ax_runtime.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Right column, row 3: Route info box
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis('off')
    
    # Get route names and wrap
    route_names = [stops["names"][i] for i in best_route]
    route_str = " → ".join(route_names)
    wrapped_route = textwrap.fill(route_str, width=35)
    
    info_text = f"Best Route ({best_name}):\n{wrapped_route}\n\n"
    info_text += f"Stops: {len(best_route)}\n"
    info_text += f"Parcels: {len(parcels)}\n"
    info_text += f"Cost: {best_cost:.1f}\n"
    info_text += f"Runtime: {(hc_time if best_name == 'Hill Climbing' else sa_time)*1000:.2f} ms"
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.suptitle('WarehouseBot: Task Sequencing Optimization', 
                 fontsize=13, weight='bold')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        print("Displaying visualization...")
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Check if matplotlib is available
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is not installed.")
        print("Please install it with: pip install matplotlib")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize WarehouseBot')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to file instead of displaying (e.g., out.png)')
    
    # Random mode options
    parser.add_argument('--random', action='store_true',
                        help='Use random warehouse instance instead of static demo')
    parser.add_argument('--rows', type=int, default=25,
                        help='Grid rows for random mode (default: 25)')
    parser.add_argument('--cols', type=int, default=25,
                        help='Grid columns for random mode (default: 25)')
    parser.add_argument('--obstacle', type=float, default=0.20,
                        help='Obstacle probability for random mode (default: 0.20)')
    parser.add_argument('--parcels', type=int, default=7,
                        help='Number of parcels for random mode (default: 7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--algo', type=str, choices=['greedy', 'hill', 'sa', 'best'],
                        default='best', help='Algorithm to visualize (default: best)')
    
    # Interleaving search
    parser.add_argument('--find-interleaving', action='store_true',
                        help='Search for an instance with interleaved drops')
    parser.add_argument('--max-tries', type=int, default=50,
                        help='Maximum seeds to try when finding interleaving (default: 50)')
    
    args = parser.parse_args()
    
    # Random mode requires --save
    if args.random and not args.save:
        print("ERROR: --random mode requires --save PATH")
        sys.exit(1)
    
    # Generate visualization
    if args.random:
        if args.find_interleaving:
            # Search for interleaving example
            find_and_save_interleaving_example(
                rows=args.rows,
                cols=args.cols,
                obstacle_prob=args.obstacle,
                n_parcels=args.parcels,
                start_seed=args.seed,
                max_tries=args.max_tries,
                algo=args.algo,
                save_path=args.save
            )
        else:
            # Single random instance
            plot_random_instance(
                rows=args.rows,
                cols=args.cols,
                obstacle_prob=args.obstacle,
                n_parcels=args.parcels,
                seed=args.seed,
                algo=args.algo,
                save_path=args.save
            )
    else:
        # Static demo mode
        plot_static_comparison(save_path=args.save)
    
    print("\nVisualization complete!")
