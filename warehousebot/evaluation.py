"""
Evaluation and benchmarking for WarehouseBot optimization algorithms.
Compares greedy, hill climbing, and simulated annealing across multiple scenarios.
"""

import argparse
import csv
import math
import random
import statistics
import time
from typing import List, Dict, Tuple, Any

from .warehouse import make_static_demo, build_stops, generate_random_warehouse
from .cost_matrix import compute_cost_matrix
from .sequencing import (
    greedy_route,
    hill_climb,
    simulated_annealing,
    route_cost,
    is_route_feasible,
    random_feasible_route
)


def write_csv(rows: list, path: str) -> None:
    """
    Write list of dicts to CSV using DictWriter.
    
    Args:
        rows: List of dict, each dict is one row
        path: Output CSV file path
    """
    if not rows:
        print(f"Warning: No rows to write to {path}")
        return
    
    # Collect all unique keys across all dicts
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    # Sort headers alphabetically
    headers = sorted(all_keys)
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"✓ Exported {len(rows)} rows to {path}")


def evaluate_instance(
    grid: List[List[int]],
    start: Tuple[int, int],
    parcels: List[Dict],
    heuristic: str = "manhattan",
    seed: int = 0,
    hill_kwargs: dict = None,
    sa_kwargs: dict = None
) -> Dict:
    """
    Evaluate all optimization algorithms on a single warehouse instance.
    
    Args:
        grid: 2D warehouse grid
        start: Starting position
        parcels: List of parcel dicts with "pickup" and "drop"
        heuristic: "manhattan" or "euclidean"
        seed: Random seed for optimization algorithms
        hill_kwargs: Optional kwargs for hill_climb (default {})
        sa_kwargs: Optional kwargs for simulated_annealing (default {})
    
    Returns:
        dict with performance metrics for all algorithms
    """
    if hill_kwargs is None:
        hill_kwargs = {}
    if sa_kwargs is None:
        sa_kwargs = {}
    
    # Build stops
    stops = build_stops(start, parcels)
    n_stops = len(stops["names"])
    n_parcels = len(parcels)
    
    # Compute cost matrix
    t0 = time.perf_counter()
    cost_result = compute_cost_matrix(grid, stops["coords"], heuristic=heuristic)
    cost_matrix_time = time.perf_counter() - t0
    
    cost_matrix = cost_result["matrix"]
    expanded_total = cost_result["expanded_total"]
    
    # Greedy algorithm (with error handling for impossible cases)
    try:
        t0 = time.perf_counter()
        greedy = greedy_route(cost_matrix, stops["precedence"])
        greedy_time = time.perf_counter() - t0
        
        greedy_cost = route_cost(greedy, cost_matrix)
        greedy_feasible = is_route_feasible(greedy, stops["precedence"])
    except RuntimeError as e:
        # Greedy failed (e.g., circular precedence or unreachable nodes)
        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        
        return {
            "status": "fail",
            "fail_reason": str(e),
            "n_stops": n_stops,
            "n_parcels": n_parcels,
            "rows": rows,
            "cols": cols,
            "obstacle_prob": None,
            "seed": seed,
            "heuristic": heuristic,
            "cost_matrix_time": cost_matrix_time,
            "greedy_time": 0.0,
            "hill_time": 0.0,
            "sa_time": 0.0,
            "greedy_cost": math.inf,
            "hill_cost": math.inf,
            "sa_cost": math.inf,
            "hill_impr_pct": 0.0,
            "sa_impr_pct": 0.0,
            "expanded_total": expanded_total,
            "greedy_feasible": False,
            "hill_feasible": False,
            "sa_feasible": False,
            "warnings": [f"Greedy failed: {str(e)}"],
        }
    
    # Hill climbing - merge default parameters with user-provided kwargs
    hill_params = {
        "max_iters": 2000,
        "neighbors_per_iter": 50,
        "seed": seed,
        "restarts": 10
    }
    hill_params.update(hill_kwargs)
    
    t0 = time.perf_counter()
    hc_result = hill_climb(
        cost_matrix,
        stops["precedence"],
        **hill_params
    )
    hill_time = time.perf_counter() - t0
    
    hill_route = hc_result["best_route"]
    hill_cost = hc_result["best_cost"]
    hill_feasible = is_route_feasible(hill_route, stops["precedence"])
    
    # Simulated annealing - merge default parameters with user-provided kwargs
    sa_params = {
        "max_iters": 5000,
        "t0": 50.0,
        "alpha": 0.995,
        "seed": seed
    }
    sa_params.update(sa_kwargs)
    
    t0 = time.perf_counter()
    sa_result = simulated_annealing(
        cost_matrix,
        stops["precedence"],
        **sa_params
    )
    sa_time = time.perf_counter() - t0
    
    sa_route = sa_result["best_route"]
    sa_cost = sa_result["best_cost"]
    sa_feasible = is_route_feasible(sa_route, stops["precedence"])
    
    # Calculate improvement percentages
    if greedy_cost > 0 and greedy_cost != math.inf:
        hill_impr_pct = ((greedy_cost - hill_cost) / greedy_cost) * 100
        sa_impr_pct = ((greedy_cost - sa_cost) / greedy_cost) * 100
    else:
        hill_impr_pct = 0.0
        sa_impr_pct = 0.0
    
    # Check for infeasible routes
    warnings = []
    if not greedy_feasible:
        warnings.append("Greedy route is infeasible!")
    if not hill_feasible:
        warnings.append("Hill climbing route is infeasible!")
    if not sa_feasible:
        warnings.append("Simulated annealing route is infeasible!")
    
    # Config fields (obstacle_prob not available from caller, set to None)
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    
    return {
        "status": "success",
        "n_stops": n_stops,
        "n_parcels": n_parcels,
        "rows": rows,
        "cols": cols,
        "obstacle_prob": None,  # Not available from grid alone
        "seed": seed,
        "heuristic": heuristic,
        "cost_matrix_time": cost_matrix_time,
        "greedy_time": greedy_time,
        "hill_time": hill_time,
        "sa_time": sa_time,
        "greedy_cost": greedy_cost,
        "hill_cost": hill_cost,
        "sa_cost": sa_cost,
        "hill_impr_pct": hill_impr_pct,
        "sa_impr_pct": sa_impr_pct,
        "expanded_total": expanded_total,
        "greedy_feasible": greedy_feasible,
        "hill_feasible": hill_feasible,
        "sa_feasible": sa_feasible,
        "warnings": warnings
    }


def run_static_demo() -> Dict:
    """
    Evaluate algorithms on the static demo warehouse.
    
    Returns:
        Evaluation results dict
    """
    print("=" * 70)
    print("EVALUATION: STATIC DEMO WAREHOUSE")
    print("=" * 70)
    
    demo = make_static_demo()
    
    print(f"\nWarehouse: {len(demo['grid'])}x{len(demo['grid'][0])} grid")
    print(f"Parcels: {len(demo['parcels'])}")
    print(f"Start: {demo['start']}")
    
    result = evaluate_instance(
        demo["grid"],
        demo["start"],
        demo["parcels"],
        heuristic="manhattan",
        seed=42
    )
    
    print(f"\nStops: {result['n_stops']}")
    print(f"Heuristic: {result['heuristic']}")
    
    print("\n" + "-" * 70)
    print("ALGORITHM PERFORMANCE")
    print("-" * 70)
    
    print(f"\n{'Algorithm':<20} {'Cost':<10} {'Time (ms)':<12} {'Improvement':<12}")
    print("-" * 70)
    
    print(f"{'Cost Matrix':<20} {'-':<10} {result['cost_matrix_time']*1000:>10.2f} ms {'-':<12}")
    print(f"{'Greedy':<20} {result['greedy_cost']:<10.1f} {result['greedy_time']*1000:>10.2f} ms {'(baseline)':<12}")
    print(f"{'Hill Climbing':<20} {result['hill_cost']:<10.1f} {result['hill_time']*1000:>10.2f} ms {result['hill_impr_pct']:>10.2f} %")
    print(f"{'Simulated Annealing':<20} {result['sa_cost']:<10.1f} {result['sa_time']*1000:>10.2f} ms {result['sa_impr_pct']:>10.2f} %")
    
    print(f"\nA* nodes expanded: {result['expanded_total']}")
    print(f"All routes feasible: {result['greedy_feasible'] and result['hill_feasible'] and result['sa_feasible']}")
    
    if result['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in result['warnings']:
            print(f"   {warning}")
    
    return result


def run_random_trials(
    n_trials: int = 20,
    rows: int = 12,
    cols: int = 12,
    obstacle_prob: float = 0.2,
    n_parcels: int = 4,
    heuristic: str = "manhattan",
    seed: int = 0
) -> List[Dict]:
    """
    Run evaluation on multiple randomly generated warehouses.
    
    Args:
        n_trials: Number of random instances to generate
        rows: Grid rows
        cols: Grid columns
        obstacle_prob: Obstacle probability (0.0 to 1.0)
        n_parcels: Number of parcels per instance
        heuristic: "manhattan" or "euclidean"
        seed: Base random seed
    
    Returns:
        List of evaluation result dicts (per-trial rows)
    """
    print("\n" + "=" * 70)
    print(f"EVALUATION: {n_trials} RANDOM WAREHOUSE INSTANCES")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {rows}x{cols}")
    print(f"  Obstacle probability: {obstacle_prob:.2f}")
    print(f"  Parcels per instance: {n_parcels}")
    print(f"  Heuristic: {heuristic}")
    print(f"  Base seed: {seed}")
    
    results = []
    skipped = 0
    
    print(f"\nRunning trials...", end="", flush=True)
    
    for trial in range(n_trials):
        if (trial + 1) % 5 == 0:
            print(f" {trial + 1}", end="", flush=True)
        
        try:
            # Generate random warehouse
            warehouse = generate_random_warehouse(
                rows=rows,
                cols=cols,
                obstacle_prob=obstacle_prob,
                n_parcels=n_parcels,
                seed=seed + trial,
                max_tries=200
            )
            
            # Evaluate
            result = evaluate_instance(
                warehouse["grid"],
                warehouse["start"],
                warehouse["parcels"],
                heuristic=heuristic,
                seed=seed + trial
            )
            
            # Check if evaluation failed
            if result.get("status") == "fail":
                skipped += 1
            else:
                # Add config fields for CSV export
                result["obstacle_prob"] = obstacle_prob
                results.append(result)
            
        except RuntimeError as e:
            # Failed to generate feasible warehouse
            skipped += 1
    
    print(" Done!")
    
    if skipped > 0:
        print(f"\n⚠️  Skipped {skipped} trials (failed to generate feasible warehouse)")
    
    print(f"\nSuccessfully evaluated {len(results)} instances")
    
    return results


def summarize_results(results: List[Dict]) -> Dict:
    """
    Compute summary statistics across multiple evaluation results.
    
    Args:
        results: List of evaluation result dicts
    
    Returns:
        dict with summary statistics
    """
    if not results:
        print("\n⚠️  No results to summarize")
        return {}
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Extract metrics
    greedy_costs = [r["greedy_cost"] for r in results if r["greedy_cost"] != math.inf]
    hill_costs = [r["hill_cost"] for r in results if r["hill_cost"] != math.inf]
    sa_costs = [r["sa_cost"] for r in results if r["sa_cost"] != math.inf]
    
    hill_impr_pcts = [r["hill_impr_pct"] for r in results]
    sa_impr_pcts = [r["sa_impr_pct"] for r in results]
    
    # Total runtimes (cost matrix + algorithm)
    greedy_times = [r["cost_matrix_time"] + r["greedy_time"] for r in results]
    hill_times = [r["cost_matrix_time"] + r["hill_time"] for r in results]
    sa_times = [r["cost_matrix_time"] + r["sa_time"] for r in results]
    
    # Compute statistics
    summary = {
        "n_trials": len(results),
        "greedy_cost_mean": statistics.mean(greedy_costs) if greedy_costs else 0,
        "greedy_cost_stdev": statistics.pstdev(greedy_costs) if len(greedy_costs) > 1 else 0,
        "hill_cost_mean": statistics.mean(hill_costs) if hill_costs else 0,
        "hill_cost_stdev": statistics.pstdev(hill_costs) if len(hill_costs) > 1 else 0,
        "sa_cost_mean": statistics.mean(sa_costs) if sa_costs else 0,
        "sa_cost_stdev": statistics.pstdev(sa_costs) if len(sa_costs) > 1 else 0,
        "hill_impr_mean": statistics.mean(hill_impr_pcts) if hill_impr_pcts else 0,
        "hill_impr_stdev": statistics.pstdev(hill_impr_pcts) if len(hill_impr_pcts) > 1 else 0,
        "sa_impr_mean": statistics.mean(sa_impr_pcts) if sa_impr_pcts else 0,
        "sa_impr_stdev": statistics.pstdev(sa_impr_pcts) if len(sa_impr_pcts) > 1 else 0,
        "greedy_time_mean": statistics.mean(greedy_times) if greedy_times else 0,
        "hill_time_mean": statistics.mean(hill_times) if hill_times else 0,
        "sa_time_mean": statistics.mean(sa_times) if sa_times else 0,
    }
    
    # Print summary table
    print(f"\nTrials analyzed: {summary['n_trials']}")
    
    print("\n" + "-" * 70)
    print("COST COMPARISON")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Mean Cost':<15} {'Std Dev':<15}")
    print("-" * 70)
    print(f"{'Greedy':<20} {summary['greedy_cost_mean']:>12.2f}   {summary['greedy_cost_stdev']:>12.2f}")
    print(f"{'Hill Climbing':<20} {summary['hill_cost_mean']:>12.2f}   {summary['hill_cost_stdev']:>12.2f}")
    print(f"{'Simulated Annealing':<20} {summary['sa_cost_mean']:>12.2f}   {summary['sa_cost_stdev']:>12.2f}")
    
    print("\n" + "-" * 70)
    print("IMPROVEMENT OVER GREEDY (%)")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Mean Impr %':<15} {'Std Dev':<15}")
    print("-" * 70)
    print(f"{'Hill Climbing':<20} {summary['hill_impr_mean']:>12.2f} % {summary['hill_impr_stdev']:>12.2f}")
    print(f"{'Simulated Annealing':<20} {summary['sa_impr_mean']:>12.2f} % {summary['sa_impr_stdev']:>12.2f}")
    
    print("\n" + "-" * 70)
    print("RUNTIME COMPARISON (seconds)")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Mean Time':<15}")
    print("-" * 70)
    print(f"{'Greedy':<20} {summary['greedy_time_mean']:>12.4f} s")
    print(f"{'Hill Climbing':<20} {summary['hill_time_mean']:>12.4f} s")
    print(f"{'Simulated Annealing':<20} {summary['sa_time_mean']:>12.4f} s")
    
    # Speedup factors
    if summary['greedy_time_mean'] > 0:
        print("\n" + "-" * 70)
        print("RUNTIME OVERHEAD vs GREEDY")
        print("-" * 70)
        hill_overhead = summary['hill_time_mean'] / summary['greedy_time_mean']
        sa_overhead = summary['sa_time_mean'] / summary['greedy_time_mean']
        print(f"{'Hill Climbing':<20} {hill_overhead:>12.2f}x slower")
        print(f"{'Simulated Annealing':<20} {sa_overhead:>12.2f}x slower")
    
    return summary


def run_scaling_study(
    n_trials: int,
    grid_sizes: List[Tuple[int, int]],
    obstacle_probs: List[float],
    parcel_counts: List[int],
    heuristic: str = "manhattan",
    seed: int = 0
) -> List[Dict]:
    """
    Run a scaling study across multiple configurations.
    
    Args:
        n_trials: Number of trials per configuration
        grid_sizes: List of (rows, cols) tuples
        obstacle_probs: List of obstacle probabilities
        parcel_counts: List of parcel counts
        heuristic: Heuristic to use
        seed: Base seed
    
    Returns:
        List of summary dicts (one per config)
    """
    print("\n" + "=" * 70)
    print("SCALING STUDY")
    print("=" * 70)
    
    summaries = []
    
    for grid_size in grid_sizes:
        for obstacle_prob in obstacle_probs:
            for n_parcels in parcel_counts:
                rows, cols = grid_size
                
                print(f"\n--- Config: {rows}x{cols}, obstacle={obstacle_prob:.2f}, parcels={n_parcels} ---")
                
                # Run trials for this config
                results = run_random_trials(
                    n_trials=n_trials,
                    rows=rows,
                    cols=cols,
                    obstacle_prob=obstacle_prob,
                    n_parcels=n_parcels,
                    heuristic=heuristic,
                    seed=seed
                )
                
                if not results:
                    print("⚠️  No valid results for this config, skipping...")
                    continue
                
                # Compute summary statistics
                greedy_costs = [r["greedy_cost"] for r in results if r["greedy_cost"] != math.inf]
                hill_costs = [r["hill_cost"] for r in results if r["hill_cost"] != math.inf]
                sa_costs = [r["sa_cost"] for r in results if r["sa_cost"] != math.inf]
                
                greedy_times = [r["cost_matrix_time"] + r["greedy_time"] for r in results]
                hill_times = [r["cost_matrix_time"] + r["hill_time"] for r in results]
                sa_times = [r["cost_matrix_time"] + r["sa_time"] for r in results]
                
                hill_impr_pcts = [r["hill_impr_pct"] for r in results]
                sa_impr_pcts = [r["sa_impr_pct"] for r in results]
                
                expanded_totals = [r["expanded_total"] for r in results]
                
                summary = {
                    "rows": rows,
                    "cols": cols,
                    "obstacle_prob": obstacle_prob,
                    "n_parcels": n_parcels,
                    "heuristic": heuristic,
                    "n_trials": len(results),
                    "greedy_cost_mean": statistics.mean(greedy_costs) if greedy_costs else 0,
                    "greedy_cost_stdev": statistics.pstdev(greedy_costs) if len(greedy_costs) > 1 else 0,
                    "hill_cost_mean": statistics.mean(hill_costs) if hill_costs else 0,
                    "hill_cost_stdev": statistics.pstdev(hill_costs) if len(hill_costs) > 1 else 0,
                    "sa_cost_mean": statistics.mean(sa_costs) if sa_costs else 0,
                    "sa_cost_stdev": statistics.pstdev(sa_costs) if len(sa_costs) > 1 else 0,
                    "hill_impr_pct_mean": statistics.mean(hill_impr_pcts) if hill_impr_pcts else 0,
                    "hill_impr_pct_stdev": statistics.pstdev(hill_impr_pcts) if len(hill_impr_pcts) > 1 else 0,
                    "sa_impr_pct_mean": statistics.mean(sa_impr_pcts) if sa_impr_pcts else 0,
                    "sa_impr_pct_stdev": statistics.pstdev(sa_impr_pcts) if len(sa_impr_pcts) > 1 else 0,
                    "greedy_time_mean": statistics.mean(greedy_times) if greedy_times else 0,
                    "greedy_time_stdev": statistics.pstdev(greedy_times) if len(greedy_times) > 1 else 0,
                    "hill_time_mean": statistics.mean(hill_times) if hill_times else 0,
                    "hill_time_stdev": statistics.pstdev(hill_times) if len(hill_times) > 1 else 0,
                    "sa_time_mean": statistics.mean(sa_times) if sa_times else 0,
                    "sa_time_stdev": statistics.pstdev(sa_times) if len(sa_times) > 1 else 0,
                    "expanded_total_mean": statistics.mean(expanded_totals) if expanded_totals else 0,
                    "expanded_total_stdev": statistics.pstdev(expanded_totals) if len(expanded_totals) > 1 else 0,
                }
                
                summaries.append(summary)
                
                # Print compact summary
                print(f"  Greedy:  cost={summary['greedy_cost_mean']:.2f}±{summary['greedy_cost_stdev']:.2f}, time={summary['greedy_time_mean']:.4f}s")
                print(f"  Hill:    cost={summary['hill_cost_mean']:.2f}±{summary['hill_cost_stdev']:.2f}, impr={summary['hill_impr_pct_mean']:.2f}%, time={summary['hill_time_mean']:.4f}s")
                print(f"  SA:      cost={summary['sa_cost_mean']:.2f}±{summary['sa_cost_stdev']:.2f}, impr={summary['sa_impr_pct_mean']:.2f}%, time={summary['sa_time_mean']:.4f}s")
                print(f"  A* expanded: {summary['expanded_total_mean']:.0f}±{summary['expanded_total_stdev']:.0f}")
    
    print(f"\n✓ Scaling study complete: {len(summaries)} configurations")
    return summaries


def run_sa_sweep(
    n_trials: int,
    rows: int,
    cols: int,
    obstacle_prob: float,
    n_parcels: int,
    t0_values: List[float],
    alpha_values: List[float],
    heuristic: str = "manhattan",
    seed: int = 0
) -> List[Dict]:
    """
    Run SA parameter sensitivity sweep.
    
    Args:
        n_trials: Number of trials per parameter combo
        rows: Grid rows
        cols: Grid cols
        obstacle_prob: Obstacle probability
        n_parcels: Number of parcels
        t0_values: List of initial temperature values
        alpha_values: List of cooling rate values
        heuristic: Heuristic to use
        seed: Base seed
    
    Returns:
        List of summary dicts (one per t0/alpha combo)
    """
    print("\n" + "=" * 70)
    print("SA PARAMETER SWEEP")
    print("=" * 70)
    print(f"\nGrid: {rows}x{cols}, obstacle={obstacle_prob:.2f}, parcels={n_parcels}")
    print(f"Trials per config: {n_trials}")
    
    summaries = []
    
    for t0 in t0_values:
        for alpha in alpha_values:
            print(f"\n--- Testing: t0={t0}, alpha={alpha} ---")
            
            results = []
            skipped = 0
            
            for trial in range(n_trials):
                try:
                    warehouse = generate_random_warehouse(
                        rows=rows,
                        cols=cols,
                        obstacle_prob=obstacle_prob,
                        n_parcels=n_parcels,
                        seed=seed + trial,
                        max_tries=200
                    )
                    
                    result = evaluate_instance(
                        warehouse["grid"],
                        warehouse["start"],
                        warehouse["parcels"],
                        heuristic=heuristic,
                        seed=seed + trial,
                        sa_kwargs={"t0": t0, "alpha": alpha}
                    )
                    
                    results.append(result)
                    
                except RuntimeError:
                    skipped += 1
            
            if not results:
                print("⚠️  No valid results for this config, skipping...")
                continue
            
            # Compute summary
            greedy_costs = [r["greedy_cost"] for r in results if r["greedy_cost"] != math.inf]
            hill_costs = [r["hill_cost"] for r in results if r["hill_cost"] != math.inf]
            sa_costs = [r["sa_cost"] for r in results if r["sa_cost"] != math.inf]
            sa_impr_pcts = [r["sa_impr_pct"] for r in results]
            
            summary = {
                "rows": rows,
                "cols": cols,
                "obstacle_prob": obstacle_prob,
                "n_parcels": n_parcels,
                "t0": t0,
                "alpha": alpha,
                "n_trials": len(results),
                "greedy_cost_mean": statistics.mean(greedy_costs) if greedy_costs else 0,
                "hill_cost_mean": statistics.mean(hill_costs) if hill_costs else 0,
                "sa_cost_mean": statistics.mean(sa_costs) if sa_costs else 0,
                "sa_cost_stdev": statistics.pstdev(sa_costs) if len(sa_costs) > 1 else 0,
                "sa_impr_pct_mean": statistics.mean(sa_impr_pcts) if sa_impr_pcts else 0,
                "sa_impr_pct_stdev": statistics.pstdev(sa_impr_pcts) if len(sa_impr_pcts) > 1 else 0,
            }
            
            summaries.append(summary)
            
            print(f"  SA cost: {summary['sa_cost_mean']:.2f}±{summary['sa_cost_stdev']:.2f}")
            print(f"  SA impr: {summary['sa_impr_pct_mean']:.2f}%±{summary['sa_impr_pct_stdev']:.2f}%")
    
    # Sort by best mean SA cost (ascending)
    summaries.sort(key=lambda x: x["sa_cost_mean"])
    
    print("\n" + "=" * 70)
    print("SA SWEEP RESULTS (sorted by best mean SA cost)")
    print("=" * 70)
    print(f"{'t0':<10} {'alpha':<10} {'SA Cost':<20} {'SA Impr %':<20}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['t0']:<10.1f} {s['alpha']:<10.3f} {s['sa_cost_mean']:>8.2f}±{s['sa_cost_stdev']:<8.2f} {s['sa_impr_pct_mean']:>8.2f}%±{s['sa_impr_pct_stdev']:<8.2f}%")
    
    print(f"\n✓ SA sweep complete: {len(summaries)} configurations")
    return summaries


def run_ablation(
    n_trials: int,
    rows: int,
    cols: int,
    obstacle_prob: float,
    n_parcels: int,
    heuristic: str = "manhattan",
    seed: int = 0
) -> List[Dict]:
    """
    Run ablation study comparing algorithm variants.
    
    Compares:
    1. Hill climb restarts=1 vs restarts=10
    2. SA with greedy start vs SA with random feasible start
    
    Args:
        n_trials: Number of trials
        rows: Grid rows
        cols: Grid cols
        obstacle_prob: Obstacle probability
        n_parcels: Number of parcels
        heuristic: Heuristic to use
        seed: Base seed
    
    Returns:
        List of summary dicts (one per variant)
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"\nGrid: {rows}x{cols}, obstacle={obstacle_prob:.2f}, parcels={n_parcels}")
    print(f"Trials: {n_trials}")
    
    # Generate shared random instances for fair comparison
    warehouses = []
    for trial in range(n_trials):
        try:
            warehouse = generate_random_warehouse(
                rows=rows,
                cols=cols,
                obstacle_prob=obstacle_prob,
                n_parcels=n_parcels,
                seed=seed + trial,
                max_tries=200
            )
            warehouses.append(warehouse)
        except RuntimeError:
            pass
    
    if not warehouses:
        print("⚠️  Failed to generate any warehouses")
        return []
    
    print(f"\nGenerated {len(warehouses)} warehouses")
    
    # Variant 1: Hill climb restarts=1
    print("\n--- Hill Climb: restarts=1 ---")
    hill1_results = []
    for i, warehouse in enumerate(warehouses):
        result = evaluate_instance(
            warehouse["grid"],
            warehouse["start"],
            warehouse["parcels"],
            heuristic=heuristic,
            seed=seed + i,
            hill_kwargs={"restarts": 1}
        )
        hill1_results.append(result)
    
    # Variant 2: Hill climb restarts=10 (default)
    print("--- Hill Climb: restarts=10 ---")
    hill10_results = []
    for i, warehouse in enumerate(warehouses):
        result = evaluate_instance(
            warehouse["grid"],
            warehouse["start"],
            warehouse["parcels"],
            heuristic=heuristic,
            seed=seed + i,
            hill_kwargs={"restarts": 10}
        )
        hill10_results.append(result)
    
    # Variant 3: SA with greedy start (default)
    print("--- SA: greedy start (default) ---")
    sa_greedy_results = []
    for i, warehouse in enumerate(warehouses):
        result = evaluate_instance(
            warehouse["grid"],
            warehouse["start"],
            warehouse["parcels"],
            heuristic=heuristic,
            seed=seed + i
        )
        sa_greedy_results.append(result)
    
    # Variant 4: SA with random feasible start
    print("--- SA: random feasible start ---")
    sa_random_results = []
    for i, warehouse in enumerate(warehouses):
        # Build cost matrix and stops
        stops = build_stops(warehouse["start"], warehouse["parcels"])
        cost_result = compute_cost_matrix(warehouse["grid"], stops["coords"], heuristic=heuristic)
        cost_matrix = cost_result["matrix"]
        
        # Generate random feasible start route
        n_stops = len(cost_matrix)
        random_start = random_feasible_route(n_stops, stops["precedence"], seed=seed + i)
        
        # Run SA with this start route
        t0 = time.perf_counter()
        sa_result = simulated_annealing(
            cost_matrix,
            stops["precedence"],
            start_route=random_start,
            max_iters=5000,
            t0=50.0,
            alpha=0.995,
            seed=seed + i
        )
        sa_time = time.perf_counter() - t0
        
        # Create result dict matching evaluate_instance format
        result = {
            "sa_cost": sa_result["best_cost"],
            "sa_time": sa_time,
            "greedy_cost": sa_greedy_results[i]["greedy_cost"],  # Use same greedy for comparison
        }
        
        if result["greedy_cost"] > 0 and result["greedy_cost"] != math.inf:
            result["sa_impr_pct"] = ((result["greedy_cost"] - result["sa_cost"]) / result["greedy_cost"]) * 100
        else:
            result["sa_impr_pct"] = 0.0
        
        sa_random_results.append(result)
    
    # Compute summaries
    summaries = []
    
    # Hill climb restarts=1
    hill1_costs = [r["hill_cost"] for r in hill1_results]
    hill1_impr = [r["hill_impr_pct"] for r in hill1_results]
    hill1_times = [r["hill_time"] for r in hill1_results]
    summaries.append({
        "variant": "Hill_restarts=1",
        "mean_cost": statistics.mean(hill1_costs) if hill1_costs else 0,
        "mean_impr_pct": statistics.mean(hill1_impr) if hill1_impr else 0,
        "mean_time": statistics.mean(hill1_times) if hill1_times else 0,
    })
    
    # Hill climb restarts=10
    hill10_costs = [r["hill_cost"] for r in hill10_results]
    hill10_impr = [r["hill_impr_pct"] for r in hill10_results]
    hill10_times = [r["hill_time"] for r in hill10_results]
    summaries.append({
        "variant": "Hill_restarts=10",
        "mean_cost": statistics.mean(hill10_costs) if hill10_costs else 0,
        "mean_impr_pct": statistics.mean(hill10_impr) if hill10_impr else 0,
        "mean_time": statistics.mean(hill10_times) if hill10_times else 0,
    })
    
    # SA greedy start
    sa_greedy_costs = [r["sa_cost"] for r in sa_greedy_results]
    sa_greedy_impr = [r["sa_impr_pct"] for r in sa_greedy_results]
    sa_greedy_times = [r["sa_time"] for r in sa_greedy_results]
    summaries.append({
        "variant": "SA_greedy_start",
        "mean_cost": statistics.mean(sa_greedy_costs) if sa_greedy_costs else 0,
        "mean_impr_pct": statistics.mean(sa_greedy_impr) if sa_greedy_impr else 0,
        "mean_time": statistics.mean(sa_greedy_times) if sa_greedy_times else 0,
    })
    
    # SA random start
    sa_random_costs = [r["sa_cost"] for r in sa_random_results]
    sa_random_impr = [r["sa_impr_pct"] for r in sa_random_results]
    sa_random_times = [r["sa_time"] for r in sa_random_results]
    summaries.append({
        "variant": "SA_random_start",
        "mean_cost": statistics.mean(sa_random_costs) if sa_random_costs else 0,
        "mean_impr_pct": statistics.mean(sa_random_impr) if sa_random_impr else 0,
        "mean_time": statistics.mean(sa_random_times) if sa_random_times else 0,
    })
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Variant':<25} {'Mean Cost':<15} {'Mean Impr %':<15} {'Mean Time (s)':<15}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['variant']:<25} {s['mean_cost']:>12.2f}   {s['mean_impr_pct']:>12.2f} % {s['mean_time']:>12.4f} s")
    
    print(f"\n✓ Ablation study complete")
    return summaries


def run_stress_tests(seed: int = 123) -> None:
    """
    Run stress tests with larger grids and more parcels.
    
    Args:
        seed: Base random seed
    """
    print("\n" + "=" * 70)
    print("STRESS TESTS - LARGE SCALE SCENARIOS")
    print("=" * 70)
    
    configs = [
        {"rows": 25, "cols": 25, "obstacle_prob": 0.20, "n_parcels": 7, "n_trials": 8},
        {"rows": 30, "cols": 30, "obstacle_prob": 0.20, "n_parcels": 8, "n_trials": 5},
    ]
    
    for i, config in enumerate(configs):
        print("\n" + "=" * 70)
        print(f"STRESS TEST {i+1}/{len(configs)}")
        print("=" * 70)
        print(f"Grid: {config['rows']}x{config['cols']}")
        print(f"Obstacle probability: {config['obstacle_prob']:.2f}")
        print(f"Parcels: {config['n_parcels']}")
        print(f"Trials: {config['n_trials']}")
        
        # Run trials
        results = run_random_trials(
            n_trials=config["n_trials"],
            rows=config["rows"],
            cols=config["cols"],
            obstacle_prob=config["obstacle_prob"],
            n_parcels=config["n_parcels"],
            heuristic="manhattan",
            seed=seed
        )
        
        # Summarize
        summarize_results(results)
        
        # Print mean A* nodes expanded
        if results:
            expanded_totals = [r["expanded_total"] for r in results]
            mean_expanded = statistics.mean(expanded_totals) if expanded_totals else 0
            print(f"\nMean A* nodes expanded: {mean_expanded:.0f}")
    
    print("\n" + "=" * 70)
    print("STRESS TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WarehouseBot Evaluation")
    parser.add_argument("--csv", type=str, help="Export random trials to CSV file")
    parser.add_argument("--stress", action="store_true", help="Run stress tests with large grids")
    parser.add_argument("--scaling", action="store_true", help="Run scaling study")
    parser.add_argument("--sa-sweep", action="store_true", help="Run SA parameter sweep")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--baseline-large", action="store_true", help="Run baseline evaluation with 25x25 grid and 7 parcels")
    
    args = parser.parse_args()
    
    # Always run static demo first
    run_static_demo()
    
    # If baseline-large flag is set, run with large configuration
    if args.baseline_large:
        print("\n" + "=" * 70)
        print("EVALUATION: LARGE BASELINE (25x25, parcels=7)")
        print("=" * 70)
        
        results = run_random_trials(
            n_trials=30,
            rows=25,
            cols=25,
            obstacle_prob=0.20,
            n_parcels=7,
            heuristic="manhattan",
            seed=42
        )
        
        # Summarize results
        summarize_results(results)
        
        # CSV export if requested
        if args.csv:
            write_csv(results, args.csv)
    # If stress flag is set, run stress tests instead of default random trials
    elif args.stress:
        run_stress_tests(seed=42)
    else:
        # Run random trials (default behavior)
        results = run_random_trials(
            n_trials=30,
            rows=15,
            cols=15,
            obstacle_prob=0.20,
            n_parcels=5,
            heuristic="manhattan",
            seed=42
        )
        
        # Summarize results
        summarize_results(results)
        
        # CSV export if requested
        if args.csv:
            write_csv(results, args.csv)
    
    # Additional studies if requested (run in order)
    if args.scaling:
        scaling_results = run_scaling_study(
            n_trials=15,
            grid_sizes=[(10, 10), (15, 15), (20, 20)],
            obstacle_probs=[0.1, 0.2, 0.3],
            parcel_counts=[3, 5],
            heuristic="manhattan",
            seed=42
        )
        write_csv(scaling_results, "scaling_study.csv")
    
    if args.sa_sweep:
        sa_sweep_results = run_sa_sweep(
            n_trials=15,
            rows=15,
            cols=15,
            obstacle_prob=0.2,
            n_parcels=5,
            t0_values=[10.0, 50.0, 100.0],
            alpha_values=[0.99, 0.995, 0.999],
            heuristic="manhattan",
            seed=42
        )
        write_csv(sa_sweep_results, "sa_sweep.csv")
    
    if args.ablation:
        ablation_results = run_ablation(
            n_trials=20,
            rows=15,
            cols=15,
            obstacle_prob=0.2,
            n_parcels=5,
            heuristic="manhattan",
            seed=42
        )
        write_csv(ablation_results, "ablation_study.csv")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
