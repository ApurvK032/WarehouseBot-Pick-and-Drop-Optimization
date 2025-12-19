#!/usr/bin/env python3
"""
One-shot artifact generator for WarehouseBot submission.

Usage:
    python3 generate_artifacts.py

This script generates all CSVs, images, and logs required for submission.
Safe to re-run (overwrites existing files).
"""

import sys
import subprocess
from pathlib import Path
import json


def create_directories():
    """Create output directories if they don't exist."""
    dirs = ["results", "images", "logs"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("✓ Created directories: results/, images/, logs/")


def generate_csvs():
    """Generate all required CSV files using evaluation functions."""
    print("\n=== Generating CSV Files ===")
    
    # Import here so we can catch import errors
    try:
        from warehousebot.evaluation import (
            run_random_trials, run_scaling_study, run_ablation,
            run_sa_sweep, write_csv, summarize_results
        )
    except ImportError as e:
        print(f"ERROR: Failed to import warehousebot.evaluation: {e}")
        return False, {}
    
    summaries = {}
    
    # 1. Random trials 15x15
    print("\n[1/5] Running random_trials_15x15_p5_o0.20_seed42...")
    try:
        trials_15x15 = run_random_trials(
            n_trials=30, rows=15, cols=15, 
            obstacle_prob=0.20, n_parcels=5, 
            heuristic="manhattan", seed=42
        )
        write_csv(trials_15x15, "results/random_trials_15x15_p5_o0.20_seed42.csv")
        summaries["random_trials_15x15"] = summarize_results(trials_15x15)
        print("  ✓ Saved: results/random_trials_15x15_p5_o0.20_seed42.csv")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False, summaries
    
    # 2. Baseline large 25x25
    print("\n[2/5] Running baseline_large_25x25_p7_o0.20_seed42...")
    try:
        trials_25x25 = run_random_trials(
            n_trials=30, rows=25, cols=25, 
            obstacle_prob=0.20, n_parcels=7, 
            heuristic="manhattan", seed=42
        )
        write_csv(trials_25x25, "results/baseline_large_25x25_p7_o0.20_seed42.csv")
        summaries["baseline_large_25x25"] = summarize_results(trials_25x25)
        print("  ✓ Saved: results/baseline_large_25x25_p7_o0.20_seed42.csv")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False, summaries
    
    # 3. Scaling study (already summarized data)
    print("\n[3/5] Running scaling_study_seed42...")
    try:
        scaling_data = run_scaling_study(
            n_trials=15, 
            grid_sizes=[(10,10), (15,15), (20,20)], 
            obstacle_probs=[0.1, 0.2, 0.3], 
            parcel_counts=[3, 5], 
            heuristic="manhattan", 
            seed=42
        )
        write_csv(scaling_data, "results/scaling_study_seed42.csv")
        # Store raw data - it's already a summary
        summaries["scaling_study"] = scaling_data
        print("  ✓ Saved: results/scaling_study_seed42.csv")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False, summaries
    
    # 4. Ablation study (already summarized data)
    print("\n[4/5] Running ablation_study_seed42...")
    try:
        ablation_data = run_ablation(
            n_trials=20, rows=15, cols=15, 
            obstacle_prob=0.2, n_parcels=5, 
            heuristic="manhattan", seed=42
        )
        write_csv(ablation_data, "results/ablation_study_seed42.csv")
        # Store raw data - it's already a summary
        summaries["ablation_study"] = ablation_data
        print("  ✓ Saved: results/ablation_study_seed42.csv")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False, summaries
    
    # 5. SA parameter sweep (already summarized data)
    print("\n[5/5] Running sa_sweep_seed42...")
    try:
        sa_data = run_sa_sweep(
            n_trials=15, rows=15, cols=15, 
            obstacle_prob=0.2, n_parcels=5, 
            t0_values=[10.0, 50.0, 100.0], 
            alpha_values=[0.99, 0.995, 0.999], 
            heuristic="manhattan", 
            seed=42
        )
        write_csv(sa_data, "results/sa_sweep_seed42.csv")
        # Store raw data - it's already a summary
        summaries["sa_sweep"] = sa_data
        print("  ✓ Saved: results/sa_sweep_seed42.csv")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False, summaries
    
    print("\n✓ All CSV files generated successfully")
    return True, summaries


def write_summary_log(summaries):
    """Write summary statistics to logs/summary.txt."""
    print("\n=== Writing Summary Log ===")
    
    log_path = Path("logs/summary.txt")
    try:
        with open(log_path, "w") as f:
            f.write("WarehouseBot Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Process each study type differently
            for study_name, data in summaries.items():
                f.write(f"{study_name.upper()}\n")
                f.write("-" * 60 + "\n")
                
                if study_name in ["random_trials_15x15", "baseline_large_25x25"]:
                    # These are summarize_results() outputs (dicts)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                f.write(f"  {key}:\n")
                                for subkey, subval in value.items():
                                    f.write(f"    {subkey}: {subval}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {data}\n")
                
                elif study_name == "scaling_study":
                    # Scaling data is a list of configuration summary dicts
                    f.write(f"  Number of configurations: {len(data)}\n\n")
                    
                    # Find best configurations by hill_cost_mean and sa_cost_mean
                    if data:
                        best_hill = min(data, key=lambda x: x.get('hill_cost_mean', float('inf')))
                        best_sa = min(data, key=lambda x: x.get('sa_cost_mean', float('inf')))
                        
                        f.write("  Best Hill Climbing configuration:\n")
                        f.write(f"    rows={best_hill.get('rows')}, cols={best_hill.get('cols')}, ")
                        f.write(f"obstacle_prob={best_hill.get('obstacle_prob')}, ")
                        f.write(f"n_parcels={best_hill.get('n_parcels')}\n")
                        f.write(f"    hill_cost_mean: {best_hill.get('hill_cost_mean')}\n")
                        f.write(f"    hill_impr_pct_mean: {best_hill.get('hill_impr_pct_mean')}\n\n")
                        
                        f.write("  Best Simulated Annealing configuration:\n")
                        f.write(f"    rows={best_sa.get('rows')}, cols={best_sa.get('cols')}, ")
                        f.write(f"obstacle_prob={best_sa.get('obstacle_prob')}, ")
                        f.write(f"n_parcels={best_sa.get('n_parcels')}\n")
                        f.write(f"    sa_cost_mean: {best_sa.get('sa_cost_mean')}\n")
                        f.write(f"    sa_impr_pct_mean: {best_sa.get('sa_impr_pct_mean')}\n")
                
                elif study_name == "sa_sweep":
                    # SA sweep data is a list of parameter combination dicts
                    # Sort by sa_cost_mean and show top 5
                    sorted_params = sorted(data, key=lambda x: x.get('sa_cost_mean', float('inf')))
                    top_5 = sorted_params[:5]
                    
                    f.write("  Top 5 parameter combinations (by sa_cost_mean):\n\n")
                    for i, params in enumerate(top_5, 1):
                        f.write(f"  {i}. t0={params.get('t0')}, alpha={params.get('alpha')}\n")
                        f.write(f"     sa_cost_mean: {params.get('sa_cost_mean')}\n")
                        f.write(f"     sa_impr_pct_mean: {params.get('sa_impr_pct_mean')}\n")
                        if i < len(top_5):
                            f.write("\n")
                
                elif study_name == "ablation_study":
                    # Ablation data is a list of variant summary dicts
                    f.write("  Variant summaries:\n\n")
                    for variant in data:
                        variant_name = variant.get('variant', 'unknown')
                        f.write(f"  {variant_name}:\n")
                        f.write(f"    mean_cost: {variant.get('mean_cost')}\n")
                        f.write(f"    mean_impr_pct: {variant.get('mean_impr_pct')}\n")
                        f.write(f"    mean_time: {variant.get('mean_time')}\n")
                        f.write("\n")
                
                f.write("\n")
        
        print(f"  ✓ Saved: {log_path}")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to write summary log: {e}")


def generate_images():
    """Generate visualization images via subprocess calls."""
    print("\n=== Generating Images ===")
    
    images = [
        {
            "name": "static_comparison.png",
            "cmd": [
                sys.executable, "-m", "warehousebot.visualize",
                "--save", "images/static_comparison.png"
            ],
            "desc": "Static comparison figure"
        },
        {
            "name": "random_30x30_seed42.png",
            "cmd": [
                sys.executable, "-m", "warehousebot.visualize",
                "--random", "--rows", "30", "--cols", "30",
                "--obstacle", "0.20", "--parcels", "8",
                "--seed", "42", "--algo", "best",
                "--save", "images/random_30x30_seed42.png"
            ],
            "desc": "Random instance (30x30, 8 parcels)"
        },
        {
            "name": "interleaving_30x30.png",
            "cmd": [
                sys.executable, "-m", "warehousebot.visualize",
                "--random", "--find-interleaving",
                "--rows", "30", "--cols", "30",
                "--obstacle", "0.20", "--parcels", "8",
                "--seed", "42", "--max-tries", "200",
                "--algo", "best",
                "--save", "images/interleaving_30x30.png"
            ],
            "desc": "Interleaving example search"
        }
    ]
    
    failed = []
    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Generating {img['desc']}...")
        try:
            result = subprocess.run(
                img["cmd"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per image
            )
            
            if result.returncode == 0:
                print(f"  ✓ Saved: images/{img['name']}")
            else:
                failed.append(img['name'])
                print(f"  ⚠ Warning: Failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"    stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            failed.append(img['name'])
            print(f"  ⚠ Warning: Timed out after 5 minutes")
        except FileNotFoundError:
            failed.append(img['name'])
            print(f"  ⚠ Warning: visualize module not found or matplotlib not installed")
        except Exception as e:
            failed.append(img['name'])
            print(f"  ⚠ Warning: {e}")
    
    if not failed:
        print("\n✓ All images generated successfully")
    else:
        print(f"\n⚠ {len(failed)} image(s) failed (see warnings above)")
        print("  Images are optional; CSVs are complete.")


def print_final_checklist():
    """Print final checklist of generated artifacts."""
    print("\n" + "=" * 60)
    print("DONE - Artifact Generation Complete")
    print("=" * 60)
    
    csv_files = [
        "results/random_trials_15x15_p5_o0.20_seed42.csv",
        "results/baseline_large_25x25_p7_o0.20_seed42.csv",
        "results/scaling_study_seed42.csv",
        "results/ablation_study_seed42.csv",
        "results/sa_sweep_seed42.csv"
    ]
    
    image_files = [
        "images/static_comparison.png",
        "images/random_30x30_seed42.png",
        "images/interleaving_30x30.png"
    ]
    
    log_files = [
        "logs/summary.txt"
    ]
    
    print("\nCSV Results:")
    for f in csv_files:
        p = Path(f)
        status = "✓" if p.exists() else "✗"
        print(f"  {status} {f}")
    
    print("\nImages:")
    for f in image_files:
        p = Path(f)
        status = "✓" if p.exists() else "⚠"
        print(f"  {status} {f}")
    
    print("\nLogs:")
    for f in log_files:
        p = Path(f)
        status = "✓" if p.exists() else "⚠"
        print(f"  {status} {f}")
    
    print("\n" + "=" * 60)


def main():
    """Main execution flow."""
    print("WarehouseBot Artifact Generator")
    print("=" * 60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Generate CSVs
    csv_success, summaries = generate_csvs()
    if not csv_success:
        print("\n✗ CRITICAL ERROR: CSV generation failed")
        print("  Exiting with error code 1")
        sys.exit(1)
    
    # Step 3: Write summary log
    write_summary_log(summaries)
    
    # Step 4: Generate images (non-critical, warnings only)
    generate_images()
    
    # Step 5: Print final checklist
    print_final_checklist()
    
    print("\nAll critical artifacts generated successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
