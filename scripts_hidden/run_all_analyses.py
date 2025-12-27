#!/usr/bin/env python3
"""
Wrapper script to run all portfolio analyses for Assignment 3

This script runs portfolio_analysis.py with all required parameter combinations:
1. Size portfolios, 1927-2013
2. Size portfolios, 1927-2024
3. Value (BE-ME) portfolios, 1927-2013
4. Value (BE-ME) portfolios, 1927-2024

Analyses are run in parallel using multiprocessing for faster execution.
"""

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Configuration: List of analyses to run
# Required analyses (steps 2-7)
REQUIRED_ANALYSES = [
    {
        'name': 'Size Portfolios (1927-2013)',
        'args': ['--start-year', '1927', '--end-year', '2013', '--portfolio-type', 'size']
    },
    {
        'name': 'Size Portfolios (1927-2024)',
        'args': ['--start-year', '1927', '--end-year', '2024', '--portfolio-type', 'size']
    },
    {
        'name': 'Value Portfolios (1927-2013)',
        'args': ['--start-year', '1927', '--end-year', '2013', '--portfolio-type', 'value']
    },
    {
        'name': 'Value Portfolios (1927-2024)',
        'args': ['--start-year', '1927', '--end-year', '2024', '--portfolio-type', 'value']
    }
]

# Optional analyses (steps 8-9)
# Step 8: Exclude small caps
OPTIONAL_EXCLUDE_SMALL_CAPS = [
    {
        'name': 'Size Portfolios (1927-2013) - Exclude Small Caps',
        'args': ['--start-year', '1927', '--end-year', '2013', '--portfolio-type', 'size', '--exclude-small-caps']
    },
    {
        'name': 'Size Portfolios (1927-2024) - Exclude Small Caps',
        'args': ['--start-year', '1927', '--end-year', '2024', '--portfolio-type', 'size', '--exclude-small-caps']
    }
]

# Step 9: Recentre data
OPTIONAL_RECENTRE = [
    {
        'name': 'Size Portfolios (1927-2013) - Recentred',
        'args': ['--start-year', '1927', '--end-year', '2013', '--portfolio-type', 'size', '--recentre']
    },
    {
        'name': 'Size Portfolios (1927-2024) - Recentred',
        'args': ['--start-year', '1927', '--end-year', '2024', '--portfolio-type', 'size', '--recentre']
    },
    {
        'name': 'Value Portfolios (1927-2013) - Recentred',
        'args': ['--start-year', '1927', '--end-year', '2013', '--portfolio-type', 'value', '--recentre']
    },
    {
        'name': 'Value Portfolios (1927-2024) - Recentred',
        'args': ['--start-year', '1927', '--end-year', '2024', '--portfolio-type', 'value', '--recentre']
    }
]

# Step 9 (continued): Recentre + Exclude Small Caps
OPTIONAL_BOTH = [
    {
        'name': 'Size Portfolios (1927-2013) - Exclude Small Caps + Recentred',
        'args': ['--start-year', '1927', '--end-year', '2013', '--portfolio-type', 'size', '--exclude-small-caps', '--recentre']
    },
    {
        'name': 'Size Portfolios (1927-2024) - Exclude Small Caps + Recentred',
        'args': ['--start-year', '1927', '--end-year', '2024', '--portfolio-type', 'size', '--exclude-small-caps', '--recentre']
    }
]

# Combine all analyses (set to False to skip optional analyses)
INCLUDE_OPTIONAL = True

if INCLUDE_OPTIONAL:
    ANALYSES = (REQUIRED_ANALYSES + 
                OPTIONAL_EXCLUDE_SMALL_CAPS + 
                OPTIONAL_RECENTRE + 
                OPTIONAL_BOTH)
else:
    ANALYSES = REQUIRED_ANALYSES

def run_analysis_worker(analysis_tuple):
    """
    Worker function to run a single analysis (for multiprocessing)
    
    Parameters:
    -----------
    analysis_tuple : tuple
        (name, args) tuple where args is a list of command-line arguments
    
    Returns:
    --------
    tuple : (name, success, error_message)
    """
    name, args = analysis_tuple
    script_path = Path(__file__).parent / "portfolio_analysis.py"
    cmd = [sys.executable, str(script_path)] + args
    
    try:
        # Capture output to avoid interleaving in parallel execution
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return (name, True, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Error code: {e.returncode}"
        if e.stderr:
            error_msg += f"\n  {e.stderr[:200]}"  # First 200 chars of error
        return (name, False, error_msg)
    except Exception as e:
        return (name, False, str(e))


def run_analysis(name, args):
    """Run a single analysis (for sequential execution)"""
    print("\n" + "=" * 70)
    print(f"Running: {name}")
    print("=" * 70)
    
    script_path = Path(__file__).parent / "portfolio_analysis.py"
    cmd = [sys.executable, str(script_path)] + args
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Completed: {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {name}")
        print(f"  Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {name}: {e}")
        return False

def main(parallel=True, max_workers=None):
    """
    Run all analyses
    
    Parameters:
    -----------
    parallel : bool
        If True, run analyses in parallel using multiprocessing
    max_workers : int, optional
        Maximum number of parallel workers (default: number of CPU cores)
    """
    print("=" * 70)
    print("Running All Portfolio Analyses for Assignment 3")
    print("=" * 70)
    
    if INCLUDE_OPTIONAL:
        print(f"\nRequired analyses: {len(REQUIRED_ANALYSES)}")
        print(f"Optional analyses: {len(OPTIONAL_EXCLUDE_SMALL_CAPS + OPTIONAL_RECENTRE + OPTIONAL_BOTH)}")
        print(f"  - Exclude small caps: {len(OPTIONAL_EXCLUDE_SMALL_CAPS)}")
        print(f"  - Recentre: {len(OPTIONAL_RECENTRE)}")
        print(f"  - Both: {len(OPTIONAL_BOTH)}")
    else:
        print(f"\nRunning required analyses only (set INCLUDE_OPTIONAL=True for optional steps)")
    
    print(f"\nTotal analyses to run: {len(ANALYSES)}")
    
    if parallel:
        # Determine number of workers
        if max_workers is None:
            max_workers = min(cpu_count(), len(ANALYSES))
        print(f"\nRunning in parallel with {max_workers} workers...")
        
        # Prepare analysis tuples for parallel execution
        analysis_tuples = [(analysis['name'], analysis['args']) for analysis in ANALYSES]
        
        results = []
        completed = 0
        
        # Run analyses in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_analysis = {
                executor.submit(run_analysis_worker, (name, args)): (name, args)
                for name, args in analysis_tuples
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_analysis):
                name, success, error_msg = future.result()
                completed += 1
                
                if success:
                    print(f"[{completed}/{len(ANALYSES)}] ✓ Completed: {name}")
                else:
                    print(f"[{completed}/{len(ANALYSES)}] ✗ Failed: {name}")
                    if error_msg:
                        print(f"  {error_msg}")
                
                results.append((name, success))
    else:
        # Sequential execution (original behavior)
        print("\nRunning sequentially...")
        results = []
        for i, analysis in enumerate(ANALYSES, 1):
            print(f"\n[{i}/{len(ANALYSES)}] {analysis['name']}")
            success = run_analysis(analysis['name'], analysis['args'])
            results.append((analysis['name'], success))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    successful = [name for name, success in results if success]
    failed = [name for name, success in results if not success]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"  - {name}")
    
    print("\n" + "=" * 70)
    if not failed:
        print("All analyses completed successfully!")
    else:
        print(f"Completed with {len(failed)} failure(s)")
    print("=" * 70)
    
    return 0 if not failed else 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all portfolio analyses")
    parser.add_argument('--sequential', action='store_true',
                       help='Run analyses sequentially instead of in parallel')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    sys.exit(main(parallel=not args.sequential, max_workers=args.max_workers))

