#!/usr/bin/env python3
"""
Wrapper script to run all portfolio analyses for Assignment 3

This script runs portfolio_analysis.py with all required parameter combinations:
1. Size portfolios, 1927-2013
2. Size portfolios, 1927-2024
3. Value (BE-ME) portfolios, 1927-2013
4. Value (BE-ME) portfolios, 1927-2024
"""

import subprocess
import sys
from pathlib import Path

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

def run_analysis(name, args):
    """Run a single analysis"""
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

def main():
    """Run all analyses"""
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
    sys.exit(main())

