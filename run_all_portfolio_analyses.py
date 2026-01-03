#!/usr/bin/env python3
"""
Run All Portfolio Analyses

This script runs all portfolio analysis scripts in the correct order:
1. Construct Investment Opportunity Set (IOS)
2. Find MSMP Portfolio
3. Find Optimal CRRA Portfolio
4. Find Zero-Beta Portfolio (ZBP)
5. Estimate Jensen's Alpha
6. Plot IOS and Key Portfolios

Dependencies are handled automatically (e.g., ZBP requires optimal CRRA).
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Script names
SCRIPTS = {
    'ios': 'construct_investment_opportunity_set.py',
    'msmp': 'find_msmp_portfolio.py',
    'optimal_crra': 'find_optimal_crra_portfolio.py',
    'zbp': 'find_zero_beta_portfolio.py',
    'jensens_alpha': 'estimate_jensens_alpha.py',
    'plot': 'plot_portfolio_frontier.py'
}


def run_script(script_name, args_list, step_num, total_steps):
    """
    Run a Python script with given arguments
    
    Parameters:
    -----------
    script_name : str
        Name of the script to run
    args_list : list
        List of command-line arguments
    step_num : int
        Current step number (for display)
    total_steps : int
        Total number of steps (for display)
    
    Returns:
    --------
    success : bool
        True if script ran successfully, False otherwise
    """
    print("=" * 70)
    print(f"Step {step_num}/{total_steps}: Running {script_name}")
    print("=" * 70)
    
    cmd = ['python', script_name] + args_list
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print()
        print(f"✓ Step {step_num}/{total_steps} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ Step {step_num}/{total_steps} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print()
        print(f"✗ Step {step_num}/{total_steps} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run all portfolio analysis scripts in the correct order'
    )
    parser.add_argument(
        '--portfolio-type', type=str, required=False,
        choices=['size', 'value'],
        help='Portfolio type: size or value (required unless --all is used)'
    )
    parser.add_argument(
        '--start-year', type=int, required=False,
        help='Start year (e.g., 1927) (required unless --all is used)'
    )
    parser.add_argument(
        '--end-year', type=int, required=False,
        help='End year (e.g., 2013) (required unless --all is used)'
    )
    parser.add_argument(
        '--rra', type=float, default=4.0,
        help='Relative Risk Aversion coefficient for optimal portfolio (default: 4.0)'
    )
    parser.add_argument(
        '--on-frontier', action='store_true',
        help='Constrain ZBP to be on the efficient frontier (default: False, just on IOS)'
    )
    parser.add_argument(
        '--closed-form', action='store_true',
        help='Use closed-form analytical solutions instead of optimization (default: False, use optimization)'
    )
    parser.add_argument(
        '--no-short-selling', action='store_true',
        help='Restrict portfolio weights to be non-negative (no short selling)'
    )
    parser.add_argument(
        '--skip-plot', action='store_true',
        help='Skip the final plotting step'
    )
    parser.add_argument(
        '--plot-alt-cones', action='store_true',
        help='Plot alternative comparison cones (cyan R_f↔MSMP and orange R_ZBP↔MSMP lines)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run for all combinations: size/value × 1927-2013/1927-2024'
    )
    
    args = parser.parse_args()
    
    # If --all flag, run for all combinations
    if args.all:
        print("=" * 70)
        print("Running All Portfolio Analyses for All Combinations")
        print("=" * 70)
        print()
        
        combinations = [
            ('size', 1927, 2013),
            ('size', 1927, 2024),
            ('value', 1927, 2013),
            ('value', 1927, 2024),
        ]
        
        total_combinations = len(combinations)
        for idx, (ptype, start, end) in enumerate(combinations, 1):
            print("\n" + "=" * 70)
            print(f"Combination {idx}/{total_combinations}: {ptype} portfolios, {start}-{end}")
            print("=" * 70)
            print()
            
            # Create new args for this combination
            combo_args = argparse.Namespace(
                portfolio_type=ptype,
                start_year=start,
                end_year=end,
                rra=args.rra,
                no_short_selling=args.no_short_selling,
                skip_plot=args.skip_plot,
                on_frontier=args.on_frontier,
                closed_form=args.closed_form,
                plot_alt_cones=args.plot_alt_cones,
                all=False
            )
            
            success = run_all_steps(combo_args)
            if not success:
                print(f"\n✗ Failed for combination {idx}/{total_combinations}")
                sys.exit(1)
        
        print("\n" + "=" * 70)
        print("✓ All combinations completed successfully!")
        print("=" * 70)
        return
    
    # Validate required arguments for single combination
    if args.portfolio_type is None or args.start_year is None or args.end_year is None:
        parser.error("--portfolio-type, --start-year, and --end-year are required unless --all is used")
    
    # Run for single combination
    print("=" * 70)
    print("Running All Portfolio Analyses")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"RRA: {args.rra}")
    print(f"Short selling: {'NOT ALLOWED' if args.no_short_selling else 'ALLOWED'}")
    print(f"ZBP on frontier: {'YES' if args.on_frontier else 'NO (on IOS only)'}")
    print(f"Method: {'CLOSED-FORM' if args.closed_form else 'OPTIMIZATION'}")
    print()
    
    success = run_all_steps(args)
    if not success:
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ All steps completed successfully!")
    print("=" * 70)


def run_all_steps(args):
    """
    Run all analysis steps for given arguments
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    success : bool
        True if all steps completed successfully
    """
    # Build common arguments
    common_args = [
        '--portfolio-type', args.portfolio_type,
        '--start-year', str(args.start_year),
        '--end-year', str(args.end_year),
    ]
    
    if args.no_short_selling:
        common_args.append('--no-short-selling')
    
    # Step 1: Construct Investment Opportunity Set (no plotting)
    step1_args = common_args.copy()
    if args.closed_form:
        step1_args.append('--closed-form')
    if not run_script(SCRIPTS['ios'], step1_args, 1, 6):
        return False
    
    # Step 2: Find MSMP Portfolio
    step2_args = common_args.copy()
    if args.closed_form:
        step2_args.append('--closed-form')
    if not run_script(SCRIPTS['msmp'], step2_args, 2, 6):
        return False
    
    # Step 3: Find Optimal CRRA Portfolio
    step3_args = common_args.copy() + ['--rra', str(args.rra)]
    if args.closed_form:
        step3_args.append('--closed-form')
    if not run_script(SCRIPTS['optimal_crra'], step3_args, 3, 6):
        return False
    
    # Step 4: Find Zero-Beta Portfolio (depends on step 3)
    step4_args = common_args.copy() + ['--rra', str(args.rra)]
    if args.on_frontier and not args.closed_form:
        step4_args.append('--on-frontier')
    if args.closed_form:
        step4_args.append('--closed-form')
    if not run_script(SCRIPTS['zbp'], step4_args, 4, 6):
        return False
    
    # Step 5: Estimate Jensen's Alpha (depends on steps 3 and 4)
    step5_args = common_args.copy() + ['--rra', str(args.rra)]
    if not run_script(SCRIPTS['jensens_alpha'], step5_args, 5, 6):
        return False
    
    # Step 6: Plot IOS and Key Portfolios (depends on steps 1, 2, 3, 4)
    if not args.skip_plot:
        step6_args = common_args.copy() + ['--rra', str(args.rra)]
        if args.closed_form:
            step6_args.append('--closed-form')
        if args.plot_alt_cones:
            step6_args.append('--plot-alt-cones')
        if not run_script(SCRIPTS['plot'], step6_args, 6, 6):
            return False
    else:
        print("\n" + "=" * 70)
        print("Step 6/6: Skipping plot (--skip-plot flag set)")
        print("=" * 70)
    
    return True


if __name__ == "__main__":
    main()

