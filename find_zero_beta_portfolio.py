#!/usr/bin/env python3
"""
Find Zero-Beta Portfolio (ZBP)

The Zero-Beta Portfolio has zero covariance with the optimal portfolio:
Cov(R_z, R_m) = w_z'Σw_m = 0

This script finds the ZBP for a given optimal portfolio (typically the optimal CRRA portfolio),
and saves the results including flags, expected return, and volatility.

The ZBP is expected to have a return close to 1 (gross return) from theory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import argparse

# Directories
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_portfolio_returns(portfolio_type, start_year, end_year):
    """
    Load portfolio returns from processed data
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year (e.g., 1927)
    end_year : int
        End year (e.g., 2013)
    
    Returns:
    --------
    returns : pd.DataFrame
        Portfolio returns (gross returns R = 1 + r)
        Index: dates, Columns: portfolio names
    """
    if portfolio_type.lower() == 'size':
        pattern = "*Portfolios_Formed_on_ME*Value_Weight_Returns___Annual*.csv"
    elif portfolio_type.lower() == 'value':
        pattern = "*Portfolios_Formed_on_BE-ME*Value_Weight_Returns___Annual*.csv"
    else:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")
    
    # Find matching file
    files = list(DATA_DIR.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No file found matching pattern: {pattern}")
    if len(files) > 1:
        raise ValueError(f"Multiple files found matching pattern: {pattern}")
    
    file_path = files[0]
    print(f"  Loading: {file_path.name}")
    
    # Load and filter by date range
    returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
    returns = returns.loc[f'{start_year}-01-01':f'{end_year}-01-01']
    
    # Remove portfolios with all NaN
    returns = returns.dropna(axis=1, how='all')
    
    # Drop rows with any NaN (complete cases only)
    returns = returns.dropna()
    
    print(f"  Loaded {len(returns)} observations, {len(returns.columns)} portfolios")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    return returns


def compute_moments(returns):
    """
    Compute mean returns and covariance matrix
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio returns (gross returns R = 1 + r)
        Index: dates, Columns: portfolio names
    
    Returns:
    --------
    mu : np.array
        Mean gross returns (E[R])
    Sigma : np.array
        Covariance matrix of gross returns (Cov(R))
    portfolio_names : list
        Portfolio names
    """
    # Filter out portfolios with zero variance (constant returns)
    portfolio_vars = returns.var()
    valid_portfolios = portfolio_vars[portfolio_vars > 1e-10].index.tolist()
    
    if len(valid_portfolios) < len(returns.columns):
        print(f"  Filtered out {len(returns.columns) - len(valid_portfolios)} portfolios with zero variance")
    
    returns_filtered = returns[valid_portfolios]
    
    # Compute moments
    mu = returns_filtered.mean().values  # Mean gross returns
    Sigma = returns_filtered.cov().values  # Covariance matrix of gross returns
    portfolio_names = valid_portfolios
    
    return mu, Sigma, portfolio_names


def load_optimal_portfolio_weights(portfolio_type, start_year, end_year, rra=4.0, allow_short_selling=True):
    """
    Load optimal CRRA portfolio weights from saved CSV
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    rra : float
        Relative Risk Aversion coefficient
    allow_short_selling : bool
        Whether short selling was allowed
    
    Returns:
    --------
    w_opt : np.array
        Optimal portfolio weights
    portfolio_names : list
        Portfolio names (must match order of weights)
    """
    suffix = f"_{portfolio_type}_{start_year}_{end_year}"
    weights_file = RESULTS_DIR / f"optimal_crra_weights{suffix}.csv"
    
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Optimal CRRA weights file not found: {weights_file}\n"
            f"Please run find_optimal_crra_portfolio.py first with matching parameters."
        )
    
    print(f"  Loading optimal portfolio weights from: {weights_file.name}")
    weights_df = pd.read_csv(weights_file)
    
    # Extract portfolio names and weights
    portfolio_names = weights_df['portfolio'].tolist()
    w_opt = weights_df['weight'].values
    
    return w_opt, portfolio_names


def find_zero_beta_portfolio(mu, Sigma, w_m, allow_short_selling=True, on_frontier=True):
    """
    Find Zero-Beta Portfolio (ZBP)
    
    ZBP satisfies: Cov(R_z, R_m) = w_z'Σw_m = 0
    
    If on_frontier=True, the ZBP is constrained to lie on the efficient frontier
    (minimum variance for its return level). The ZBP should be on the inefficient limb.
    
    Parameters:
    -----------
    mu : np.array
        Mean gross returns (E[R])
    Sigma : np.array
        Covariance matrix of gross returns (Cov(R))
    w_m : np.array
        Optimal portfolio weights (market portfolio)
    allow_short_selling : bool
        If True, allows negative weights (free portfolio)
    on_frontier : bool
        If True, find ZBP on the efficient frontier
    
    Returns:
    --------
    w_z : np.array
        ZBP portfolio weights
    z_return : float
        ZBP expected gross return (E[w_z'R])
    z_vol : float
        ZBP volatility (std of w_z'R)
    """
    n = len(mu)
    
    # Find MVP for reference
    ones = np.ones(n)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.linalg.pinv(Sigma)
    
    w_mvp = inv_Sigma @ ones / (ones.T @ inv_Sigma @ ones)
    mu_mvp = mu.T @ w_mvp
    mu_opt = mu.T @ w_m
    
    # Get range of returns for searching opposite limb
    mu_min = mu.min()  # Minimum individual asset return
    mu_max = mu.max()  # Maximum individual asset return
    
    if on_frontier:
        # Search along the frontier for ZBP
        # ZBP should be on the OPPOSITE limb from the optimal portfolio
        # If optimal portfolio is on efficient limb (above MVP), ZBP should be on inefficient limb (below MVP)
        # If optimal portfolio is on inefficient limb (below MVP), ZBP should be on efficient limb (above MVP)
        
        is_optimal_efficient = mu_opt >= mu_mvp
        
        if is_optimal_efficient:
            # Optimal is on efficient limb, so ZBP should be on inefficient limb (below MVP)
            # Search from well below MVP down to minimum asset return (or even lower)
            search_max = mu_mvp * 0.999  # Just below MVP
            search_min = min(mu_min, mu_mvp * 0.80)  # At least 20% below MVP, or minimum asset return if lower
            limb_name = "inefficient"
        else:
            # Optimal is on inefficient limb, so ZBP should be on efficient limb (above MVP)
            # Search from just above MVP up to maximum asset return
            search_min = mu_mvp * 1.001  # Just above MVP
            search_max = mu_max  # Up to maximum asset return
            limb_name = "efficient"
        
        # Generate target returns
        num_targets = 200
        target_returns = np.linspace(search_min, search_max, num_targets)
        
        best_result = None
        best_cov = np.inf
        
        print(f"  Searching for ZBP on frontier ({limb_name} limb)...")
        print(f"  Optimal portfolio is on {'efficient' if is_optimal_efficient else 'inefficient'} limb (return: {mu_opt:.6f})")
        print(f"  MVP return: {mu_mvp:.6f}")
        print(f"  Search range: [{search_min:.6f}, {search_max:.6f}] (gross returns)")
        
        for i, z_return_target in enumerate(target_returns):
            # Minimize variance subject to:
            # 1. Budget constraint: sum(w) = 1
            # 2. Return constraint: mu'w = z_return_target (ensures on frontier)
            # 3. Zero-beta constraint: w'Σw_m = 0
            
            def objective(w):
                return w.T @ Sigma @ w
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: mu.T @ w - z_return_target},
                {'type': 'eq', 'fun': lambda w: w.T @ Sigma @ w_m}
            ]
            
            # Set bounds
            if allow_short_selling:
                bounds = [(-1, 2) for _ in range(n)]
            else:
                bounds = [(0, 1) for _ in range(n)]
            
            # Initial guesses
            initial_guesses = []
            if z_return_target < mu_mvp:
                # Inefficient part: try negative MVP (scaled and normalized)
                w_neg = -w_mvp.copy()
                if np.abs(np.sum(w_neg)) > 1e-10:
                    w_neg = w_neg / np.sum(w_neg)
                    if np.all(w_neg >= -1) and np.all(w_neg <= 2):
                        initial_guesses.append(w_neg)
                # Try scaled MVP
                w_scaled = w_mvp * 0.5
                w_scaled = w_scaled / np.sum(w_scaled) if np.abs(np.sum(w_scaled)) > 1e-10 else np.ones(n) / n
                initial_guesses.append(w_scaled)
            else:
                # Efficient part: start from MVP
                initial_guesses.append(w_mvp.copy())
            
            # Always include equal weights
            initial_guesses.append(np.ones(n) / n)
            
            for w0 in initial_guesses:
                try:
                    result = minimize(objective, w0, method='SLSQP',
                                     bounds=bounds, constraints=constraints,
                                     options={'maxiter': 2000, 'ftol': 1e-9})
                    
                    if result.success:
                        budget_check = abs(np.sum(result.x) - 1)
                        return_check = abs(mu.T @ result.x - z_return_target)
                        cov_check = abs(result.x.T @ Sigma @ w_m)
                        
                        if budget_check < 1e-5 and return_check < 1e-5 and cov_check < 1e-3:
                            if cov_check < best_cov:
                                best_result = result
                                best_cov = cov_check
                                break  # Found good solution for this target
                except:
                    continue
            
            if best_result is not None and best_cov < 1e-5:
                break  # Found very good solution, stop searching
        
        if best_result is None:
            raise ValueError("Failed to find zero-β portfolio on frontier. "
                           "Try relaxing constraints or checking if optimization is feasible.")
        
        w_z = best_result.x
        z_return = mu.T @ w_z
        z_vol = np.sqrt(w_z.T @ Sigma @ w_z)
        
        # Verify constraints
        zero_beta_check = abs(w_z.T @ Sigma @ w_m)
        if zero_beta_check > 1e-4:
            print(f"  Warning: Zero-beta constraint not perfectly satisfied (error: {zero_beta_check:.2e})")
        
        return w_z, z_return, z_vol
    
    else:
        # ZBP on IOS only - use closed-form formula
        # Formula: z = (1 - (v'Σw)/(v'Σw - w'Σw)) v + ((v'Σw)/(v'Σw - w'Σw)) w
        # where v = MVP, w = optimal portfolio (CRRA), z = ZBP
        print(f"  Computing ZBP using closed-form formula...")
        print(f"  MVP return: {mu_mvp:.6f}, Optimal return: {mu_opt:.6f}")
        
        # Compute terms needed for the formula
        # v'Σw = covariance between MVP and optimal portfolio
        v_Sigma_w = w_mvp.T @ Sigma @ w_m
        
        # w'Σw = variance of optimal portfolio
        w_Sigma_w = w_m.T @ Sigma @ w_m
        
        # Denominator: v'Σw - w'Σw
        denominator = v_Sigma_w - w_Sigma_w
        
        if abs(denominator) < 1e-10:
            raise ValueError(f"Denominator too small ({denominator:.2e}). "
                           "MVP and optimal portfolio may be too similar.")
        
        # Coefficient for MVP: 1 - (v'Σw)/(v'Σw - w'Σw)
        alpha = 1 - (v_Sigma_w / denominator)
        
        # Coefficient for optimal portfolio: (v'Σw)/(v'Σw - w'Σw)
        beta = v_Sigma_w / denominator
        
        # ZBP weights: z = alpha * v + beta * w
        w_z = alpha * w_mvp + beta * w_m
        
        # Verify budget constraint (should sum to 1)
        budget_sum = np.sum(w_z)
        if abs(budget_sum - 1) > 1e-10:
            # Normalize if needed (shouldn't be necessary, but just in case)
            print(f"  Warning: ZBP weights sum to {budget_sum:.10f}, normalizing...")
            w_z = w_z / budget_sum
        
        # Compute ZBP return and volatility
        z_return = mu.T @ w_z
        z_vol = np.sqrt(w_z.T @ Sigma @ w_z)
        
        # Verify zero-beta constraint (should be exactly 0)
        zero_beta_check = abs(w_z.T @ Sigma @ w_m)
        
        # Verify budget constraint
        budget_check = abs(np.sum(w_z) - 1)
        
        print(f"  Formula coefficients: alpha (MVP) = {alpha:.6f}, beta (Optimal) = {beta:.6f}")
        print(f"  Zero-beta check (should be ~0): {zero_beta_check:.2e}")
        print(f"  Budget constraint check (should be ~0): {budget_check:.2e}")
        
        if zero_beta_check > 1e-10:
            print(f"  Warning: Zero-beta constraint error ({zero_beta_check:.2e}) is larger than expected.")
            print(f"  This may indicate numerical precision issues.")
        
        # Check if on inefficient limb
        if z_return < mu_mvp:
            print(f"  ZBP is on inefficient limb (return {z_return:.6f} < MVP {mu_mvp:.6f})")
        else:
            print(f"  ZBP is on efficient limb (return {z_return:.6f} >= MVP {mu_mvp:.6f})")
        
        return w_z, z_return, z_vol


def main():
    parser = argparse.ArgumentParser(
        description='Find Zero-Beta Portfolio (ZBP)'
    )
    parser.add_argument(
        '--portfolio-type', type=str, required=True,
        choices=['size', 'value'],
        help='Portfolio type: size or value'
    )
    parser.add_argument(
        '--start-year', type=int, required=True,
        help='Start year (e.g., 1927)'
    )
    parser.add_argument(
        '--end-year', type=int, required=True,
        help='End year (e.g., 2013)'
    )
    parser.add_argument(
        '--rra', type=float, default=4.0,
        help='Relative Risk Aversion coefficient for optimal portfolio (default: 4.0)'
    )
    parser.add_argument(
        '--output-suffix', type=str, default='',
        help='Suffix to add to output filename (e.g., "_test")'
    )
    parser.add_argument(
        '--on-frontier', action='store_true',
        help='Constrain ZBP to be on the frontier (hyperbola boundary). Default: True when using optimization, False when using --closed-form'
    )
    parser.add_argument(
        '--closed-form', action='store_true',
        help='Use closed-form analytical solution (forces on_frontier=False, uses closed-form formula)'
    )
    parser.add_argument(
        '--print-only', action='store_true',
        help='Only print the result, do not save to files'
    )
    parser.add_argument(
        '--no-short-selling', action='store_true',
        help='Restrict portfolio weights to be non-negative (no short selling)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Finding Zero-Beta Portfolio (ZBP)")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"Optimal portfolio RRA: {args.rra}")
    
    allow_short = not args.no_short_selling
    # When using optimization (non-closed-form), default to on_frontier=True
    # to ensure ZBP is on the frontier (hyperbola boundary)
    # If --closed-form is set, force on_frontier=False and use closed-form formula
    if args.closed_form:
        on_frontier = False
        print(f"Short selling: {'ALLOWED (free portfolio)' if allow_short else 'NOT ALLOWED (long-only)'}")
        print(f"Closed-form: YES (using analytical formula, on_frontier=False)")
    else:
        # Default to on_frontier=True when using optimization to ensure ZBP is on frontier
        on_frontier = args.on_frontier if args.on_frontier else True
        print(f"Short selling: {'ALLOWED (free portfolio)' if allow_short else 'NOT ALLOWED (long-only)'}")
        print(f"On frontier: {'YES' if on_frontier else 'NO (on IOS only)'}")
    print()
    
    # Load data
    print("Loading portfolio returns...")
    returns = load_portfolio_returns(args.portfolio_type, args.start_year, args.end_year)
    
    # Compute moments
    print("\nComputing moments...")
    mu, Sigma, portfolio_names = compute_moments(returns)
    print(f"  Mean returns range: [{mu.min():.4f}, {mu.max():.4f}] (gross)")
    print(f"  Volatility range: [{np.sqrt(np.diag(Sigma)).min():.4f}, {np.sqrt(np.diag(Sigma)).max():.4f}]")
    
    # Load optimal portfolio weights
    print(f"\nLoading optimal CRRA portfolio (RRA={args.rra})...")
    try:
        w_opt, opt_portfolio_names = load_optimal_portfolio_weights(
            args.portfolio_type, args.start_year, args.end_year, args.rra, allow_short
        )
        
        # Ensure portfolio names match
        if opt_portfolio_names != portfolio_names:
            print(f"  Warning: Portfolio name mismatch. Reordering weights...")
            # Reorder weights to match current portfolio order
            opt_df = pd.DataFrame({'portfolio': opt_portfolio_names, 'weight': w_opt})
            opt_df = opt_df.set_index('portfolio').reindex(portfolio_names)
            w_opt = opt_df['weight'].values
            # Handle any missing portfolios
            w_opt = np.nan_to_num(w_opt, nan=0.0)
            w_opt = w_opt / np.sum(w_opt)  # Renormalize
        
        opt_return = mu.T @ w_opt
        opt_vol = np.sqrt(w_opt.T @ Sigma @ w_opt)
        print(f"  Optimal portfolio return (gross): {opt_return:.6f}")
        print(f"  Optimal portfolio volatility: {opt_vol:.4%}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return
    
    # Find Zero-Beta Portfolio
    print(f"\nFinding Zero-Beta Portfolio...")
    if on_frontier:
        print(f"  Constraint: On efficient frontier (minimum variance for return level)")
    else:
        print(f"  Constraint: On Investment Opportunity Set (IOS) only")
    
    w_z, z_return, z_vol = find_zero_beta_portfolio(
        mu, Sigma, w_opt, allow_short_selling=allow_short, on_frontier=on_frontier
    )
    
    # Convert gross returns to net returns for display
    z_return_net = z_return - 1
    
    print(f"  ZBP return (gross): {z_return:.6f}")
    print(f"  ZBP return (net): {z_return_net:.4%}")
    print(f"  ZBP volatility: {z_vol:.4%}")
    
    # Verify zero-beta constraint
    zero_beta_check = abs(w_z.T @ Sigma @ w_opt)
    print(f"  Zero-beta check (should be ~0): {zero_beta_check:.2e}")
    
    # Verify budget constraint
    budget_check = abs(np.sum(w_z) - 1)
    print(f"  Budget constraint check (should be ~0): {budget_check:.2e}")
    
    # Print result
    print("\n" + "=" * 70)
    print("ZBP Result (on IOS, not necessarily on frontier):")
    print("=" * 70)
    print(f"Return (gross): {z_return:.6f}")
    print(f"Return (net): {z_return_net:.4%}")
    print(f"Volatility: {z_vol:.4%}")
    print(f"Zero-beta error: {zero_beta_check:.2e}")
    print(f"Budget constraint error: {budget_check:.2e}")
    print("=" * 70)
    
    if args.print_only:
        print("\n(Print-only mode: results not saved)")
        return
    
    # Save results
    print("\nSaving results...")
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    
    # Save weights
    weights_df = pd.DataFrame({
        'portfolio': portfolio_names,
        'weight': w_z
    })
    weights_df.to_csv(RESULTS_DIR / f"zbp_weights{suffix}.csv", index=False)
    print(f"  Saved: zbp_weights{suffix}.csv")
    
    # Save summary with flags, expected return, and volatility
    summary_df = pd.DataFrame([{
        'portfolio_type': args.portfolio_type,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'rra': args.rra,
        'allow_short_selling': allow_short,
        'expected_return_gross': z_return,
        'expected_return_net': z_return_net,
        'volatility': z_vol,
        'zero_beta_error': zero_beta_check,
        'num_portfolios': len(portfolio_names),
        'num_observations': len(returns)
    }])
    
    # Check if summary file exists, append if it does
    summary_file = RESULTS_DIR / "zbp_summary.csv"
    if summary_file.exists():
        existing_df = pd.read_csv(summary_file)
        # Drop 'on_frontier' column if it exists (backward compatibility)
        if 'on_frontier' in existing_df.columns:
            existing_df = existing_df.drop(columns=['on_frontier'])
        # Check if this combination already exists
        mask = (
            (existing_df['portfolio_type'] == args.portfolio_type) &
            (existing_df['start_year'] == args.start_year) &
            (existing_df['end_year'] == args.end_year) &
            (existing_df['rra'] == args.rra) &
            (existing_df['allow_short_selling'] == allow_short)
        )
        if mask.any():
            # Remove all matching rows (in case of duplicates from old on_frontier column)
            existing_df = existing_df[~mask]
            # Append new row
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            print(f"  Updated existing entry in: zbp_summary.csv")
        else:
            # Append new row
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            print(f"  Appended to: zbp_summary.csv")
    else:
        print(f"  Created: zbp_summary.csv")
    
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"ZBP return (gross): {z_return:.6f} (expected to be close to 1.0)")
    print(f"ZBP return (net): {z_return_net:.4%}")
    print(f"ZBP volatility: {z_vol:.4%}")
    print(f"Zero-beta constraint error: {zero_beta_check:.2e} (should be < 1e-4)")
    print(f"Number of portfolios: {len(portfolio_names)}")
    print(f"Number of observations: {len(returns)}")
    print("\n✓ Zero-Beta Portfolio calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

