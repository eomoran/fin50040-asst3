#!/usr/bin/env python3
"""
Construct the Investment Opportunity Set (Mean-Variance Frontier)

This script constructs the full investment opportunity set, which includes
both the efficient and inefficient limbs of the mean-variance frontier.
The frontier forms a U-shape when plotted (volatility vs return).

The investment opportunity set consists of all portfolios that minimize
variance for a given expected return, subject to the budget constraint
(sum of weights = 1). No riskless asset is included.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Directories
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


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
        raise ValueError(f"Unknown portfolio type: {portfolio_type}. Use 'size' or 'value'")
    
    # Find matching file
    files = list(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No portfolio file found matching: {pattern}")
    if len(files) > 1:
        # Prefer the annual value-weighted file
        files = [f for f in files if "Annual" in f.name]
    if not files:
        raise FileNotFoundError(f"Multiple files found, but none with 'Annual' in name")
    
    file_path = files[0]
    print(f"  Loading: {file_path.name}")
    
    # Load data
    returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Filter by date range
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    returns = returns.loc[start_date:end_date]
    
    # Remove portfolios with all NaN
    returns = returns.dropna(axis=1, how='all')
    
    # Remove rows with any NaN (for clean covariance calculation)
    returns = returns.dropna(axis=0, how='any')
    
    print(f"  Loaded {len(returns)} observations, {len(returns.columns)} portfolios")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    return returns


def compute_moments(returns):
    """
    Compute mean and covariance matrix of gross returns
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio gross returns (R = 1 + r)
    
    Returns:
    --------
    mu : np.array
        Mean gross returns (E[R])
    Sigma : np.array
        Covariance matrix of gross returns (Cov(R))
    portfolio_names : list
        Portfolio names
    """
    # Convert to numpy arrays
    R = returns.values  # Gross returns
    
    # Mean gross returns
    mu = np.mean(R, axis=0)
    
    # Covariance matrix of gross returns
    Sigma = np.cov(R, rowvar=False)
    
    portfolio_names = returns.columns.tolist()
    
    return mu, Sigma, portfolio_names


def optimize_single_portfolio(args_tuple):
    """
    Optimize a single portfolio for a given target return.
    This function is designed to be called in parallel.
    
    Parameters:
    -----------
    args_tuple : tuple
        (target_return, mu, Sigma, w_mvp, mu_mvp, allow_short_selling, n)
    
    Returns:
    --------
    result : dict or None
        Dictionary with 'return', 'vol', 'weights' if successful, None otherwise
    """
    target_return, mu, Sigma, w_mvp, mu_mvp, allow_short_selling, n = args_tuple
    
    def objective(w):
        return w.T @ Sigma @ w
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: mu.T @ w - target_return}
    ]
    
    if allow_short_selling:
        bounds = [(-1, 2) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]
    
    initial_guesses = []
    if target_return < mu_mvp:
        high_return_indices = np.argsort(mu)[-3:]
        low_return_indices = np.argsort(mu)[:3]
        
        for short_frac in [0.3, 0.5, 0.7]:
            w_ineff = np.zeros(n)
            for idx in high_return_indices:
                w_ineff[idx] = -short_frac / len(high_return_indices)
            long_weight = 1 - np.sum(w_ineff)
            for idx in low_return_indices:
                w_ineff[idx] += long_weight / len(low_return_indices)
            
            if np.all(w_ineff >= -1) and np.all(w_ineff <= 2):
                initial_guesses.append(w_ineff)
        
        w_neg = -w_mvp.copy()
        if np.abs(np.sum(w_neg)) > 1e-10:
            w_neg = w_neg / np.sum(w_neg)
            if np.all(w_neg >= -1) and np.all(w_neg <= 2):
                initial_guesses.append(w_neg)
        
        high_return_idx = np.argmax(mu)
        w_short = np.zeros(n)
        w_short[high_return_idx] = -0.8
        low_return_idx = np.argmin(mu)
        w_short[low_return_idx] = 1.8
        if np.all(w_short >= -1) and np.all(w_short <= 2):
            initial_guesses.append(w_short)
    else:
        initial_guesses.append(w_mvp.copy())
    
    initial_guesses.append(np.ones(n) / n)
    
    best_result = None
    best_variance = np.inf
    
    for w0 in initial_guesses:
        try:
            result = minimize(objective, w0, method='SLSQP',
                             bounds=bounds, constraints=constraints,
                             options={'maxiter': 2000, 'ftol': 1e-9})
            
            if not result.success or abs(mu.T @ result.x - target_return) > 1e-4:
                try:
                    result = minimize(objective, w0, method='trust-constr',
                                     bounds=bounds, constraints=constraints,
                                     options={'maxiter': 2000})
                except:
                    continue
            
            if result.success:
                budget_check = abs(np.sum(result.x) - 1)
                return_check = abs(mu.T @ result.x - target_return)
                
                if budget_check < 1e-5 and return_check < 1e-5:
                    if result.fun < best_variance:
                        best_result = result
                        best_variance = result.fun
        except:
            continue
    
    if best_result is not None:
        w = best_result.x
        portfolio_return = mu.T @ w
        portfolio_vol = np.sqrt(w.T @ Sigma @ w)
        return {
            'return': portfolio_return,
            'vol': portfolio_vol,
            'weights': w
        }
    
    return None


def construct_investment_opportunity_set(mu, Sigma, num_portfolios=200, allow_short_selling=True, n_jobs=None, closed_form=False):
    """
    Construct the Investment Opportunity Set (mean-variance frontier)
    
    This includes BOTH the efficient and inefficient limbs, forming a U-shape.
    For each target return, we find the minimum variance portfolio.
    
    Parameters:
    -----------
    mu : np.array
        Mean gross returns (E[R])
    Sigma : np.array
        Covariance matrix of gross returns (Cov(R))
    num_portfolios : int
        Number of portfolios on the frontier
    
    Returns:
    --------
    frontier_data : dict
        Dictionary with 'returns', 'volatilities', 'weights' for the frontier
    """
    n = len(mu)
    
    # Find Minimum Variance Portfolio (MVP)
    ones = np.ones(n)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.linalg.pinv(Sigma)
    
    w_mvp = inv_Sigma @ ones / (ones.T @ inv_Sigma @ ones)
    mu_mvp = mu.T @ w_mvp
    
    if closed_form:
        # Closed-form solution using hyperbola formula from professor's notes
        # σ_p = sqrt( (C(μ_p)^2 - 2Bμ_p + A) / (AC - B^2) )
        # where:
        # A = μ^T Σ^-1 μ
        # B = μ^T Σ^-1 1_N
        # C = 1_N^T Σ^-1 1_N
        
        print(f"  Using closed-form hyperbola formula for IOS...")
        
        # Compute constants A, B, C
        A = mu.T @ inv_Sigma @ mu
        B = mu.T @ inv_Sigma @ ones
        C = ones.T @ inv_Sigma @ ones
        D = A * C - B * B  # AC - B^2
        
        print(f"  Constants: A = {A:.6f}, B = {B:.6f}, C = {C:.6f}, AC-B^2 = {D:.6f}")
        
        if D <= 0:
            raise ValueError(f"AC - B^2 = {D:.6f} <= 0, cannot construct frontier")
        
        # Find range of returns
        mu_min = mu.min()  # Minimum individual asset return
        mu_max = mu.max()  # Maximum individual asset return
        
        # Extend range for inefficient limb (below MVP)
        # Need to extend far enough to capture MSMP which is typically well below MVP
        range_below_mvp = max(
            abs(mu_mvp - mu_min),
            mu_mvp * 0.20  # Extend at least 20% below MVP to capture MSMP
        )
        search_min = mu_mvp - range_below_mvp * 2.0  # Extend 2x the range below MVP
        search_min = max(search_min, mu_mvp * 0.70)  # At least 30% below MVP
        
        # Generate target returns
        num_inefficient = int(num_portfolios * 0.5)
        num_efficient = num_portfolios - num_inefficient
        
        target_returns_inefficient = np.linspace(search_min, mu_mvp, num_inefficient + 1)[:-1]
        target_returns_efficient = np.linspace(mu_mvp, mu_max, num_efficient)
        target_returns = np.concatenate([target_returns_inefficient, target_returns_efficient])
        
        frontier_returns = []
        frontier_vols = []
        frontier_weights = []
        
        print(f"  Computing {len(target_returns)} frontier points using closed-form formula...")
        
        for mu_p in target_returns:
            # Compute volatility using hyperbola formula
            numerator = C * mu_p * mu_p - 2 * B * mu_p + A
            if numerator < 0:
                # Skip if negative (outside feasible region)
                continue
            
            sigma_p = np.sqrt(numerator / D)
            
            # Compute portfolio weights using formula from professor's notes:
            # w = (Σ^-1 (λμ + δ1_N)) / (1_N^T Σ^-1 (λμ + δ1_N))
            # where:
            # λ = (Cμ_p - B) / (AC - B^2)
            # δ = (A - Bμ_p) / (AC - B^2)
            lambda_val = (C * mu_p - B) / D
            delta_val = (A - B * mu_p) / D
            
            w = inv_Sigma @ (lambda_val * mu + delta_val * ones)
            w = w / (ones.T @ inv_Sigma @ (lambda_val * mu + delta_val * ones))
            
            frontier_returns.append(mu_p)
            frontier_vols.append(sigma_p)
            frontier_weights.append(w)
        
        print(f"  Successfully computed {len(frontier_returns)} frontier points using closed-form formula")
        
        return {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_vols),
            'weights': np.array(frontier_weights)
        }
    
    # Find range of returns
    mu_min = mu.min()  # Minimum individual asset return
    mu_max = mu.max()  # Maximum individual asset return
    
    # For the inefficient limb, we need to extend below MVP
    # The inefficient limb has higher volatility for the same or lower return
    # We'll search from well below MVP return to capture the full U-shape
    # Note: mu_mvp might be above or below mu_min depending on the data
    # We want to search BELOW MVP, so extend below mu_mvp
    # Use a percentage-based approach to ensure we extend far enough
    # Extend at least 20% below MVP to capture MSMP which is typically on inefficient limb
    range_below_mvp = max(
        abs(mu_mvp - mu_min),  # Distance between MVP and min asset return
        mu_mvp * 0.20  # At least 20% below MVP
    )
    search_min = mu_mvp - range_below_mvp * 2.0  # Extend 2x the range below MVP
    
    # Ensure search_min is reasonable (not negative or too small)
    search_min = max(search_min, mu_mvp * 0.70)  # At least 30% below MVP
    
    # Generate target returns covering both limbs
    # Inefficient limb: from search_min to MVP (below MVP)
    # Efficient limb: from MVP to mu_max (above MVP)
    num_inefficient = int(num_portfolios * 0.5)  # 50% for inefficient part
    num_efficient = num_portfolios - num_inefficient  # 50% for efficient part
    
    target_returns_inefficient = np.linspace(search_min, mu_mvp, num_inefficient + 1)[:-1]  # Exclude MVP duplicate
    target_returns_efficient = np.linspace(mu_mvp, mu_max, num_efficient)
    target_returns = np.concatenate([target_returns_inefficient, target_returns_efficient])
    
    frontier_returns = []
    frontier_vols = []
    frontier_weights = []
    
    print(f"  Constructing investment opportunity set with {len(target_returns)} target returns...")
    print(f"  MVP return (gross): {mu_mvp:.6f} (net: {mu_mvp - 1:.4%})")
    
    # Determine number of parallel workers
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    print(f"  Using {n_jobs} parallel workers...")
    
    # Prepare arguments for parallel processing
    args_list = [(tr, mu, Sigma, w_mvp, mu_mvp, allow_short_selling, n) 
                 for tr in target_returns]
    
    frontier_returns = []
    frontier_vols = []
    frontier_weights = []
    
    # Process in parallel
    successful = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_return = {executor.submit(optimize_single_portfolio, args): args[0] 
                           for args in args_list}
        
        completed = 0
        for future in as_completed(future_to_return):
            completed += 1
            result = future.result()
            
            if result is not None:
                frontier_returns.append(result['return'])
                frontier_vols.append(result['vol'])
                frontier_weights.append(result['weights'])
                successful += 1
            
            if completed % 20 == 0:
                print(f"  Progress: {completed}/{len(target_returns)} portfolios optimized ({successful} successful)")
    
    print(f"  Successfully constructed {successful}/{len(target_returns)} frontier points")
    
    if len(frontier_returns) == 0:
        raise ValueError("Failed to generate any frontier points. Check optimization constraints.")
    
    return {
        'returns': np.array(frontier_returns),
        'volatilities': np.array(frontier_vols),
        'weights': np.array(frontier_weights)
    }


def plot_investment_opportunity_set(frontier, mu, Sigma, portfolio_names, 
                                    portfolio_type, start_year, end_year,
                                    figsize=(12, 8)):
    """
    Plot the Investment Opportunity Set (mean-variance frontier)
    
    Parameters:
    -----------
    frontier : dict
        Dictionary with 'returns', 'volatilities', 'weights'
    mu : np.array
        Mean gross returns
    Sigma : np.array
        Covariance matrix
    portfolio_names : list
        Portfolio names
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    figsize : tuple
        Figure size
    """
    # Use gross returns throughout (R = 1 + r) for plotting
    frontier_returns = frontier['returns']  # Gross returns
    frontier_vols = frontier['volatilities']
    
    # Find MVP (minimum volatility point)
    mvp_idx = np.argmin(frontier_vols)
    mvp_return = frontier_returns[mvp_idx]  # Gross return
    mvp_vol = frontier_vols[mvp_idx]
    
    # Separate efficient and inefficient limbs
    # Inefficient limb: return < MVP return AND volatility > MVP volatility
    # Efficient limb: return >= MVP return OR volatility <= MVP volatility (but not both <)
    inefficient_mask = (frontier_returns < mvp_return) & (frontier_vols > mvp_vol)
    efficient_mask = ~inefficient_mask
    
    # Sort each limb separately for proper plotting
    # Inefficient limb: sort by volatility DESCENDING (from highest vol down to MVP)
    # This ensures smooth connection to MVP
    inefficient_vols = frontier_vols[inefficient_mask]
    inefficient_returns = frontier_returns[inefficient_mask]
    if len(inefficient_vols) > 0:
        inefficient_sort = np.argsort(inefficient_vols)[::-1]  # Descending order
        inefficient_vols_sorted = inefficient_vols[inefficient_sort]
        inefficient_returns_sorted = inefficient_returns[inefficient_sort]
    else:
        inefficient_vols_sorted = np.array([])
        inefficient_returns_sorted = np.array([])
    
    # Efficient limb: sort by volatility (ascending, from MVP up)
    efficient_vols = frontier_vols[efficient_mask]
    efficient_returns = frontier_returns[efficient_mask]
    efficient_sort = np.argsort(efficient_vols)
    efficient_vols_sorted = efficient_vols[efficient_sort]
    efficient_returns_sorted = efficient_returns[efficient_sort]
    
    # Combine: inefficient limb (descending vol) -> MVP -> efficient limb (ascending vol)
    # This creates the full U-shape with smooth connections
    frontier_vols_sorted = np.concatenate([inefficient_vols_sorted, [mvp_vol], efficient_vols_sorted])
    frontier_returns_sorted = np.concatenate([inefficient_returns_sorted, [mvp_return], efficient_returns_sorted])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot investment opportunity set (full U-shape) - using gross returns
    ax.plot(frontier_vols_sorted, frontier_returns_sorted, 
            'b-', linewidth=2, label='Investment Opportunity Set', alpha=0.7, zorder=2)
    
    # Plot MVP - using gross returns
    ax.scatter([mvp_vol], [mvp_return],
              c='purple', s=200, marker='^', edgecolors='black', linewidths=1.5,
              label=f'MVP (R={mvp_return:.4f}, σ={mvp_vol:.2%})', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('Volatility (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (R)', fontsize=12, fontweight='bold')
    
    # Title
    title = f'Investment Opportunity Set\n{portfolio_type.upper()} Portfolios ({start_year}-{end_year})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Format axes: x-axis as percentages, y-axis as decimals (gross returns)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    # Y-axis: gross returns (R), format as decimals (e.g., 1.10 for 10% return)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Adjust limits: x and y mins fixed at zero, maxes with padding
    min_vol = 0.0  # Fixed at zero
    max_vol = frontier_vols_sorted.max() * 1.1
    min_ret = 0.0  # Fixed at zero
    max_ret = frontier_returns_sorted.max() * 1.1
    
    ax.set_xlim(min_vol, max_vol)
    ax.set_ylim(min_ret, max_ret)
    
    plt.tight_layout()
    
    # Save figure
    suffix = f"_{portfolio_type}_{start_year}_{end_year}"
    filename = PLOTS_DIR / f'investment_opportunity_set{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {filename}")
    
    # Also save as PDF for high quality
    filename_pdf = PLOTS_DIR / f'investment_opportunity_set{suffix}.pdf'
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"  Saved plot to {filename_pdf}")
    
    plt.close(fig)  # Close the figure to free memory


def main():
    parser = argparse.ArgumentParser(
        description='Construct Investment Opportunity Set (Mean-Variance Frontier)'
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
        '--num-portfolios', type=int, default=200,
        help='Number of portfolios on the frontier (default: 200)'
    )
    parser.add_argument(
        '--output-suffix', type=str, default='',
        help='Suffix to add to output filename (e.g., "_test")'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate and save plot of the investment opportunity set'
    )
    parser.add_argument(
        '--no-short-selling', action='store_true',
        help='Restrict weights to be >= 0 (no short selling)'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--closed-form', action='store_true',
        help='Use closed-form analytical solution instead of optimization'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Constructing Investment Opportunity Set")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"Number of frontier points: {args.num_portfolios}")
    print(f"Method: {'CLOSED-FORM' if args.closed_form else 'OPTIMIZATION'}")
    print()
    
    # Load data
    print("Loading portfolio returns...")
    returns = load_portfolio_returns(args.portfolio_type, args.start_year, args.end_year)
    
    # Compute moments
    print("\nComputing moments...")
    mu, Sigma, portfolio_names = compute_moments(returns)
    print(f"  Mean returns range: [{mu.min():.4f}, {mu.max():.4f}] (gross)")
    print(f"  Volatility range: [{np.sqrt(np.diag(Sigma)).min():.4f}, {np.sqrt(np.diag(Sigma)).max():.4f}]")
    
    # Construct investment opportunity set
    print("\nConstructing investment opportunity set...")
    allow_short = not args.no_short_selling
    if allow_short:
        print("  Short selling: ALLOWED (free portfolio)")
    else:
        print("  Short selling: NOT ALLOWED (long-only)")
    frontier = construct_investment_opportunity_set(mu, Sigma, num_portfolios=args.num_portfolios, 
                                                     allow_short_selling=allow_short, n_jobs=args.n_jobs,
                                                     closed_form=args.closed_form)
    
    # Save results
    print("\nSaving results...")
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    
    # Save frontier data (gross returns)
    frontier_df = pd.DataFrame({
        'return_gross': frontier['returns'],
        'return_net': frontier['returns'] - 1,  # Keep net for reference
        'volatility': frontier['volatilities']
    })
    frontier_df.to_csv(RESULTS_DIR / f"investment_opportunity_set{suffix}.csv", index=False)
    print(f"  Saved: investment_opportunity_set{suffix}.csv")
    
    # Save weights
    weights_df = pd.DataFrame(
        frontier['weights'],
        columns=portfolio_names
    )
    weights_df.to_csv(RESULTS_DIR / f"ios_weights{suffix}.csv", index=False)
    print(f"  Saved: ios_weights{suffix}.csv")
    
    # Find MVP for summary
    mvp_idx = np.argmin(frontier['volatilities'])
    mvp_return_gross = frontier_df.iloc[mvp_idx]['return_gross']
    mvp_return_net = frontier_df.iloc[mvp_idx]['return_net']
    mvp_vol = frontier_df.iloc[mvp_idx]['volatility']
    
    # Save summary with flags, MVP stats, and ranges
    summary_df = pd.DataFrame([{
        'portfolio_type': args.portfolio_type,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'allow_short_selling': allow_short,
        'mvp_return_gross': mvp_return_gross,
        'mvp_return_net': mvp_return_net,
        'mvp_volatility': mvp_vol,
        'return_range_gross_min': frontier_df['return_gross'].min(),
        'return_range_gross_max': frontier_df['return_gross'].max(),
        'return_range_net_min': frontier_df['return_net'].min(),
        'return_range_net_max': frontier_df['return_net'].max(),
        'volatility_range_min': frontier_df['volatility'].min(),
        'volatility_range_max': frontier_df['volatility'].max(),
        'num_portfolios': len(portfolio_names),
        'num_observations': len(returns),
        'num_frontier_points': len(frontier['returns'])
    }])
    
    # Check if summary file exists, append if it does
    summary_file = RESULTS_DIR / "ios_summary.csv"
    if summary_file.exists():
        existing_df = pd.read_csv(summary_file)
        # Check if this combination already exists
        mask = (
            (existing_df['portfolio_type'] == args.portfolio_type) &
            (existing_df['start_year'] == args.start_year) &
            (existing_df['end_year'] == args.end_year) &
            (existing_df['allow_short_selling'] == allow_short)
        )
        if mask.any():
            # Update existing row
            existing_df.loc[mask, summary_df.columns] = summary_df.values[0]
            summary_df = existing_df
            print(f"  Updated existing entry in: ios_summary.csv")
        else:
            # Append new row
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            print(f"  Appended to: ios_summary.csv")
    else:
        print(f"  Created: ios_summary.csv")
    
    summary_df.to_csv(summary_file, index=False)
    
    # Plot if requested
    if args.plot:
        print("\nGenerating plot...")
        plot_investment_opportunity_set(
            frontier, mu, Sigma, portfolio_names,
            args.portfolio_type, args.start_year, args.end_year
        )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Frontier points: {len(frontier['returns'])}")
    print(f"Return range (gross): [{frontier_df['return_gross'].min():.6f}, {frontier_df['return_gross'].max():.6f}]")
    print(f"Return range (net): [{frontier_df['return_net'].min():.4%}, {frontier_df['return_net'].max():.4%}]")
    print(f"Volatility range: [{frontier_df['volatility'].min():.4%}, {frontier_df['volatility'].max():.4%}]")
    
    print(f"\nMinimum Variance Portfolio (MVP):")
    print(f"  Return (gross): {mvp_return_gross:.6f}")
    print(f"  Return (net): {mvp_return_net:.4%}")
    print(f"  Volatility: {mvp_vol:.4%}")
    
    print("\n✓ Investment Opportunity Set construction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

