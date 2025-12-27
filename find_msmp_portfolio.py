#!/usr/bin/env python3
"""
Find Minimum Second Moment Portfolio (MSMP)

The MSMP minimizes E[(w'R)²] where R are gross returns.
E[(w'R)²] = w'Σ_R w + (w'μ_R)² = w'(Σ_R + μ_R μ_R')w

This script finds the MSMP for a given portfolio type and time period,
and saves the results including flags, expected return, and volatility.
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


def find_msmp(mu, Sigma, allow_short_selling=True):
    """
    Find Minimum Second Moment Portfolio (MSMP)
    
    MSMP minimizes E[(w'R)²] where R are gross returns.
    E[(w'R)²] = w'Σ_R w + (w'μ_R)² = w'(Σ_R + μ_R μ_R')w
    
    Parameters:
    -----------
    mu : np.array
        Mean gross returns (E[R])
    Sigma : np.array
        Covariance matrix of gross returns (Cov(R))
    allow_short_selling : bool
        If True, allows negative weights (free portfolio)
    
    Returns:
    --------
    w_msmp : np.array
        MSMP portfolio weights
    msmp_return : float
        MSMP expected gross return (E[w'R])
    msmp_vol : float
        MSMP volatility (std of w'R)
    """
    n = len(mu)
    
    # Second moment matrix: Σ_R + μ_R μ_R'
    M = Sigma + np.outer(mu, mu)
    
    # Minimize w'Mw = E[(w'R)²] subject to budget constraint
    def objective(w):
        return w.T @ M @ w
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Set bounds based on short selling constraint
    if allow_short_selling:
        # "Free portfolio" means short selling is allowed
        # Bounds: weights can be between -1 and 2 (allows shorting one asset to go long in another)
        bounds = [(-1, 2) for _ in range(n)]
    else:
        # No short selling: weights must be >= 0
        bounds = [(0, 1) for _ in range(n)]
    
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 2000, 'ftol': 1e-9})
    
    if not result.success:
        # Try trust-constr as fallback
        result = minimize(objective, w0, method='trust-constr',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 2000, 'gtol': 1e-8})
    
    if not result.success:
        raise ValueError(f"Failed to find MSMP: {result.message}")
    
    w_msmp = result.x
    msmp_return = mu.T @ w_msmp  # E[w'R] (gross return)
    msmp_vol = np.sqrt(w_msmp.T @ Sigma @ w_msmp)  # std(w'R)
    
    return w_msmp, msmp_return, msmp_vol


def main():
    parser = argparse.ArgumentParser(
        description='Find Minimum Second Moment Portfolio (MSMP)'
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
        '--output-suffix', type=str, default='',
        help='Suffix to add to output filename (e.g., "_test")'
    )
    parser.add_argument(
        '--no-short-selling', action='store_true',
        help='Restrict portfolio weights to be non-negative (no short selling)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Finding Minimum Second Moment Portfolio (MSMP)")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    
    allow_short = not args.no_short_selling
    print(f"Short selling: {'ALLOWED (free portfolio)' if allow_short else 'NOT ALLOWED (long-only)'}")
    print()
    
    # Load data
    print("Loading portfolio returns...")
    returns = load_portfolio_returns(args.portfolio_type, args.start_year, args.end_year)
    
    # Compute moments
    print("\nComputing moments...")
    mu, Sigma, portfolio_names = compute_moments(returns)
    print(f"  Mean returns range: [{mu.min():.4f}, {mu.max():.4f}] (gross)")
    print(f"  Volatility range: [{np.sqrt(np.diag(Sigma)).min():.4f}, {np.sqrt(np.diag(Sigma)).max():.4f}]")
    
    # Find MSMP
    print("\nFinding MSMP portfolio...")
    w_msmp, msmp_return, msmp_vol = find_msmp(mu, Sigma, allow_short_selling=allow_short)
    
    # Convert gross returns to net returns for display
    msmp_return_net = msmp_return - 1
    
    print(f"  MSMP return (gross): {msmp_return:.6f} (net: {msmp_return_net:.4%})")
    print(f"  MSMP volatility: {msmp_vol:.4%}")
    
    # Save results
    print("\nSaving results...")
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    
    # Save weights
    weights_df = pd.DataFrame({
        'portfolio': portfolio_names,
        'weight': w_msmp
    })
    weights_df.to_csv(RESULTS_DIR / f"msmp_weights{suffix}.csv", index=False)
    print(f"  Saved: msmp_weights{suffix}.csv")
    
    # Save summary with flags, expected return, and volatility
    summary_df = pd.DataFrame([{
        'portfolio_type': args.portfolio_type,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'allow_short_selling': allow_short,
        'expected_return_gross': msmp_return,
        'expected_return_net': msmp_return_net,
        'volatility': msmp_vol,
        'num_portfolios': len(portfolio_names),
        'num_observations': len(returns)
    }])
    
    # Check if summary file exists, append if it does
    summary_file = RESULTS_DIR / "msmp_summary.csv"
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
            print(f"  Updated existing entry in: msmp_summary.csv")
        else:
            # Append new row
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            print(f"  Appended to: msmp_summary.csv")
    else:
        print(f"  Created: msmp_summary.csv")
    
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"MSMP return (net): {msmp_return_net:.4%}")
    print(f"MSMP volatility: {msmp_vol:.4%}")
    print(f"Number of portfolios: {len(portfolio_names)}")
    print(f"Number of observations: {len(returns)}")
    print("\n✓ MSMP calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

