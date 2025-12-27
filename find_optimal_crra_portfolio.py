#!/usr/bin/env python3
"""
Find Optimal CRRA Portfolio

For a CRRA investor with utility U(W) = W^(1-γ) / (1-γ) where γ = RRA,
the optimal portfolio maximizes: μ'w - (γ/2) * w'Σw

This script finds the optimal CRRA portfolio for a given RRA coefficient,
portfolio type, and time period, and saves the results including flags,
expected return, and volatility.
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


def find_optimal_crra_portfolio(mu, Sigma, rra=4, allow_short_selling=True):
    """
    Find optimal portfolio for CRRA investor
    
    CRRA utility: U(W) = W^(1-γ) / (1-γ) where γ = RRA
    
    For normal returns, optimal portfolio maximizes:
    μ'w - (γ/2) * w'Σw
    
    Parameters:
    -----------
    mu : np.array
        Mean gross returns (E[R])
    Sigma : np.array
        Covariance matrix of gross returns (Cov(R))
    rra : float
        Relative Risk Aversion coefficient (default: 4)
    allow_short_selling : bool
        If True, allows negative weights (free portfolio)
    
    Returns:
    --------
    w_opt : np.array
        Optimal portfolio weights
    opt_return : float
        Optimal expected gross return (E[w'R])
    opt_vol : float
        Optimal volatility (std of w'R)
    """
    n = len(mu)
    
    # Maximize μ'w - (γ/2) * w'Σw
    # Equivalent to minimizing: (γ/2) * w'Σw - μ'w
    def objective(w):
        return (rra / 2) * (w.T @ Sigma @ w) - mu.T @ w
    
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
        raise ValueError(f"Failed to find optimal CRRA portfolio: {result.message}")
    
    w_opt = result.x
    opt_return = mu.T @ w_opt  # E[w'R] (gross return)
    opt_vol = np.sqrt(w_opt.T @ Sigma @ w_opt)  # std(w'R)
    
    return w_opt, opt_return, opt_vol


def main():
    parser = argparse.ArgumentParser(
        description='Find Optimal CRRA Portfolio'
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
        help='Relative Risk Aversion coefficient (default: 4.0)'
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
    print("Finding Optimal CRRA Portfolio")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"Relative Risk Aversion (RRA): {args.rra}")
    
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
    
    # Find optimal CRRA portfolio
    print(f"\nFinding optimal CRRA portfolio (RRA={args.rra})...")
    w_opt, opt_return, opt_vol = find_optimal_crra_portfolio(
        mu, Sigma, rra=args.rra, allow_short_selling=allow_short
    )
    
    # Convert gross returns to net returns for display
    opt_return_net = opt_return - 1
    
    print(f"  Optimal return (gross): {opt_return:.6f}")
    print(f"  Optimal return (net): {opt_return_net:.4%}")
    print(f"  Optimal volatility: {opt_vol:.4%}")
    
    # Save results
    print("\nSaving results...")
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    
    # Save weights
    weights_df = pd.DataFrame({
        'portfolio': portfolio_names,
        'weight': w_opt
    })
    weights_df.to_csv(RESULTS_DIR / f"optimal_crra_weights{suffix}.csv", index=False)
    print(f"  Saved: optimal_crra_weights{suffix}.csv")
    
    # Save summary with flags, expected return, and volatility
    summary_df = pd.DataFrame([{
        'portfolio_type': args.portfolio_type,
        'start_year': args.start_year,
        'end_year': args.end_year,
        'rra': args.rra,
        'allow_short_selling': allow_short,
        'expected_return_gross': opt_return,
        'expected_return_net': opt_return_net,
        'volatility': opt_vol,
        'num_portfolios': len(portfolio_names),
        'num_observations': len(returns)
    }])
    
    # Check if summary file exists, append if it does
    summary_file = RESULTS_DIR / "optimal_crra_summary.csv"
    if summary_file.exists():
        existing_df = pd.read_csv(summary_file)
        # Check if this combination already exists
        mask = (
            (existing_df['portfolio_type'] == args.portfolio_type) &
            (existing_df['start_year'] == args.start_year) &
            (existing_df['end_year'] == args.end_year) &
            (existing_df['rra'] == args.rra) &
            (existing_df['allow_short_selling'] == allow_short)
        )
        if mask.any():
            # Update existing row
            existing_df.loc[mask, summary_df.columns] = summary_df.values[0]
            summary_df = existing_df
            print(f"  Updated existing entry in: optimal_crra_summary.csv")
        else:
            # Append new row
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            print(f"  Appended to: optimal_crra_summary.csv")
    else:
        print(f"  Created: optimal_crra_summary.csv")
    
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Optimal return (gross): {opt_return:.6f}")
    print(f"Optimal return (net): {opt_return_net:.4%}")
    print(f"Optimal volatility: {opt_vol:.4%}")
    print(f"Number of portfolios: {len(portfolio_names)}")
    print(f"Number of observations: {len(returns)}")
    print("\n✓ Optimal CRRA portfolio calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

