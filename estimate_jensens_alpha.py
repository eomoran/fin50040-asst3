#!/usr/bin/env python3
"""
Estimate Jensen's Alpha

This script estimates Jensen's alpha for each portfolio in two settings:
1. Zero-Beta CAPM: r_i - r_z = α_i + β_i(r_m - r_z)
   where r_z is the Zero-Beta Portfolio return, r_m is the optimal portfolio return
2. Risk-Free CAPM: r_i - r_f = α_i + β_i(r_m - r_f)
   where r_f is the risk-free rate, r_m is the optimal portfolio return

Results are saved to CSV with flags, alpha values, and statistics for each portfolio.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
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


def load_factors(start_year, end_year):
    """
    Load Fama-French factors and risk-free rate
    
    Parameters:
    -----------
    start_year : int
        Start year
    end_year : int
        End year
    
    Returns:
    --------
    factors : pd.DataFrame
        Factor data with columns: Mkt-RF, SMB, HML, RF
        RF is in gross returns (R_f = 1 + r_f)
        Other factors are excess returns
    """
    # Prefer 3-factor model for 1927+ period
    files = list(DATA_DIR.glob("*F-F_Research_Data_Factors*Annual*.csv"))
    if not files:
        raise FileNotFoundError("No annual factors file found")
    
    file_path = files[0]
    print(f"  Loading: {file_path.name}")
    
    factors = pd.read_csv(file_path, index_col=0, parse_dates=True)
    factors = factors.loc[f'{start_year}-01-01':f'{end_year}-01-01']
    
    # Drop rows with any NaN
    factors = factors.dropna()
    
    print(f"  Loaded {len(factors)} observations")
    print(f"  Columns: {list(factors.columns)}")
    
    return factors


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
        Portfolio names (in order matching weights)
    """
    suffix = f"_{portfolio_type}_{start_year}_{end_year}"
    weights_file = RESULTS_DIR / f"optimal_crra_weights{suffix}.csv"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"Optimal portfolio weights not found: {weights_file}")
    
    print(f"  Loading optimal portfolio weights from: {weights_file.name}")
    weights_df = pd.read_csv(weights_file)
    
    w_opt = weights_df['weight'].values
    portfolio_names = weights_df['portfolio'].tolist()
    
    return w_opt, portfolio_names


def load_zbp_weights(portfolio_type, start_year, end_year, rra=4.0, allow_short_selling=True):
    """
    Load Zero-Beta Portfolio weights from saved CSV
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    rra : float
        Relative Risk Aversion coefficient for optimal portfolio
    allow_short_selling : bool
        Whether short selling was allowed
    
    Returns:
    --------
    w_zbp : np.array
        ZBP portfolio weights
    portfolio_names : list
        Portfolio names (in order matching weights)
    """
    suffix = f"_{portfolio_type}_{start_year}_{end_year}"
    weights_file = RESULTS_DIR / f"zbp_weights{suffix}.csv"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"ZBP weights not found: {weights_file}")
    
    print(f"  Loading ZBP weights from: {weights_file.name}")
    weights_df = pd.read_csv(weights_file)
    
    w_zbp = weights_df['weight'].values
    portfolio_names = weights_df['portfolio'].tolist()
    
    return w_zbp, portfolio_names


def compute_portfolio_return(returns, weights, portfolio_names):
    """
    Compute portfolio return time series from weights
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (gross returns R = 1 + r)
        Index: dates, Columns: portfolio names
    weights : np.array
        Portfolio weights
    portfolio_names : list
        Portfolio names matching weights order
    
    Returns:
    --------
    portfolio_returns : pd.Series
        Portfolio return time series (gross returns R)
        Index: dates
    """
    # Ensure portfolio names match
    if list(returns.columns) != portfolio_names:
        # Reorder returns to match weights
        returns = returns[portfolio_names]
    
    # Compute portfolio return: R_p = w'R
    portfolio_returns = (returns * weights).sum(axis=1)
    
    return portfolio_returns


def estimate_jensens_alpha(returns, market_return, risk_free_or_zbp_return, model_name="CAPM"):
    """
    Estimate Jensen's alpha for each portfolio
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio returns (gross returns R = 1 + r)
        Index: dates, Columns: portfolio names
    market_return : pd.Series
        Market portfolio return (gross returns R_m = 1 + r_m)
        Index: dates (must align with returns)
    risk_free_or_zbp_return : pd.Series
        Risk-free rate or ZBP return (gross returns R_f or R_z = 1 + r_f or 1 + r_z)
        Index: dates (must align with returns)
    model_name : str
        Name of the model (for display)
    
    Returns:
    --------
    results : pd.DataFrame
        DataFrame with columns: portfolio, alpha, beta, alpha_tstat, alpha_pvalue, r_squared
    """
    # Align all data to common dates
    common_dates = returns.index.intersection(market_return.index).intersection(risk_free_or_zbp_return.index)
    common_dates = common_dates.drop_duplicates()
    
    returns_aligned = returns.loc[common_dates]
    market_aligned = market_return.loc[common_dates]
    rf_or_zbp_aligned = risk_free_or_zbp_return.loc[common_dates]
    
    # Convert to net returns for regression
    # r_i = R_i - 1, r_m = R_m - 1, r_f = R_f - 1
    returns_net = returns_aligned - 1
    market_net = market_aligned - 1
    rf_or_zbp_net = rf_or_zbp_aligned - 1
    
    # Dependent variable: r_i - r_f (or r_i - r_z)
    # Independent variable: r_m - r_f (or r_m - r_z)
    y = returns_net.sub(rf_or_zbp_net, axis=0)  # r_i - r_f (or r_i - r_z)
    x = market_net - rf_or_zbp_net  # r_m - r_f (or r_m - r_z)
    
    # Estimate alpha and beta for each portfolio
    results = []
    
    for portfolio in returns_net.columns:
        y_portfolio = y[portfolio].dropna()
        x_portfolio = x.loc[y_portfolio.index]
        
        if len(y_portfolio) < 2:
            continue
        
        # Add constant for intercept (alpha)
        X = sm.add_constant(x_portfolio)
        
        # OLS regression: y = α + β*x + ε
        model = sm.OLS(y_portfolio, X)
        result = model.fit()
        
        alpha = result.params['const']
        beta = result.params.iloc[1] if len(result.params) > 1 else np.nan
        alpha_tstat = result.tvalues['const']
        alpha_pvalue = result.pvalues['const']
        r_squared = result.rsquared
        
        results.append({
            'portfolio': portfolio,
            'alpha': alpha,
            'beta': beta,
            'alpha_tstat': alpha_tstat,
            'alpha_pvalue': alpha_pvalue,
            'r_squared': r_squared,
            'n_observations': len(y_portfolio)
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate Jensen\'s Alpha for Zero-Beta CAPM and Risk-Free CAPM'
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
        '--no-short-selling', action='store_true',
        help='Restrict portfolio weights to be non-negative (no short selling)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Estimating Jensen's Alpha")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    print(f"Optimal portfolio RRA: {args.rra}")
    allow_short = not args.no_short_selling
    print(f"Short selling: {'ALLOWED (free portfolio)' if allow_short else 'NOT ALLOWED (long-only)'}")
    print()
    
    # Load data
    print("Loading data...")
    returns = load_portfolio_returns(args.portfolio_type, args.start_year, args.end_year)
    factors = load_factors(args.start_year, args.end_year)
    
    # Load optimal portfolio weights
    print(f"\nLoading optimal CRRA portfolio (RRA={args.rra})...")
    w_opt, opt_portfolio_names = load_optimal_portfolio_weights(
        args.portfolio_type, args.start_year, args.end_year, args.rra, allow_short
    )
    
    # Ensure portfolio names match
    if opt_portfolio_names != list(returns.columns):
        print(f"  Warning: Portfolio name mismatch. Reordering weights...")
        opt_df = pd.DataFrame({'portfolio': opt_portfolio_names, 'weight': w_opt})
        opt_df = opt_df.set_index('portfolio').reindex(returns.columns)
        w_opt = opt_df['weight'].values
        w_opt = np.nan_to_num(w_opt, nan=0.0)
        w_opt = w_opt / np.sum(w_opt)  # Renormalize
        opt_portfolio_names = list(returns.columns)
    
    # Compute optimal portfolio return (market portfolio)
    market_return = compute_portfolio_return(returns, w_opt, opt_portfolio_names)
    market_return_net = market_return - 1
    print(f"  Optimal portfolio return (gross): {market_return.mean():.6f}")
    print(f"  Optimal portfolio return (net): {market_return_net.mean():.4%}")
    
    # Load ZBP weights
    print(f"\nLoading Zero-Beta Portfolio...")
    w_zbp, zbp_portfolio_names = load_zbp_weights(
        args.portfolio_type, args.start_year, args.end_year, args.rra, allow_short
    )
    
    # Ensure portfolio names match
    if zbp_portfolio_names != list(returns.columns):
        print(f"  Warning: Portfolio name mismatch. Reordering weights...")
        zbp_df = pd.DataFrame({'portfolio': zbp_portfolio_names, 'weight': w_zbp})
        zbp_df = zbp_df.set_index('portfolio').reindex(returns.columns)
        w_zbp = zbp_df['weight'].values
        w_zbp = np.nan_to_num(w_zbp, nan=0.0)
        w_zbp = w_zbp / np.sum(w_zbp)  # Renormalize
        zbp_portfolio_names = list(returns.columns)
    
    # Compute ZBP return
    zbp_return = compute_portfolio_return(returns, w_zbp, zbp_portfolio_names)
    zbp_return_net = zbp_return - 1
    print(f"  ZBP return (gross): {zbp_return.mean():.6f}")
    print(f"  ZBP return (net): {zbp_return_net.mean():.4%}")
    
    # Extract risk-free rate
    rf = factors['RF']  # Already in gross returns
    rf_net = rf - 1
    print(f"\nRisk-free rate (gross): {rf.mean():.6f}")
    print(f"Risk-free rate (net): {rf_net.mean():.4%}")
    
    # Estimate Jensen's Alpha: Zero-Beta CAPM
    print(f"\nEstimating Jensen's Alpha: Zero-Beta CAPM...")
    print(f"  Model: r_i - r_z = α_i + β_i(r_m - r_z)")
    alpha_zbp = estimate_jensens_alpha(returns, market_return, zbp_return, model_name="Zero-Beta CAPM")
    print(f"  Estimated alpha for {len(alpha_zbp)} portfolios")
    
    # Estimate Jensen's Alpha: Risk-Free CAPM
    print(f"\nEstimating Jensen's Alpha: Risk-Free CAPM...")
    print(f"  Model: r_i - r_f = α_i + β_i(r_m - r_f)")
    alpha_rf = estimate_jensens_alpha(returns, market_return, rf, model_name="Risk-Free CAPM")
    print(f"  Estimated alpha for {len(alpha_rf)} portfolios")
    
    # Save results
    print("\nSaving results...")
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    
    # Save Zero-Beta CAPM results
    alpha_zbp_file = RESULTS_DIR / f"jensens_alpha_zerobeta{suffix}.csv"
    alpha_zbp.to_csv(alpha_zbp_file, index=False)
    print(f"  Saved: jensens_alpha_zerobeta{suffix}.csv")
    
    # Save Risk-Free CAPM results
    alpha_rf_file = RESULTS_DIR / f"jensens_alpha_riskfree{suffix}.csv"
    alpha_rf.to_csv(alpha_rf_file, index=False)
    print(f"  Saved: jensens_alpha_riskfree{suffix}.csv")
    
    # Save summary with flags
    summary_rows = []
    
    # Zero-Beta CAPM summary
    for _, row in alpha_zbp.iterrows():
        summary_rows.append({
            'portfolio_type': args.portfolio_type,
            'start_year': args.start_year,
            'end_year': args.end_year,
            'rra': args.rra,
            'allow_short_selling': allow_short,
            'model': 'zero_beta_capm',
            'portfolio': row['portfolio'],
            'alpha': row['alpha'],
            'beta': row['beta'],
            'alpha_tstat': row['alpha_tstat'],
            'alpha_pvalue': row['alpha_pvalue'],
            'r_squared': row['r_squared'],
            'n_observations': row['n_observations']
        })
    
    # Risk-Free CAPM summary
    for _, row in alpha_rf.iterrows():
        summary_rows.append({
            'portfolio_type': args.portfolio_type,
            'start_year': args.start_year,
            'end_year': args.end_year,
            'rra': args.rra,
            'allow_short_selling': allow_short,
            'model': 'risk_free_capm',
            'portfolio': row['portfolio'],
            'alpha': row['alpha'],
            'beta': row['beta'],
            'alpha_tstat': row['alpha_tstat'],
            'alpha_pvalue': row['alpha_pvalue'],
            'r_squared': row['r_squared'],
            'n_observations': row['n_observations']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Check if summary file exists, append if it does
    summary_file = RESULTS_DIR / "jensens_alpha_summary.csv"
    if summary_file.exists():
        existing_df = pd.read_csv(summary_file)
        # Remove any existing entries for this combination
        mask = ~(
            (existing_df['portfolio_type'] == args.portfolio_type) &
            (existing_df['start_year'] == args.start_year) &
            (existing_df['end_year'] == args.end_year) &
            (existing_df['rra'] == args.rra) &
            (existing_df['allow_short_selling'] == allow_short)
        )
        existing_df = existing_df[mask]
        # Append new results
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        print(f"  Updated: jensens_alpha_summary.csv")
    else:
        print(f"  Created: jensens_alpha_summary.csv")
    
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"\nZero-Beta CAPM:")
    print(f"  Mean alpha: {alpha_zbp['alpha'].mean():.6f} ({alpha_zbp['alpha'].mean()*100:.4f}%)")
    print(f"  Std alpha: {alpha_zbp['alpha'].std():.6f} ({alpha_zbp['alpha'].std()*100:.4f}%)")
    print(f"  Min alpha: {alpha_zbp['alpha'].min():.6f} ({alpha_zbp['alpha'].min()*100:.4f}%)")
    print(f"  Max alpha: {alpha_zbp['alpha'].max():.6f} ({alpha_zbp['alpha'].max()*100:.4f}%)")
    print(f"  Mean R²: {alpha_zbp['r_squared'].mean():.4f}")
    
    print(f"\nRisk-Free CAPM:")
    print(f"  Mean alpha: {alpha_rf['alpha'].mean():.6f} ({alpha_rf['alpha'].mean()*100:.4f}%)")
    print(f"  Std alpha: {alpha_rf['alpha'].std():.6f} ({alpha_rf['alpha'].std()*100:.4f}%)")
    print(f"  Min alpha: {alpha_rf['alpha'].min():.6f} ({alpha_rf['alpha'].min()*100:.4f}%)")
    print(f"  Max alpha: {alpha_rf['alpha'].max():.6f} ({alpha_rf['alpha'].max()*100:.4f}%)")
    print(f"  Mean R²: {alpha_rf['r_squared'].mean():.4f}")
    
    print("\n✓ Jensen's Alpha estimation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

