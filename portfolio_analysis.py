"""
Portfolio Analysis for Assignment 3
Mean-Variance Portfolio Theory and the CAPM

This script performs:
1. Constructs (ex-post) MV frontier (no riskless asset, budget constraint only)
2. Finds Minimum Second Moment Portfolio (MSMP)
3. Finds optimal portfolio for CRRA investor with RRA=4
4. Finds Zero-β portfolio for optimal portfolio
5. Estimates Jensen's α for each portfolio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data paths
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_portfolio_data(portfolio_type="size", start_year=1927, end_year=2013):
    """
    Load portfolio return data
    
    Parameters:
    -----------
    portfolio_type : str
        'size' for size portfolios, 'value' for BE-ME portfolios
    start_year : int
        Start year for data
    end_year : int
        End year for data
    """
    if portfolio_type == "size":
        file_pattern = "*Portfolios_Formed_on_ME*.csv"
    elif portfolio_type == "value":
        file_pattern = "*Portfolios_Formed_on_BE-ME*.csv"
    else:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")
    
    # Find the file
    files = list(DATA_DIR.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No file found matching {file_pattern}")
    
    df = pd.read_csv(files[0], index_col=0)
    
    # Parse dates - Fama-French uses YYYYMM format
    if df.index.dtype == 'object' or not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse as YYYYMM format
        try:
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m', errors='coerce')
        except:
            # If that fails, try standard parsing
            df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Drop any rows with NaT dates
    df = df[df.index.notna()]
    
    # Filter date range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Convert returns from percentage to decimal if needed
    # Fama-French data is typically in percentage form
    if df.abs().max().max() > 1:
        df = df / 100.0
    
    return df


def load_factors(start_year=1927, end_year=2013):
    """Load Fama-French factors and risk-free rate"""
    files = list(DATA_DIR.glob("*Factors*.csv"))
    if not files:
        raise FileNotFoundError("No factors file found")
    
    df = pd.read_csv(files[0], index_col=0)
    
    # Parse dates - Fama-French uses YYYYMM format
    if df.index.dtype == 'object' or not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse as YYYYMM format
        try:
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m', errors='coerce')
        except:
            # If that fails, try standard parsing
            df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Drop any rows with NaT dates
    df = df[df.index.notna()]
    
    # Filter date range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Convert from percentage to decimal if needed
    if df.abs().max().max() > 1:
        df = df / 100.0
    
    return df


def compute_moments(returns):
    """
    Compute mean returns and covariance matrix
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data (time x assets)
    
    Returns:
    --------
    mu : np.array
        Mean returns
    Sigma : np.array
        Covariance matrix
    portfolio_names : list
        Names of portfolios used (after filtering)
    """
    # Remove portfolios with zero or very low variance (constant returns)
    variances = returns.var()
    min_variance = 1e-10  # Minimum variance threshold
    valid_portfolios = variances > min_variance
    
    if valid_portfolios.sum() < 2:
        raise ValueError("Not enough portfolios with non-zero variance")
    
    # Filter returns to only valid portfolios
    returns_filtered = returns.loc[:, valid_portfolios]
    portfolio_names = returns_filtered.columns.tolist()
    
    mu = returns_filtered.mean().values
    Sigma = returns_filtered.cov().values
    
    return mu, Sigma, portfolio_names


def construct_mv_frontier(mu, Sigma, num_portfolios=100):
    """
    Construct the mean-variance frontier
    
    Parameters:
    -----------
    mu : np.array
        Mean returns
    Sigma : np.array
        Covariance matrix
    num_portfolios : int
        Number of portfolios on the frontier
    
    Returns:
    --------
    frontier_data : dict
        Dictionary with 'returns', 'volatilities', 'weights' for frontier
    """
    n = len(mu)
    
    # Find minimum and maximum expected returns
    # Minimum: minimum variance portfolio
    # Maximum: maximum return portfolio (100% in highest return asset)
    
    # Solve for minimum variance portfolio
    ones = np.ones(n)
    
    # Use pseudo-inverse for robustness (handles near-singular matrices)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        inv_Sigma = np.linalg.pinv(Sigma)
    
    # Minimum variance portfolio weights
    w_minvar = inv_Sigma @ ones / (ones.T @ inv_Sigma @ ones)
    mu_minvar = mu.T @ w_minvar
    
    # Maximum return
    mu_max = mu.max()
    
    # Generate target returns between min and max
    target_returns = np.linspace(mu_minvar, mu_max, num_portfolios)
    
    frontier_returns = []
    frontier_vols = []
    frontier_weights = []
    
    for target_return in target_returns:
        # Minimize variance subject to:
        # 1. Expected return = target_return
        # 2. Weights sum to 1 (budget constraint)
        
        def objective(w):
            return w.T @ Sigma @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget constraint
            {'type': 'eq', 'fun': lambda w: mu.T @ w - target_return}  # Return constraint
        ]
        
        bounds = [(-1, 1) for _ in range(n)]  # Allow short selling
        
        # Initial guess: equal weights
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            w = result.x
            portfolio_return = mu.T @ w
            portfolio_vol = np.sqrt(w.T @ Sigma @ w)
            
            frontier_returns.append(portfolio_return)
            frontier_vols.append(portfolio_vol)
            frontier_weights.append(w)
    
    return {
        'returns': np.array(frontier_returns),
        'volatilities': np.array(frontier_vols),
        'weights': np.array(frontier_weights)
    }


def find_msmp(mu, Sigma):
    """
    Find Minimum Second Moment Portfolio (MSMP)
    
    MSMP minimizes w'Σw + (w'μ)² = w'(Σ + μμ')w
    subject to budget constraint
    
    Parameters:
    -----------
    mu : np.array
        Mean returns
    Sigma : np.array
        Covariance matrix
    
    Returns:
    --------
    w_msmp : np.array
        MSMP portfolio weights
    msmp_return : float
        MSMP expected return
    msmp_vol : float
        MSMP volatility
    """
    n = len(mu)
    
    # Second moment matrix: Σ + μμ'
    M = Sigma + np.outer(mu, mu)
    
    # Minimize w'Mw subject to budget constraint
    def objective(w):
        return w.T @ M @ w
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(-1, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Failed to find MSMP")
    
    w_msmp = result.x
    msmp_return = mu.T @ w_msmp
    msmp_vol = np.sqrt(w_msmp.T @ Sigma @ w_msmp)
    
    return w_msmp, msmp_return, msmp_vol


def find_optimal_crra_portfolio(mu, Sigma, rra=4):
    """
    Find optimal portfolio for CRRA investor
    
    CRRA utility: U(W) = W^(1-γ) / (1-γ) where γ = RRA
    
    For normal returns, optimal portfolio maximizes:
    μ'w - (γ/2) * w'Σw
    
    Parameters:
    -----------
    mu : np.array
        Mean returns
    Sigma : np.array
        Covariance matrix
    rra : float
        Relative Risk Aversion coefficient (default: 4)
    
    Returns:
    --------
    w_opt : np.array
        Optimal portfolio weights
    opt_return : float
        Optimal expected return
    opt_vol : float
        Optimal volatility
    """
    n = len(mu)
    
    # Maximize μ'w - (γ/2) * w'Σw
    # Equivalent to minimizing: (γ/2) * w'Σw - μ'w
    def objective(w):
        return (rra / 2) * (w.T @ Sigma @ w) - mu.T @ w
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(-1, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Failed to find optimal CRRA portfolio")
    
    w_opt = result.x
    opt_return = mu.T @ w_opt
    opt_vol = np.sqrt(w_opt.T @ Sigma @ w_opt)
    
    return w_opt, opt_return, opt_vol


def find_zero_beta_portfolio(mu, Sigma, w_m):
    """
    Find Zero-β portfolio for a given portfolio w_m
    
    Zero-β portfolio w_z satisfies:
    - β_z = Cov(r_z, r_m) / Var(r_m) = 0
    - This means Cov(r_z, r_m) = 0
    - w_z'Σw_m = 0
    
    Parameters:
    -----------
    mu : np.array
        Mean returns
    Sigma : np.array
        Covariance matrix
    w_m : np.array
        Market/optimal portfolio weights
    
    Returns:
    --------
    w_z : np.array
        Zero-β portfolio weights
    z_return : float
        Zero-β expected return
    z_vol : float
        Zero-β volatility
    """
    n = len(mu)
    
    # Minimize variance subject to:
    # 1. Budget constraint: sum(w) = 1
    # 2. Zero beta: w'Σw_m = 0
    
    def objective(w):
        return w.T @ Sigma @ w
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: w.T @ Sigma @ w_m}
    ]
    
    bounds = [(-1, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Failed to find zero-β portfolio")
    
    w_z = result.x
    z_return = mu.T @ w_z
    z_vol = np.sqrt(w_z.T @ Sigma @ w_z)
    
    return w_z, z_return, z_vol


def estimate_jensens_alpha(returns, factors, rf, use_zero_beta=False, w_z=None):
    """
    Estimate Jensen's α for each portfolio
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio returns (time x portfolios)
    factors : pd.DataFrame
        Factor returns (time x factors)
    rf : pd.Series
        Risk-free rate
    use_zero_beta : bool
        If True, use zero-β CAPM; if False, use standard CAPM with Rf
    w_z : np.array, optional
        Zero-β portfolio weights (if use_zero_beta=True)
    
    Returns:
    --------
    alphas : pd.Series
        Jensen's α for each portfolio
    betas : pd.DataFrame
        Betas for each portfolio
    """
    # Align dates - use intersection of dates
    common_dates = returns.index.intersection(rf.index)
    
    # Remove duplicates if any
    common_dates = common_dates[~common_dates.duplicated()]
    
    # Filter to common dates
    returns_aligned = returns.loc[common_dates]
    rf_aligned = rf.loc[common_dates]
    factors_aligned = factors.loc[common_dates] if hasattr(factors, 'loc') else factors
    
    # Remove any remaining duplicates
    returns_aligned = returns_aligned[~returns_aligned.index.duplicated(keep='first')]
    rf_aligned = rf_aligned[~rf_aligned.index.duplicated(keep='first')]
    
    # Align again after deduplication
    common_dates = returns_aligned.index.intersection(rf_aligned.index)
    returns_aligned = returns_aligned.loc[common_dates]
    rf_aligned = rf_aligned.loc[common_dates]
    if hasattr(factors, 'loc'):
        factors_aligned = factors_aligned.loc[common_dates]
    
    # Excess returns
    excess_returns = returns_aligned.subtract(rf_aligned, axis=0)
    
    alphas = {}
    betas_dict = {}
    
    for portfolio in returns_aligned.columns:
        portfolio_excess = excess_returns[portfolio]
        
        if use_zero_beta and w_z is not None:
            # Zero-β CAPM: r_i - r_z = α_i + β_i(r_m - r_z)
            # Compute zero-β portfolio return
            # w_z should be aligned with returns_aligned.columns
            z_return = (returns_aligned * w_z).sum(axis=1)
            z_return_aligned = z_return.loc[common_dates]
            market_excess = portfolio_excess - (z_return_aligned - rf_aligned)
            
            # Regression: r_i - r_z = α_i + β_i(r_m - r_z)
            X = market_excess.values.reshape(-1, 1)
            X = np.column_stack([np.ones(len(X)), X])  # Add intercept
            y = portfolio_excess.values
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha = beta[0]
            beta_m = beta[1]
            
        else:
            # Standard CAPM: r_i - r_f = α_i + β_i(r_m - r_f)
            # Use market factor (Mkt-RF) as market return
            if hasattr(factors_aligned, 'columns') and 'Mkt-RF' in factors_aligned.columns:
                market_excess = factors_aligned['Mkt-RF'].loc[common_dates]
            else:
                # If no Mkt-RF, use first factor
                market_excess = factors_aligned.iloc[:, 0].loc[common_dates] if hasattr(factors_aligned, 'loc') else factors_aligned.iloc[:, 0]
            
            # Regression: r_i - r_f = α_i + β_i(r_m - r_f)
            X = market_excess.values.reshape(-1, 1)
            X = np.column_stack([np.ones(len(X)), X])
            y = portfolio_excess.values
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha = beta[0]
            beta_m = beta[1]
        
        alphas[portfolio] = alpha
        betas_dict[portfolio] = beta_m
    
    return pd.Series(alphas), pd.Series(betas_dict)


def main():
    """Main analysis function"""
    print("=" * 70)
    print("Portfolio Analysis for Assignment 3")
    print("=" * 70)
    print()
    
    # Analysis parameters
    start_year = 1927
    end_year = 2013
    rra = 4
    
    print(f"Analysis period: {start_year}-{end_year}")
    print(f"CRRA RRA coefficient: {rra}")
    print()
    
    # Load data
    print("Loading data...")
    try:
        returns = load_portfolio_data("size", start_year, end_year)
        factors = load_factors(start_year, end_year)
        print(f"  Loaded {len(returns.columns)} portfolios")
        print(f"  Time period: {returns.index[0].date()} to {returns.index[-1].date()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Compute moments
    print("\nComputing moments...")
    mu, Sigma, portfolio_names = compute_moments(returns)
    print(f"  Using {len(portfolio_names)} portfolios (filtered from {len(returns.columns)})")
    print(f"  Mean returns range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  Annualized volatility range: [{np.sqrt(np.diag(Sigma)).min():.4f}, {np.sqrt(np.diag(Sigma)).max():.4f}]")
    
    # Construct MV frontier
    print("\nConstructing MV frontier...")
    frontier = construct_mv_frontier(mu, Sigma)
    print(f"  Generated {len(frontier['returns'])} portfolios on frontier")
    
    # Find MSMP
    print("\nFinding MSMP portfolio...")
    w_msmp, msmp_return, msmp_vol = find_msmp(mu, Sigma)
    print(f"  MSMP return: {msmp_return:.4f}")
    print(f"  MSMP volatility: {msmp_vol:.4f}")
    
    # Find optimal CRRA portfolio
    print(f"\nFinding optimal portfolio for CRRA investor (RRA={rra})...")
    w_opt, opt_return, opt_vol = find_optimal_crra_portfolio(mu, Sigma, rra)
    print(f"  Optimal return: {opt_return:.4f}")
    print(f"  Optimal volatility: {opt_vol:.4f}")
    
    # Find Zero-β portfolio
    print("\nFinding Zero-β portfolio for optimal portfolio...")
    w_z, z_return, z_vol = find_zero_beta_portfolio(mu, Sigma, w_opt)
    print(f"  Zero-β return: {z_return:.4f}")
    print(f"  Zero-β volatility: {z_vol:.4f}")
    
    # Estimate Jensen's α
    print("\nEstimating Jensen's α...")
    if 'RF' in factors.columns:
        rf = factors['RF']
    else:
        print("  Warning: RF not found in factors, using zero")
        rf = pd.Series(0, index=factors.index)
    
    # Map w_z to full portfolio set (with zeros for filtered portfolios)
    w_z_full = pd.Series(0.0, index=returns.columns)
    w_z_full[portfolio_names] = w_z
    
    # Zero-β CAPM version
    print("  Zero-β CAPM version...")
    alphas_zb, betas_zb = estimate_jensens_alpha(returns, factors, rf, use_zero_beta=True, w_z=w_z_full.values)
    
    # Standard CAPM version
    print("  Standard CAPM (with Rf) version...")
    alphas_rf, betas_rf = estimate_jensens_alpha(returns, factors, rf, use_zero_beta=False)
    
    # Save results
    print("\nSaving results...")
    results = {
        'msmp': {
            'weights': w_msmp,
            'return': msmp_return,
            'volatility': msmp_vol
        },
        'optimal': {
            'weights': w_opt,
            'return': opt_return,
            'volatility': opt_vol
        },
        'zero_beta': {
            'weights': w_z,
            'return': z_return,
            'volatility': z_vol
        },
        'jensens_alpha_zerobeta': alphas_zb,
        'jensens_alpha_rf': alphas_rf,
        'betas_zerobeta': betas_zb,
        'betas_rf': betas_rf,
        'frontier': frontier
    }
    
    # Save to files (use portfolio_names for indexing)
    # Create full weight vectors with zeros for filtered portfolios
    w_msmp_full = pd.Series(0.0, index=returns.columns)
    w_msmp_full[portfolio_names] = w_msmp
    w_msmp_full.to_csv(OUTPUT_DIR / "msmp_weights.csv")
    
    w_opt_full = pd.Series(0.0, index=returns.columns)
    w_opt_full[portfolio_names] = w_opt
    w_opt_full.to_csv(OUTPUT_DIR / "optimal_weights.csv")
    
    w_z_full = pd.Series(0.0, index=returns.columns)
    w_z_full[portfolio_names] = w_z
    w_z_full.to_csv(OUTPUT_DIR / "zero_beta_weights.csv")
    alphas_zb.to_csv(OUTPUT_DIR / "jensens_alpha_zerobeta.csv")
    alphas_rf.to_csv(OUTPUT_DIR / "jensens_alpha_rf.csv")
    
    print(f"  Results saved to {OUTPUT_DIR}/")
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()

