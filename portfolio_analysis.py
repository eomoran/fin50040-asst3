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
import argparse
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
    
    # Find the file - prefer annual value-weighted section (as per assignment)
    # The processed files now have section names in the filename
    files = list(DATA_DIR.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No file found matching {file_pattern}")
    
    # Prefer annual value-weighted if available (as specified in assignment: "Annual value-weighted gross returns")
    annual_vw_files = [f for f in files if 'Annual' in f.name and ('Value' in f.name or 'Weight' in f.name) and 'Equal' not in f.name]
    if annual_vw_files:
        file_to_use = annual_vw_files[0]
        print(f"  Using: {file_to_use.name}")
        print(f"  Section: Annual Value-Weighted Returns (as per assignment)")
    else:
        # Fall back to monthly value-weighted
        monthly_vw_files = [f for f in files if 'Monthly' in f.name and ('Value' in f.name or 'Weight' in f.name) and 'Equal' not in f.name]
        if monthly_vw_files:
            file_to_use = monthly_vw_files[0]
            print(f"  Using: {file_to_use.name}")
            print(f"  Section: Monthly Value-Weighted (Annual not found)")
        else:
            # Fall back to first file found
            file_to_use = files[0]
            print(f"  Using: {file_to_use.name} (no section preference found)")
    
    df = pd.read_csv(file_to_use, index_col=0)
    
    # Parse dates - Fama-French uses various formats:
    # - YYYYMM format (monthly data)
    # - YYYY-MM-DD format (annual data from processed files)
    # - YYYY format (some annual data)
    if df.index.dtype == 'object' or not isinstance(df.index, pd.DatetimeIndex):
        # Try to detect format
        sample = str(df.index[0]) if len(df) > 0 else ''
        if len(sample) == 6 and sample.isdigit():
            # YYYYMM format (monthly)
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m', errors='coerce')
        elif len(sample) == 4 and sample.isdigit():
            # YYYY format (annual)
            df.index = pd.to_datetime(df.index.astype(str) + '-01-01', format='%Y-%m-%d', errors='coerce')
        else:
            # Try standard parsing (handles YYYY-MM-DD and other formats)
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


def load_factors(start_year=1927, end_year=2013, prefer_annual=False):
    """
    Load Fama-French factors and risk-free rate
    
    Parameters:
    -----------
    start_year : int
        Start year for data
    end_year : int
        End year for data
    prefer_annual : bool
        If True, prefer annual factors (for annual portfolio returns).
        If annual factors don't cover the full period, will use monthly and aggregate.
    """
    files = list(DATA_DIR.glob("*Factors*.csv"))
    if not files:
        raise FileNotFoundError("No factors file found")
    
    # Prefer annual factors if requested
    if prefer_annual:
        # Prefer US factors (F-F_Research_Data) over Asia Pacific
        # For 1927-2013 period, prefer 3-factor model (has data from 1927) over 5-factor (starts 1963)
        annual_files = [f for f in files if 'Annual' in f.name]
        # Prefer 3-factor model (covers 1927+) over 5-factor (starts 1963)
        us_annual_3f = [f for f in annual_files if 'F-F_Research_Data_Factors' in f.name and '5' not in f.name]
        us_annual_5f = [f for f in annual_files if 'F-F_Research_Data' in f.name and '5_Factors' in f.name]
        us_annual_files = us_annual_3f + us_annual_5f  # Prefer 3-factor first
        
        use_annual = False
        if us_annual_files:
            file_to_use = us_annual_files[0]
            print(f"  Trying annual factors: {file_to_use.name}")
            df = pd.read_csv(file_to_use, index_col=0)
            
            # Parse dates
            if df.index.dtype == 'object' or not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]
            
            # Check if annual factors cover the full period
            start_date = pd.Timestamp(f"{start_year}-01-01")
            end_date = pd.Timestamp(f"{end_year}-12-31")
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(df_filtered) > 0 and df.index.min() <= start_date:
                # Annual factors cover the period, use them
                df = df_filtered
                print(f"  Using annual factors: {file_to_use.name}")
                use_annual = True
        
        if not use_annual:
            # Annual factors don't cover the period or don't exist, use monthly and aggregate
            if not use_annual and us_annual_files:
                print(f"  Annual factors only cover {df.index.min().year}-{df.index.max().year}, using monthly factors and aggregating")
            
            # Use monthly factors and aggregate to annual
            # Prefer 3-factor model (covers 1927+) over 5-factor (starts 1963)
            monthly_files = [f for f in files if 'Annual' not in f.name]
            us_monthly_3f = [f for f in monthly_files if 'F-F_Research_Data_Factors' in f.name and '5' not in f.name]
            us_monthly_5f = [f for f in monthly_files if 'F-F_Research_Data' in f.name and '5_Factors' in f.name]
            us_monthly_files = us_monthly_3f + us_monthly_5f  # Prefer 3-factor first
            if us_monthly_files:
                file_to_use = us_monthly_files[0]
            elif monthly_files:
                file_to_use = monthly_files[0]
            else:
                file_to_use = files[0]
            
            print(f"  Using monthly factors: {file_to_use.name}")
            df = pd.read_csv(file_to_use, index_col=0)
            
            # Parse dates (YYYYMM format for monthly)
            if df.index.dtype == 'object' or not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index.astype(str), format='%Y%m', errors='coerce')
            df = df[df.index.notna()]
            
            # Filter date range
            start_date = pd.Timestamp(f"{start_year}-01-01")
            end_date = pd.Timestamp(f"{end_year}-12-31")
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Aggregate monthly to annual: compound returns
            df_annual = df.groupby(df.index.year).apply(lambda x: (1 + x).prod() - 1)
            df_annual.index = pd.to_datetime(df_annual.index.astype(str) + '-01-01', format='%Y-%m-%d')
            df = df_annual
            
            print(f"  Aggregated {len(df)} annual observations from monthly data")
    else:
        # Prefer monthly (non-annual) factors, and US over Asia Pacific
        # Prefer 3-factor model (covers 1927+) over 5-factor (starts 1963)
        monthly_files = [f for f in files if 'Annual' not in f.name]
        us_monthly_3f = [f for f in monthly_files if 'F-F_Research_Data_Factors' in f.name and '5' not in f.name]
        us_monthly_5f = [f for f in monthly_files if 'F-F_Research_Data' in f.name and '5_Factors' in f.name]
        us_monthly_files = us_monthly_3f + us_monthly_5f  # Prefer 3-factor first
        if us_monthly_files:
            file_to_use = us_monthly_files[0]
        elif monthly_files:
            file_to_use = monthly_files[0]
        else:
            file_to_use = files[0]
        
        df = pd.read_csv(file_to_use, index_col=0)
        
        # Parse dates (YYYYMM format for monthly)
        if df.index.dtype == 'object' or not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m', errors='coerce')
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


def estimate_jensens_alpha(returns, factors, rf, use_zero_beta=False, w_z=None, w_m=None):
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
    w_m : np.array, optional
        Market/optimal portfolio weights (if use_zero_beta=True, needed for market return)
    
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
            z_return = (returns_aligned * w_z).sum(axis=1)
            z_return_aligned = z_return.loc[common_dates]
            
            # Get market return (use optimal portfolio as market)
            if w_m is not None:
                market_return = (returns_aligned * w_m).sum(axis=1)
                market_return_aligned = market_return.loc[common_dates]
            elif hasattr(factors_aligned, 'columns') and 'Mkt-RF' in factors_aligned.columns:
                # Fallback to Mkt-RF if optimal portfolio not provided
                market_return_aligned = (factors_aligned['Mkt-RF'].loc[common_dates] + rf_aligned)
            else:
                raise ValueError("Need market portfolio weights (w_m) for zero-beta CAPM")
            
            # Market excess over zero-beta: r_m - r_z
            market_excess = market_return_aligned - z_return_aligned
            
            # Dependent variable: r_i - r_z (portfolio return minus zero-beta return)
            portfolio_return = returns_aligned[portfolio].loc[common_dates]
            y = (portfolio_return - z_return_aligned).values
            
            # Regression: r_i - r_z = α_i + β_i(r_m - r_z)
            X = market_excess.values.reshape(-1, 1)
            X = np.column_stack([np.ones(len(X)), X])  # Add intercept
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha = beta[0]
            beta_m = beta[1]
            
            # Debug: Check if regression worked
            if abs(alpha) < 1e-10 and abs(beta_m) < 1e-10:
                print(f"        Warning: Very small alpha/beta for {portfolio} - may indicate regression issue")
            
        else:
            # Standard CAPM: r_i - r_f = α_i + β_i(r_m - r_f)
            # Use market factor (Mkt-RF) as market excess return
            if hasattr(factors_aligned, 'columns') and 'Mkt-RF' in factors_aligned.columns:
                market_excess = factors_aligned['Mkt-RF'].loc[common_dates]
            else:
                # If no Mkt-RF, use first factor
                market_excess = factors_aligned.iloc[:, 0].loc[common_dates] if hasattr(factors_aligned, 'loc') else factors_aligned.iloc[:, 0]
            
            # Ensure market_excess and portfolio_excess are aligned
            common_market_dates = portfolio_excess.index.intersection(market_excess.index)
            portfolio_excess_aligned = portfolio_excess.loc[common_market_dates]
            market_excess_aligned = market_excess.loc[common_market_dates]
            
            # Regression: r_i - r_f = α_i + β_i(r_m - r_f)
            X = market_excess_aligned.values.reshape(-1, 1)
            X = np.column_stack([np.ones(len(X)), X])
            y = portfolio_excess_aligned.values
            
            # Check for valid data
            if len(y) < 2 or np.all(np.isnan(y)) or np.all(np.isnan(X[:, 1])):
                print(f"        Warning: Insufficient valid data for {portfolio}")
                alpha = 0.0
                beta_m = 0.0
            else:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                alpha = beta[0]
                beta_m = beta[1]
                
                # Debug: Check if regression worked
                if abs(alpha) < 1e-10 and abs(beta_m) < 1e-10:
                    print(f"        Warning: Very small alpha/beta for {portfolio} - may indicate regression issue")
        
        alphas[portfolio] = alpha
        betas_dict[portfolio] = beta_m
    
    return pd.Series(alphas), pd.Series(betas_dict)


def exclude_small_caps(returns, portfolio_type="size"):
    """
    Exclude small cap portfolios from the returns data
    
    For size portfolios, small caps are typically the smallest deciles.
    Common small cap portfolio names: 'Lo 10', '2-Dec', 'Lo 20', 'Lo 30', '<= 0'
    
    Note: BE-ME portfolios are sorted only by book-to-market ratio, not by size.
    This function only applies to size portfolios. For double-sorted portfolios
    (size x value), a different approach would be needed.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio returns
    portfolio_type : str
        'size' or 'value'
    
    Returns:
    --------
    returns_filtered : pd.DataFrame
        Returns with small caps excluded
    """
    if portfolio_type != "size":
        # BE-ME portfolios are sorted only by book-to-market, not by size
        # So there are no "small caps" to exclude in value portfolios
        print(f"  Note: {portfolio_type} portfolios are not sorted by size, so --exclude-small-caps has no effect")
        return returns
    
    # Small cap portfolio names (typically smallest deciles)
    small_cap_names = ['<= 0', 'Lo 10', '2-Dec', 'Lo 20', 'Lo 30']
    
    # Find columns that match small cap names
    cols_to_exclude = [col for col in returns.columns if any(sc in str(col) for sc in small_cap_names)]
    
    if cols_to_exclude:
        print(f"  Excluding small caps: {cols_to_exclude}")
        returns_filtered = returns.drop(columns=cols_to_exclude)
        return returns_filtered
    else:
        return returns


def recentre_returns(returns, factors, rf):
    """
    Recentre the data set by aligning means with CAPM and betas
    
    This adjusts portfolio returns so that their means align with CAPM predictions
    based on their betas. The recentring process:
    1. Estimate betas for each portfolio
    2. Calculate expected returns from CAPM: E[r_i] = r_f + β_i * (E[r_m] - r_f)
    3. Adjust returns: r_i_recentred = r_i - (mean(r_i) - E[r_i])
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio returns (time x portfolios)
    factors : pd.DataFrame
        Factor returns (time x factors)
    rf : pd.Series
        Risk-free rate
    
    Returns:
    --------
    returns_recentred : pd.DataFrame
        Recentred portfolio returns (same index as input returns)
    """
    # Align dates
    common_dates = returns.index.intersection(rf.index)
    common_dates = common_dates[~common_dates.duplicated()]
    
    returns_aligned = returns.loc[common_dates]
    rf_aligned = rf.loc[common_dates]
    factors_aligned = factors.loc[common_dates] if hasattr(factors, 'loc') else factors
    
    # Get market excess return
    if hasattr(factors_aligned, 'columns') and 'Mkt-RF' in factors_aligned.columns:
        market_excess = factors_aligned['Mkt-RF']
    else:
        market_excess = factors_aligned.iloc[:, 0]
    
    # Calculate market return
    market_return = market_excess + rf_aligned
    
    # Expected market return and risk-free rate
    E_rf = rf_aligned.mean()
    E_rm = market_return.mean()
    market_excess_mean = E_rm - E_rf
    
    # Recentre each portfolio
    returns_recentred = returns.copy()  # Start with original to preserve all dates
    
    for portfolio in returns_aligned.columns:
        # Estimate beta
        portfolio_excess = returns_aligned[portfolio] - rf_aligned
        common_dates_port = portfolio_excess.index.intersection(market_excess.index)
        
        if len(common_dates_port) < 2:
            continue
        
        y = portfolio_excess.loc[common_dates_port].values
        X = market_excess.loc[common_dates_port].values.reshape(-1, 1)
        X = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            beta_i = beta[1] if len(beta) > 1 else 0
        except:
            beta_i = 0
        
        # Expected return from CAPM
        E_ri = E_rf + beta_i * market_excess_mean
        
        # Actual mean return
        mean_ri = returns_aligned[portfolio].mean()
        
        # Adjustment: subtract the difference between actual and expected mean
        adjustment = mean_ri - E_ri
        
        # Apply adjustment to all dates (not just common dates)
        returns_recentred[portfolio] = returns[portfolio] - adjustment
    
    return returns_recentred


def main(start_year=1927, end_year=2013, portfolio_type="size", rra=4, 
         output_suffix=None, exclude_small_caps_flag=False, recentre_flag=False):
    """
    Main analysis function
    
    Parameters:
    -----------
    start_year : int
        Start year for analysis (default: 1927)
    end_year : int
        End year for analysis (default: 2013)
    portfolio_type : str
        Portfolio type: 'size' or 'value' (default: 'size')
    rra : float
        Relative Risk Aversion coefficient (default: 4)
    output_suffix : str, optional
        Suffix to add to output filenames (e.g., '_1927_2024' or '_value')
    exclude_small_caps_flag : bool
        If True, exclude small cap portfolios (default: False)
    recentre_flag : bool
        If True, recentre returns to align with CAPM (default: False)
    """
    print("=" * 70)
    print("Portfolio Analysis for Assignment 3")
    print("=" * 70)
    print()
    
    print(f"Analysis period: {start_year}-{end_year}")
    print(f"Portfolio type: {portfolio_type}")
    print(f"CRRA RRA coefficient: {rra}")
    if exclude_small_caps_flag:
        print(f"Excluding small caps: Yes")
    if recentre_flag:
        print(f"Recentring data: Yes")
    print()
    
    # Load data
    print("Loading data...")
    try:
        returns = load_portfolio_data(portfolio_type, start_year, end_year)
        # Check if we're using annual data
        is_annual = 'Annual' in str(returns.index[0]) if len(returns) > 0 else False
        if not is_annual:
            # Check file name based on portfolio type
            if portfolio_type == "size":
                files = list(DATA_DIR.glob("*Portfolios_Formed_on_ME*Annual*.csv"))
            else:
                files = list(DATA_DIR.glob("*Portfolios_Formed_on_BE-ME*Annual*.csv"))
            is_annual = len(files) > 0
        
        factors = load_factors(start_year, end_year, prefer_annual=is_annual)
        
        # Check if we have data after filtering
        if len(returns) == 0:
            raise ValueError(f"No portfolio data found for period {start_year}-{end_year}")
        if len(factors) == 0:
            raise ValueError(f"No factor data found for period {start_year}-{end_year}")
        
        # Exclude small caps if requested
        if exclude_small_caps_flag:
            returns = exclude_small_caps(returns, portfolio_type)
        
        # Recentre data if requested
        if recentre_flag:
            if 'RF' in factors.columns:
                rf = factors['RF']
            else:
                rf = pd.Series(0, index=factors.index)
            print("  Recentring returns to align with CAPM...")
            returns = recentre_returns(returns, factors, rf)
        
        print(f"  Loaded {len(returns.columns)} portfolios")
        print(f"  Portfolio time period: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"  Factor time period: {factors.index[0].date()} to {factors.index[-1].date()}")
        print(f"  Data frequency: {'Annual' if is_annual else 'Monthly'}")
        
        # Check overlap
        common_dates = returns.index.intersection(factors.index)
        if len(common_dates) == 0:
            raise ValueError(f"No overlapping dates between portfolios and factors. "
                           f"Portfolio range: {returns.index[0]} to {returns.index[-1]}, "
                           f"Factor range: {factors.index[0]} to {factors.index[-1]}")
        print(f"  Common dates: {len(common_dates)} observations")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
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
    w_opt_full = pd.Series(0.0, index=returns.columns)
    w_opt_full[portfolio_names] = w_opt
    alphas_zb, betas_zb = estimate_jensens_alpha(returns, factors, rf, use_zero_beta=True, 
                                                  w_z=w_z_full.values, w_m=w_opt_full.values)
    
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
    if output_suffix:
        suffix = output_suffix
    else:
        suffix = f"_{portfolio_type}_{start_year}_{end_year}"
        if exclude_small_caps_flag:
            suffix += "_no_small_caps"
        if recentre_flag:
            suffix += "_recentred"
    
    w_msmp_full = pd.Series(0.0, index=returns.columns)
    w_msmp_full[portfolio_names] = w_msmp
    w_msmp_full.to_csv(OUTPUT_DIR / f"msmp_weights{suffix}.csv")
    
    w_opt_full = pd.Series(0.0, index=returns.columns)
    w_opt_full[portfolio_names] = w_opt
    w_opt_full.to_csv(OUTPUT_DIR / f"optimal_weights{suffix}.csv")
    
    w_z_full = pd.Series(0.0, index=returns.columns)
    w_z_full[portfolio_names] = w_z
    w_z_full.to_csv(OUTPUT_DIR / f"zero_beta_weights{suffix}.csv")
    alphas_zb.to_csv(OUTPUT_DIR / f"jensens_alpha_zerobeta{suffix}.csv")
    alphas_rf.to_csv(OUTPUT_DIR / f"jensens_alpha_rf{suffix}.csv")
    
    print(f"  Results saved to {OUTPUT_DIR}/")
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Portfolio Analysis for Assignment 3: Mean-Variance Portfolio Theory and the CAPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Size portfolios, 1927-2013 (default)
  python portfolio_analysis.py
  
  # Size portfolios, 1927-2024
  python portfolio_analysis.py --start-year 1927 --end-year 2024
  
  # Value (BE-ME) portfolios, 1927-2013
  python portfolio_analysis.py --portfolio-type value
  
  # Value portfolios, 1927-2024
  python portfolio_analysis.py --portfolio-type value --start-year 1927 --end-year 2024
  
  # Exclude small caps (optional step 8)
  python portfolio_analysis.py --exclude-small-caps
  
  # Recentre data (optional step 9)
  python portfolio_analysis.py --recentre
  
  # Both optional steps
  python portfolio_analysis.py --exclude-small-caps --recentre
        """
    )
    
    parser.add_argument(
        '--start-year', type=int, default=1927,
        help='Start year for analysis (default: 1927)'
    )
    parser.add_argument(
        '--end-year', type=int, default=2013,
        help='End year for analysis (default: 2013)'
    )
    parser.add_argument(
        '--portfolio-type', type=str, default='size',
        choices=['size', 'value'],
        help='Portfolio type: "size" for size portfolios, "value" for BE-ME portfolios (default: size)'
    )
    parser.add_argument(
        '--rra', type=float, default=4.0,
        help='Relative Risk Aversion coefficient for CRRA utility (default: 4.0)'
    )
    parser.add_argument(
        '--output-suffix', type=str, default=None,
        help='Optional suffix for output filenames (default: auto-generated from parameters)'
    )
    parser.add_argument(
        '--exclude-small-caps', action='store_true',
        help='Exclude small cap portfolios from analysis (optional, step 8)'
    )
    parser.add_argument(
        '--recentre', action='store_true',
        help='Recentre data to align means with CAPM and betas (optional, step 9)'
    )
    
    args = parser.parse_args()
    
    results = main(
        start_year=args.start_year,
        end_year=args.end_year,
        portfolio_type=args.portfolio_type,
        rra=args.rra,
        output_suffix=args.output_suffix,
        exclude_small_caps_flag=args.exclude_small_caps,
        recentre_flag=args.recentre
    )

