#!/usr/bin/env python3
"""
Compute and analyze ex-post pricing kernel from MSMP

This script:
1. Loads portfolio returns and MSMP weights
2. Computes MSMP return series
3. Constructs pricing kernel: m = R_msmp / E[R_msmp^2]
4. Verifies pricing properties: E[m * R_i] ≈ 1
5. Compares pricing kernel between 1927-2013 and 1927-2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# We'll load data directly from processed files to avoid import issues

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/processed")


def compute_pricing_kernel_from_returns(returns, w_msmp):
    """
    Compute pricing kernel (SDF) from portfolio gross returns and MSMP weights
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Portfolio gross returns (R = 1 + r, time x portfolios)
    w_msmp : pd.Series
        MSMP weights (indexed by portfolio names, computed using gross returns)
    
    Returns:
    --------
    m : pd.Series
        Pricing kernel time series
    p_m : float
        Price of SDF: E[m]
    E_R_msmp : float
        Expected gross return on MSMP
    E_R2_msmp : float
        Expected squared gross return on MSMP
    msmp_returns : pd.Series
        MSMP return series (gross returns R_msmp)
    """
    # Align weights with returns columns
    w_aligned = w_msmp.reindex(returns.columns, fill_value=0.0)
    
    # Compute MSMP gross return series: R_msmp,t = sum_i w_i * R_i,t
    # Since returns are already gross returns (R = 1 + r), we don't need to add 1
    R_msmp = (returns * w_aligned).sum(axis=1)
    
    # Compute second moment: E[R_msmp^2]
    E_R2_msmp = (R_msmp ** 2).mean()
    
    # Compute expected gross return: E[R_msmp]
    E_R_msmp = R_msmp.mean()
    
    # Pricing kernel: m = R_msmp / E[R_msmp^2]
    m = R_msmp / E_R2_msmp
    
    # Price of SDF: p(m) = E[m] = E[R_msmp] / E[R_msmp^2]
    p_m = m.mean()
    
    # Verify: should equal E[R_msmp] / E[R_msmp^2]
    p_m_alt = E_R_msmp / E_R2_msmp
    
    # msmp_returns should be gross returns for consistency
    msmp_returns = R_msmp
    
    return m, p_m, p_m_alt, E_R_msmp, E_R2_msmp, msmp_returns


def verify_pricing_properties(m, returns):
    """
    Verify that pricing kernel prices assets correctly: E[m * R_i] ≈ 1
    
    Parameters:
    -----------
    m : pd.Series
        Pricing kernel (indexed by time)
    returns : pd.DataFrame
        Portfolio returns (time x portfolios)
    
    Returns:
    --------
    pricing_errors : pd.Series
        E[m * R_i] - 1 for each portfolio (should be close to 0)
    """
    # Convert returns to gross returns
    R = 1 + returns
    
    # Align m with returns index
    m_aligned = m.reindex(R.index)
    
    # Compute E[m * R_i] for each portfolio
    E_mR = (m_aligned.values.reshape(-1, 1) * R.values).mean(axis=0)
    
    # Pricing error: should be close to 1
    pricing_errors = E_mR - 1
    
    return pd.Series(pricing_errors, index=returns.columns)


def load_portfolio_returns(portfolio_type, start_year, end_year):
    """Load portfolio returns from processed CSV files"""
    if portfolio_type == "size":
        pattern = "*Portfolios_Formed_on_ME*.csv"
    else:  # value
        pattern = "*Portfolios_Formed_on_BE-ME*.csv"
    
    files = list(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    
    # Prefer the annual value-weighted file (from January to December)
    annual_vw_files = [f for f in files if 'Annual' in f.name and 'Value' in f.name and 'January' in f.name and 'December' in f.name]
    if annual_vw_files:
        file_to_use = annual_vw_files[0]
    else:
        # Fallback to any annual value-weighted file
        annual_vw_files = [f for f in files if 'Annual' in f.name and ('Value' in f.name or 'Weight' in f.name) and 'Equal' not in f.name]
        if annual_vw_files:
            file_to_use = annual_vw_files[0]
        else:
            file_to_use = files[0]
    
    print(f"  Using file: {file_to_use.name}")
    df = pd.read_csv(file_to_use, index_col=0)
    
    # Parse dates - handle various formats
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try parsing - pandas will auto-detect common formats like YYYY-MM-DD
        df.index = pd.to_datetime(df.index.astype(str), errors='coerce')
    
    df = df[df.index.notna()]
    
    # Filter date range - use year component for annual data
    if len(df) > 0:
        df_years = df.index.year
        mask = (df_years >= start_year) & (df_years <= end_year)
        df = df[mask]
        print(f"  Loaded {len(df)} observations for {start_year}-{end_year}")
    else:
        print(f"  Warning: No data after date parsing and filtering")
    
    # Convert to numeric, coercing errors
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Processed files already contain gross returns (R = 1 + r) in decimal form
    # No conversion needed - data was converted during processing
    return df


def analyze_pricing_kernel(portfolio_type, start_year, end_year):
    """Analyze pricing kernel for a given configuration"""
    print(f"\n{'='*70}")
    print(f"{portfolio_type.upper()} Portfolios ({start_year}-{end_year})")
    print(f"{'='*70}")
    
    # Load portfolio returns
    returns = load_portfolio_returns(portfolio_type, start_year, end_year)
    
    # Load MSMP weights from saved file
    suffix = f"_{portfolio_type}_{start_year}_{end_year}"
    msmp_file = RESULTS_DIR / f"msmp_weights{suffix}.csv"
    
    if not msmp_file.exists():
        raise FileNotFoundError(f"MSMP weights file not found: {msmp_file}")
    
    w_msmp_full = pd.read_csv(msmp_file, index_col=0).iloc[:, 0]
    
    # Compute pricing kernel
    m, p_m, p_m_alt, E_R_msmp, E_R2_msmp, msmp_returns = compute_pricing_kernel_from_returns(
        returns, w_msmp_full
    )
    
    # Verify pricing properties
    pricing_errors = verify_pricing_properties(m, returns)
    
    # Compute statistics
    # msmp_returns are gross returns (R)
    msmp_gross_return = msmp_returns.mean()  # Mean gross return E[R]
    msmp_net_return = msmp_gross_return - 1  # Mean net return E[r] = E[R] - 1
    msmp_vol = msmp_returns.std()  # Volatility (same for net and gross)
    
    print(f"\nMSMP Statistics:")
    print(f"  Expected return (gross, annualized): {msmp_gross_return:.6f}")
    print(f"  Expected return (net, annualized): {msmp_net_return:.6f}")
    print(f"  Volatility (annualized): {msmp_vol:.6f}")
    print(f"  Expected gross return E[R_msmp]: {E_R_msmp:.6f}")
    print(f"  Expected squared return E[R_msmp^2]: {E_R2_msmp:.6f}")
    
    print(f"\nPricing Kernel Statistics:")
    print(f"  Price of SDF p(m) = E[m]: {p_m:.6f}")
    print(f"  Price of SDF (alt) = E[R]/E[R^2]: {p_m_alt:.6f}")
    print(f"  Mean of m: {m.mean():.6f}")
    print(f"  Std of m: {m.std():.6f}")
    print(f"  Min of m: {m.min():.6f}")
    print(f"  Max of m: {m.max():.6f}")
    print(f"  Observations: {len(m)}")
    
    print(f"\nPricing Property Verification (E[m * R_i] should ≈ 1):")
    print(f"  Mean pricing error: {pricing_errors.mean():.6e}")
    print(f"  Std of pricing errors: {pricing_errors.std():.6e}")
    print(f"  Max absolute error: {pricing_errors.abs().max():.6e}")
    print(f"  Portfolios with |error| < 1e-6: {(pricing_errors.abs() < 1e-6).sum()}/{len(pricing_errors)}")
    print(f"  Portfolios with |error| < 1e-4: {(pricing_errors.abs() < 1e-4).sum()}/{len(pricing_errors)}")
    print(f"  Portfolios with |error| < 1e-2: {(pricing_errors.abs() < 1e-2).sum()}/{len(pricing_errors)}")
    
    return {
        'portfolio_type': portfolio_type,
        'start_year': start_year,
        'end_year': end_year,
        'msmp_return': msmp_net_return,
        'msmp_vol': msmp_vol,
        'E_R_msmp': E_R_msmp,
        'E_R2_msmp': E_R2_msmp,
        'p_m': p_m,
        'm_mean': m.mean(),
        'm_std': m.std(),
        'm_min': m.min(),
        'm_max': m.max(),
        'pricing_error_mean': pricing_errors.mean(),
        'pricing_error_std': pricing_errors.std(),
        'pricing_error_max': pricing_errors.abs().max(),
        'n_obs': len(m),
        'm': m,
        'pricing_errors': pricing_errors
    }


def main():
    """Run pricing kernel analysis for all configurations"""
    print("=" * 70)
    print("Ex-Post Pricing Kernel Analysis")
    print("=" * 70)
    
    results = []
    
    # Analyze both portfolio types and both periods
    for portfolio_type in ['size', 'value']:
        for start_year, end_year in [(1927, 2013), (1927, 2024)]:
            try:
                result = analyze_pricing_kernel(portfolio_type, start_year, end_year)
                results.append(result)
            except Exception as e:
                print(f"\nError analyzing {portfolio_type} {start_year}-{end_year}: {e}")
                import traceback
                traceback.print_exc()
    
    # Compare periods
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("Period Comparison: 1927-2013 vs 1927-2024")
        print("=" * 70)
        
        for portfolio_type in ['size', 'value']:
            result_2013 = next((r for r in results if r['portfolio_type'] == portfolio_type and r['end_year'] == 2013), None)
            result_2024 = next((r for r in results if r['portfolio_type'] == portfolio_type and r['end_year'] == 2024), None)
            
            if result_2013 and result_2024:
                print(f"\n{portfolio_type.upper()} Portfolios:")
                print(f"  MSMP return: {result_2013['msmp_return']:.6f} → {result_2024['msmp_return']:.6f} (change: {result_2024['msmp_return'] - result_2013['msmp_return']:+.6f})")
                print(f"  MSMP volatility: {result_2013['msmp_vol']:.6f} → {result_2024['msmp_vol']:.6f} (change: {result_2024['msmp_vol'] - result_2013['msmp_vol']:+.6f})")
                print(f"  Price of SDF p(m): {result_2013['p_m']:.6f} → {result_2024['p_m']:.6f} (change: {result_2024['p_m'] - result_2013['p_m']:+.6f})")
                print(f"  Pricing kernel std: {result_2013['m_std']:.6f} → {result_2024['m_std']:.6f} (change: {result_2024['m_std'] - result_2013['m_std']:+.6f})")
                print(f"  Mean pricing error: {result_2013['pricing_error_mean']:.6e} → {result_2024['pricing_error_mean']:.6e}")
                print(f"  Max pricing error: {result_2013['pricing_error_max']:.6e} → {result_2024['pricing_error_max']:.6e}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()

