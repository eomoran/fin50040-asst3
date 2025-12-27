#!/usr/bin/env python3
"""
Analyze ex-post pricing kernel and MSMP comparison

This script:
1. Compares MSMP weights and returns between 1927-2013 and 1927-2024
2. Computes the pricing kernel (SDF) from MSMP: m = R_msmp / E[R_msmp^2]
3. Analyzes pricing kernel performance in both periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results/combined")
DATA_DIR = Path("data/processed")


def load_msmp_returns(portfolio_type, start_year, end_year, exclude_small_caps=False, recentred=False):
    """Load portfolio returns and compute MSMP return series"""
    from portfolio_analysis import load_portfolio_data, load_factors, exclude_small_caps as exclude_func, recentre_returns, compute_moments, find_msmp
    
    # Load data
    returns = load_portfolio_data(portfolio_type, start_year, end_year)
    is_annual = 'Annual' in str(returns.index[0]) if len(returns) > 0 else False
    if not is_annual:
        if portfolio_type == "size":
            files = list(DATA_DIR.glob("*Portfolios_Formed_on_ME*Annual*.csv"))
        else:
            files = list(DATA_DIR.glob("*Portfolios_Formed_on_BE-ME*Annual*.csv"))
        is_annual = len(files) > 0
    
    factors = load_factors(start_year, end_year, prefer_annual=is_annual)
    
    # Apply filters
    if exclude_small_caps:
        returns = exclude_func(returns, portfolio_type)
    
    if recentred:
        if 'RF' in factors.columns:
            rf = factors['RF']
        else:
            rf = pd.Series(0, index=factors.index)
        returns = recentre_returns(returns, factors, rf)
    
    # Compute MSMP
    mu, Sigma, portfolio_names = compute_moments(returns)
    w_msmp, msmp_return, msmp_vol = find_msmp(mu, Sigma)
    
    # Map weights to full portfolio set
    w_msmp_full = pd.Series(0.0, index=returns.columns)
    w_msmp_full[portfolio_names] = w_msmp
    
    # Compute MSMP return series
    msmp_returns = (returns * w_msmp_full).sum(axis=1)
    
    return msmp_returns, w_msmp_full, msmp_return, msmp_vol


def compute_pricing_kernel(msmp_returns):
    """
    Compute pricing kernel (SDF) from MSMP gross returns
    
    The pricing kernel is: m = R_msmp / E[R_msmp^2]
    where R_msmp is the gross return on the MSMP portfolio
    
    The price of the SDF is: p(m) = E[m] = E[R_msmp] / E[R_msmp^2]
    
    Parameters:
    -----------
    msmp_returns : pd.Series
        MSMP gross returns (R_msmp = 1 + r_msmp)
    """
    # msmp_returns are already gross returns (R) from load_msmp_returns
    R_msmp = msmp_returns
    
    # Compute E[R_msmp^2]
    E_R2 = (R_msmp ** 2).mean()
    
    # Pricing kernel: m = R_msmp / E[R_msmp^2]
    m = R_msmp / E_R2
    
    # Price of SDF: p(m) = E[m]
    p_m = m.mean()
    
    # Also compute E[R_msmp] for comparison
    E_R = R_msmp.mean()
    
    # Verify: p(m) should equal E[R_msmp] / E[R_msmp^2]
    p_m_alt = E_R / E_R2
    
    return m, p_m, p_m_alt, E_R, E_R2


def compare_msmp_periods():
    """Compare MSMP between 1927-2013 and 1927-2024"""
    print("=" * 70)
    print("MSMP Comparison: 1927-2013 vs 1927-2024")
    print("=" * 70)
    
    # Load MSMP weights
    msmp_df = pd.read_csv(RESULTS_DIR / 'msmp_weights_combined.csv')
    
    # Filter to non-recentred, non-excluded small caps for base comparison
    base_msmp = msmp_df[
        (msmp_df['recentred'] == False) & 
        (msmp_df['exclude_small_caps'] == False)
    ]
    
    print("\n=== MSMP Weights Comparison ===")
    for portfolio_type in ['size', 'value']:
        print(f"\n{portfolio_type.upper()} Portfolios:")
        
        period_2013 = base_msmp[
            (base_msmp['portfolio_type'] == portfolio_type) &
            (base_msmp['start_year'].astype(str) == '1927') &
            (base_msmp['end_year'].astype(str) == '2013')
        ]
        
        period_2024 = base_msmp[
            (base_msmp['portfolio_type'] == portfolio_type) &
            (base_msmp['start_year'].astype(str) == '1927') &
            (base_msmp['end_year'].astype(str) == '2024')
        ]
        
        # Merge on portfolio name
        merged = period_2013[['portfolio', 'msmp_weights']].merge(
            period_2024[['portfolio', 'msmp_weights']],
            on='portfolio',
            suffixes=('_2013', '_2024')
        )
        
        print(f"  Number of portfolios: {len(merged)}")
        print(f"  Weight correlation: {merged['msmp_weights_2013'].corr(merged['msmp_weights_2024']):.4f}")
        print(f"  Mean absolute difference: {np.abs(merged['msmp_weights_2013'] - merged['msmp_weights_2024']).mean():.4f}")
        print(f"  Max absolute difference: {np.abs(merged['msmp_weights_2013'] - merged['msmp_weights_2024']).max():.4f}")
        
        # Show top 5 differences
        merged['diff'] = np.abs(merged['msmp_weights_2013'] - merged['msmp_weights_2024'])
        top_diff = merged.nlargest(5, 'diff')[['portfolio', 'msmp_weights_2013', 'msmp_weights_2024', 'diff']]
        print(f"\n  Top 5 weight differences:")
        for _, row in top_diff.iterrows():
            print(f"    {row['portfolio']:10s}: 2013={row['msmp_weights_2013']:8.4f}, 2024={row['msmp_weights_2024']:8.4f}, diff={row['diff']:.4f}")


def analyze_pricing_kernel():
    """Analyze pricing kernel performance in both periods"""
    print("\n" + "=" * 70)
    print("Pricing Kernel (SDF) Analysis")
    print("=" * 70)
    
    print("\nThe pricing kernel (stochastic discount factor) is: m = R_msmp / E[R_msmp^2]")
    print("where R_msmp = 1 + r_msmp is the gross return on the MSMP portfolio")
    print("The price of the SDF is: p(m) = E[m] = E[R_msmp] / E[R_msmp^2]")
    print()
    
    results = []
    
    for portfolio_type in ['size', 'value']:
        for start_year, end_year in [(1927, 2013), (1927, 2024)]:
            try:
                msmp_returns, w_msmp, msmp_return, msmp_vol = load_msmp_returns(
                    portfolio_type, start_year, end_year, exclude_small_caps=False, recentred=False
                )
                
                m, p_m, p_m_alt, E_R, E_R2 = compute_pricing_kernel(msmp_returns)
                
                results.append({
                    'portfolio_type': portfolio_type,
                    'period': f'{start_year}-{end_year}',
                    'msmp_return': msmp_return,
                    'msmp_vol': msmp_vol,
                    'E_R_msmp': E_R - 1,  # Convert back to net return
                    'E_R2_msmp': E_R2,
                    'p_m': p_m,
                    'p_m_alt': p_m_alt,
                    'm_mean': m.mean(),
                    'm_std': m.std(),
                    'm_min': m.min(),
                    'm_max': m.max(),
                    'n_obs': len(msmp_returns)
                })
                
                print(f"\n{portfolio_type.upper()} Portfolios ({start_year}-{end_year}):")
                # msmp_return from find_msmp is now gross return E[R]
                msmp_net = msmp_return - 1
                print(f"  MSMP return (gross, annualized): {msmp_return:.4f}")
                print(f"  MSMP return (net, annualized): {msmp_net:.4f}")
                print(f"  MSMP volatility (annualized): {msmp_vol:.4f}")
                print(f"  E[R_msmp] (gross): {E_R:.6f}")
                print(f"  E[R_msmp^2]: {E_R2:.6f}")
                print(f"  Price of SDF p(m) = E[m]: {p_m:.6f}")
                print(f"  Price of SDF (alt) = E[R]/E[R^2]: {p_m_alt:.6f}")
                print(f"  Pricing kernel mean: {m.mean():.6f}")
                print(f"  Pricing kernel std: {m.std():.6f}")
                print(f"  Pricing kernel range: [{m.min():.6f}, {m.max():.6f}]")
                print(f"  Observations: {len(msmp_returns)}")
                
            except Exception as e:
                print(f"\nError analyzing {portfolio_type} {start_year}-{end_year}: {e}")
                import traceback
                traceback.print_exc()
    
    # Compare periods
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("Period Comparison")
        print("=" * 70)
        
        df_results = pd.DataFrame(results)
        
        for portfolio_type in ['size', 'value']:
            type_results = df_results[df_results['portfolio_type'] == portfolio_type]
            if len(type_results) == 2:
                period_2013 = type_results[type_results['period'] == '1927-2013'].iloc[0]
                period_2024 = type_results[type_results['period'] == '1927-2024'].iloc[0]
                
                print(f"\n{portfolio_type.upper()} Portfolios:")
                print(f"  MSMP return change: {period_2013['msmp_return']:.4f} → {period_2024['msmp_return']:.4f} (diff: {period_2024['msmp_return'] - period_2013['msmp_return']:+.4f})")
                print(f"  MSMP volatility change: {period_2013['msmp_vol']:.4f} → {period_2024['msmp_vol']:.4f} (diff: {period_2024['msmp_vol'] - period_2013['msmp_vol']:+.4f})")
                print(f"  Price of SDF change: {period_2013['p_m']:.6f} → {period_2024['p_m']:.6f} (diff: {period_2024['p_m'] - period_2013['p_m']:+.6f})")
                print(f"  Pricing kernel std change: {period_2013['m_std']:.6f} → {period_2024['m_std']:.6f} (diff: {period_2024['m_std'] - period_2013['m_std']:+.6f})")


def explain_52_count():
    """Explain why we have 52 portfolio-period combinations"""
    print("\n" + "=" * 70)
    print("Why 52 Portfolio-Period Combinations?")
    print("=" * 70)
    
    rf_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_rf_combined.csv')
    rf_non_recentred = rf_df[rf_df['recentred'] == False]
    
    for period_end in ['2013', '2024']:
        period_data = rf_non_recentred[
            (rf_non_recentred['start_year'].astype(str) == '1927') &
            (rf_non_recentred['end_year'].astype(str) == period_end)
        ]
        
        print(f"\n1927-{period_end}:")
        print(f"  Total observations: {len(period_data)}")
        
        config_counts = period_data.groupby(['portfolio_type', 'exclude_small_caps']).size()
        print(f"  Breakdown by configuration:")
        for (ptype, excl), count in config_counts.items():
            config_name = f"{ptype}"
            if excl:
                config_name += " (no small caps)"
            print(f"    {config_name}: {count} portfolios")
        
        # Verify: size (19) + value (19) + size_no_small_caps (14) = 52
        size_full = len(period_data[(period_data['portfolio_type'] == 'size') & 
                                 (period_data['exclude_small_caps'] == False)])
        value_full = len(period_data[(period_data['portfolio_type'] == 'value') & 
                                  (period_data['exclude_small_caps'] == False)])
        size_no_small = len(period_data[(period_data['portfolio_type'] == 'size') & 
                                       (period_data['exclude_small_caps'] == True)])
        
        print(f"  Verification: {size_full} (size) + {value_full} (value) + {size_no_small} (size no small caps) = {size_full + value_full + size_no_small}")


def main():
    """Run all analyses"""
    print("=" * 70)
    print("Ex-Post Pricing Kernel and MSMP Analysis")
    print("=" * 70)
    
    # Explain the 52 count
    explain_52_count()
    
    # Compare MSMP weights
    compare_msmp_periods()
    
    # Analyze pricing kernel
    analyze_pricing_kernel()
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

