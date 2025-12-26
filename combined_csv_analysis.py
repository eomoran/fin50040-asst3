#!/usr/bin/env python3
"""
Analyze combined CSV results from portfolio analysis

This script provides summary statistics and insights from the combined
result files, including comparisons between recentred vs non-recentred data,
and small cap inclusion vs exclusion.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/combined")


def analyze_jensens_alpha():
    """Analyze Jensen's alpha results"""
    print("=" * 70)
    print("Jensen's Alpha Analysis")
    print("=" * 70)
    
    # Load combined results
    rf_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_rf_combined.csv')
    zb_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_zerobeta_combined.csv')
    
    print('\n=== Jensen\'s Alpha (Risk-Free CAPM) ===')
    print('\nRecentred = False:')
    rf_not_recentred = rf_df[rf_df['recentred'] == False]
    print(f'  Mean alpha: {rf_not_recentred["jensens_alpha_rf"].mean():.6f}')
    print(f'  Std alpha: {rf_not_recentred["jensens_alpha_rf"].std():.6f}')
    print(f'  Min: {rf_not_recentred["jensens_alpha_rf"].min():.6f}')
    print(f'  Max: {rf_not_recentred["jensens_alpha_rf"].max():.6f}')
    print(f'  Count: {len(rf_not_recentred)}')
    
    print('\nRecentred = True:')
    rf_recentred = rf_df[rf_df['recentred'] == True]
    print(f'  Mean alpha: {rf_recentred["jensens_alpha_rf"].mean():.6f}')
    print(f'  Std alpha: {rf_recentred["jensens_alpha_rf"].std():.6f}')
    print(f'  Min: {rf_recentred["jensens_alpha_rf"].min():.6f}')
    print(f'  Max: {rf_recentred["jensens_alpha_rf"].max():.6f}')
    print(f'  Count: {len(rf_recentred)}')
    print(f'  Count of essentially zero (< 1e-10): {(np.abs(rf_recentred["jensens_alpha_rf"]) < 1e-10).sum()}/{len(rf_recentred)}')
    
    print('\n=== Jensen\'s Alpha (Zero-Beta CAPM) ===')
    print('\nRecentred = False:')
    zb_not_recentred = zb_df[zb_df['recentred'] == False]
    print(f'  Mean alpha: {zb_not_recentred["jensens_alpha_zerobeta"].mean():.6f}')
    print(f'  Std alpha: {zb_not_recentred["jensens_alpha_zerobeta"].std():.6f}')
    print(f'  Min: {zb_not_recentred["jensens_alpha_zerobeta"].min():.6f}')
    print(f'  Max: {zb_not_recentred["jensens_alpha_zerobeta"].max():.6f}')
    print(f'  Count: {len(zb_not_recentred)}')
    
    print('\nRecentred = True:')
    zb_recentred = zb_df[zb_df['recentred'] == True]
    print(f'  Mean alpha: {zb_recentred["jensens_alpha_zerobeta"].mean():.6f}')
    print(f'  Std alpha: {zb_recentred["jensens_alpha_zerobeta"].std():.6f}')
    print(f'  Min: {zb_recentred["jensens_alpha_zerobeta"].min():.6f}')
    print(f'  Max: {zb_recentred["jensens_alpha_zerobeta"].max():.6f}')
    print(f'  Count: {len(zb_recentred)}')
    
    return rf_df, zb_df


def analyze_small_caps():
    """Analyze small cap inclusion vs exclusion"""
    print("\n" + "=" * 70)
    print("Small Cap Inclusion/Exclusion Analysis")
    print("=" * 70)
    
    # Load combined results
    rf_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_rf_combined.csv')
    zb_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_zerobeta_combined.csv')
    
    # Filter to size portfolios only (small cap exclusion only applies to size)
    rf_size = rf_df[rf_df['portfolio_type'] == 'size']
    zb_size = zb_df[zb_df['portfolio_type'] == 'size']
    
    print('\n=== Size Portfolios Only ===')
    
    # Compare with and without small caps (non-recentred)
    print('\nRisk-Free CAPM (Non-Recentred):')
    rf_with_small = rf_size[(rf_size['exclude_small_caps'] == False) & 
                            (rf_size['recentred'] == False)]
    rf_no_small = rf_size[(rf_size['exclude_small_caps'] == True) & 
                          (rf_size['recentred'] == False)]
    
    print(f'  With small caps:')
    print(f'    Mean alpha: {rf_with_small["jensens_alpha_rf"].mean():.6f}')
    print(f'    Std alpha: {rf_with_small["jensens_alpha_rf"].std():.6f}')
    print(f'    Count: {len(rf_with_small)}')
    
    print(f'  Without small caps:')
    print(f'    Mean alpha: {rf_no_small["jensens_alpha_rf"].mean():.6f}')
    print(f'    Std alpha: {rf_no_small["jensens_alpha_rf"].std():.6f}')
    print(f'    Count: {len(rf_no_small)}')
    
    print('\nZero-Beta CAPM (Non-Recentred):')
    zb_with_small = zb_size[(zb_size['exclude_small_caps'] == False) & 
                            (zb_size['recentred'] == False)]
    zb_no_small = zb_size[(zb_size['exclude_small_caps'] == True) & 
                          (zb_size['recentred'] == False)]
    
    print(f'  With small caps:')
    print(f'    Mean alpha: {zb_with_small["jensens_alpha_zerobeta"].mean():.6f}')
    print(f'    Std alpha: {zb_with_small["jensens_alpha_zerobeta"].std():.6f}')
    print(f'    Count: {len(zb_with_small)}')
    
    print(f'  Without small caps:')
    print(f'    Mean alpha: {zb_no_small["jensens_alpha_zerobeta"].mean():.6f}')
    print(f'    Std alpha: {zb_no_small["jensens_alpha_zerobeta"].std():.6f}')
    print(f'    Count: {len(zb_no_small)}')
    
    # Compare portfolio counts
    print('\n=== Portfolio Count Comparison ===')
    portfolios_with = set(rf_with_small['portfolio'].unique())
    portfolios_without = set(rf_no_small['portfolio'].unique())
    excluded = portfolios_with - portfolios_without
    
    print(f'  Portfolios with small caps: {len(portfolios_with)}')
    print(f'  Portfolios without small caps: {len(portfolios_without)}')
    print(f'  Excluded portfolios: {sorted(excluded)}')
    
    return rf_size, zb_size


def analyze_by_period():
    """Analyze differences between 1927-2013 and 1927-2024 periods"""
    print("\n" + "=" * 70)
    print("Period Comparison (1927-2013 vs 1927-2024)")
    print("=" * 70)
    
    rf_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_rf_combined.csv')
    zb_df = pd.read_csv(RESULTS_DIR / 'jensens_alpha_zerobeta_combined.csv')
    
    # Non-recentred only for meaningful comparison
    rf_non_recentred = rf_df[rf_df['recentred'] == False]
    zb_non_recentred = zb_df[zb_df['recentred'] == False]
    
    print('\n=== Risk-Free CAPM ===')
    # Handle both string and numeric year columns
    period_2013 = rf_non_recentred[
        (rf_non_recentred['start_year'].astype(str) == '1927') &
        (rf_non_recentred['end_year'].astype(str) == '2013')
    ]
    period_2024 = rf_non_recentred[
        (rf_non_recentred['start_year'].astype(str) == '1927') &
        (rf_non_recentred['end_year'].astype(str) == '2024')
    ]
    
    print('\n1927-2013:')
    if len(period_2013) > 0:
        print(f'  Mean alpha: {period_2013["jensens_alpha_rf"].mean():.6f}')
        print(f'  Std alpha: {period_2013["jensens_alpha_rf"].std():.6f}')
        print(f'  Count: {len(period_2013)}')
    else:
        print('  No data found')
    
    print('\n1927-2024:')
    if len(period_2024) > 0:
        print(f'  Mean alpha: {period_2024["jensens_alpha_rf"].mean():.6f}')
        print(f'  Std alpha: {period_2024["jensens_alpha_rf"].std():.6f}')
        print(f'  Count: {len(period_2024)}')
    else:
        print('  No data found')
    
    print('\n=== Zero-Beta CAPM ===')
    zb_period_2013 = zb_non_recentred[
        (zb_non_recentred['start_year'].astype(str) == '1927') &
        (zb_non_recentred['end_year'].astype(str) == '2013')
    ]
    zb_period_2024 = zb_non_recentred[
        (zb_non_recentred['start_year'].astype(str) == '1927') &
        (zb_non_recentred['end_year'].astype(str) == '2024')
    ]
    
    print('\n1927-2013:')
    if len(zb_period_2013) > 0:
        print(f'  Mean alpha: {zb_period_2013["jensens_alpha_zerobeta"].mean():.6f}')
        print(f'  Std alpha: {zb_period_2013["jensens_alpha_zerobeta"].std():.6f}')
        print(f'  Count: {len(zb_period_2013)}')
    else:
        print('  No data found')
    
    print('\n1927-2024:')
    if len(zb_period_2024) > 0:
        print(f'  Mean alpha: {zb_period_2024["jensens_alpha_zerobeta"].mean():.6f}')
        print(f'  Std alpha: {zb_period_2024["jensens_alpha_zerobeta"].std():.6f}')
        print(f'  Count: {len(zb_period_2024)}')
    else:
        print('  No data found')


def main():
    """Run all analyses"""
    print("=" * 70)
    print("Combined CSV Results Analysis")
    print("=" * 70)
    
    # Analyze Jensen's alpha
    rf_df, zb_df = analyze_jensens_alpha()
    
    # Analyze small cap inclusion/exclusion
    rf_size, zb_size = analyze_small_caps()
    
    # Analyze by period
    analyze_by_period()
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

