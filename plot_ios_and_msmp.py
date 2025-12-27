#!/usr/bin/env python3
"""
Plot Investment Opportunity Set (IOS) and MSMP together

This script reads from the saved summary CSVs and IOS curve data
to plot both the IOS curve and MSMP point on the same figure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Directories
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_ios_data(portfolio_type, start_year, end_year, allow_short_selling=True):
    """
    Load IOS curve data from CSV
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    allow_short_selling : bool
        Whether short selling was allowed
    
    Returns:
    --------
    ios_df : pd.DataFrame
        DataFrame with return_gross, return_net, volatility columns
    ios_summary : pd.Series
        Summary row from ios_summary.csv
    """
    # Load summary to verify it exists
    summary_file = RESULTS_DIR / "ios_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    print(f"  Loading IOS summary from: {summary_file}")
    summary_df = pd.read_csv(summary_file)
    
    # Find matching entry
    mask = (
        (summary_df['portfolio_type'] == portfolio_type) &
        (summary_df['start_year'] == start_year) &
        (summary_df['end_year'] == end_year) &
        (summary_df['allow_short_selling'] == allow_short_selling)
    )
    
    if not mask.any():
        raise ValueError(
            f"No IOS data found for portfolio_type={portfolio_type}, "
            f"start_year={start_year}, end_year={end_year}, "
            f"allow_short_selling={allow_short_selling}"
        )
    
    ios_summary = summary_df[mask].iloc[0]
    
    # Load the actual IOS curve data
    suffix = f"_{portfolio_type}_{start_year}_{end_year}"
    ios_file = RESULTS_DIR / f"investment_opportunity_set{suffix}.csv"
    
    if not ios_file.exists():
        raise FileNotFoundError(f"IOS curve file not found: {ios_file}")
    
    print(f"  Loading IOS curve from: {ios_file}")
    ios_df = pd.read_csv(ios_file)
    
    return ios_df, ios_summary


def load_msmp_data(portfolio_type, start_year, end_year, allow_short_selling=True):
    """
    Load MSMP data from summary CSV
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    allow_short_selling : bool
        Whether short selling was allowed
    
    Returns:
    --------
    msmp_summary : pd.Series
        Summary row from msmp_summary.csv, or None if not found
    """
    summary_file = RESULTS_DIR / "msmp_summary.csv"
    if not summary_file.exists():
        print(f"  MSMP summary file not found: {summary_file}")
        return None
    
    print(f"  Loading MSMP summary from: {summary_file}")
    summary_df = pd.read_csv(summary_file)
    
    # Find matching entry
    mask = (
        (summary_df['portfolio_type'] == portfolio_type) &
        (summary_df['start_year'] == start_year) &
        (summary_df['end_year'] == end_year) &
        (summary_df['allow_short_selling'] == allow_short_selling)
    )
    
    if not mask.any():
        print(f"  No matching MSMP entry found in summary")
        return None
    
    return summary_df[mask].iloc[0]


def load_optimal_crra_data(portfolio_type, start_year, end_year, rra=4.0, allow_short_selling=True):
    """
    Load optimal CRRA portfolio data from summary CSV
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    rra : float
        Relative Risk Aversion coefficient (default: 4.0)
    allow_short_selling : bool
        Whether short selling was allowed
    
    Returns:
    --------
    optimal_crra_summary : pd.Series
        Summary row from optimal_crra_summary.csv, or None if not found
    """
    summary_file = RESULTS_DIR / "optimal_crra_summary.csv"
    if not summary_file.exists():
        print(f"  Optimal CRRA summary file not found: {summary_file}")
        return None
    
    print(f"  Loading optimal CRRA summary from: {summary_file}")
    summary_df = pd.read_csv(summary_file)
    
    # Find matching entry
    mask = (
        (summary_df['portfolio_type'] == portfolio_type) &
        (summary_df['start_year'] == start_year) &
        (summary_df['end_year'] == end_year) &
        (summary_df['rra'] == rra) &
        (summary_df['allow_short_selling'] == allow_short_selling)
    )
    
    if not mask.any():
        print(f"  No matching optimal CRRA entry found in summary")
        return None
    
    return summary_df[mask].iloc[0]


def plot_ios_and_msmp(ios_df, ios_summary, msmp_summary=None, optimal_crra_summary=None,
                      portfolio_type=None, start_year=None, end_year=None,
                      figsize=(12, 8)):
    """
    Plot Investment Opportunity Set curve and MSMP point
    
    Parameters:
    -----------
    ios_df : pd.DataFrame
        IOS curve data with return_gross, return_net, volatility columns
    ios_summary : pd.Series
        IOS summary row
    msmp_summary : pd.Series or None
        MSMP summary row (optional)
    portfolio_type : str
        Portfolio type for title
    start_year : int
        Start year for title
    end_year : int
        End year for title
    figsize : tuple
        Figure size
    """
    # Separate efficient and inefficient limbs for proper plotting
    # Find MVP (minimum volatility point)
    mvp_idx = ios_df['volatility'].idxmin()
    mvp_return = ios_df.loc[mvp_idx, 'return_gross']
    mvp_vol = ios_df.loc[mvp_idx, 'volatility']
    
    # Separate limbs
    inefficient_mask = (ios_df['return_gross'] < mvp_return) & (ios_df['volatility'] > mvp_vol)
    efficient_mask = ~inefficient_mask
    
    # Sort each limb separately
    inefficient_df = ios_df[inefficient_mask].sort_values('volatility', ascending=False)
    efficient_df = ios_df[efficient_mask].sort_values('volatility', ascending=True)
    
    # Combine: inefficient -> MVP -> efficient
    frontier_vols = np.concatenate([
        inefficient_df['volatility'].values,
        [mvp_vol],
        efficient_df['volatility'].values
    ])
    frontier_returns = np.concatenate([
        inefficient_df['return_gross'].values,
        [mvp_return],
        efficient_df['return_gross'].values
    ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot IOS curve (gross returns)
    ax.plot(frontier_vols, frontier_returns,
            'b-', linewidth=2, label='Investment Opportunity Set', alpha=0.7, zorder=2)
    
    # Plot MVP
    ax.scatter([mvp_vol], [mvp_return],
              c='purple', s=200, marker='^', edgecolors='black', linewidths=1.5,
              label=f'MVP (R={mvp_return:.4f}, σ={mvp_vol:.2%})', zorder=5)
    
    # Plot MSMP if available
    if msmp_summary is not None:
        msmp_return = msmp_summary['expected_return_gross']
        msmp_vol = msmp_summary['volatility']
        ax.scatter([msmp_vol], [msmp_return],
                  c='red', s=200, marker='*', edgecolors='black', linewidths=1.5,
                  label=f'MSMP (R={msmp_return:.4f}, σ={msmp_vol:.2%})', zorder=5)
    
    # Plot optimal CRRA portfolio if available
    if optimal_crra_summary is not None:
        opt_return = optimal_crra_summary['expected_return_gross']
        opt_vol = optimal_crra_summary['volatility']
        rra = optimal_crra_summary['rra']
        ax.scatter([opt_vol], [opt_return],
                  c='green', s=200, marker='o', edgecolors='black', linewidths=1.5,
                  label=f'Optimal CRRA (RRA={rra:.1f}, R={opt_return:.4f}, σ={opt_vol:.2%})', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('Volatility (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (R)', fontsize=12, fontweight='bold')
    
    # Title
    if portfolio_type and start_year and end_year:
        title_parts = ['Investment Opportunity Set']
        if msmp_summary is not None:
            title_parts.append('MSMP')
        if optimal_crra_summary is not None:
            title_parts.append('Optimal CRRA')
        title = f"{', '.join(title_parts)}\n{portfolio_type.upper()} Portfolios ({start_year}-{end_year})"
    else:
        title = 'Investment Opportunity Set'
        if msmp_summary is not None:
            title += ' and MSMP'
        if optimal_crra_summary is not None:
            title += ' and Optimal CRRA'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Format axes: x-axis as percentages, y-axis as decimals (gross returns)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Adjust limits: x and y mins fixed at zero, maxes with padding
    min_vol = 0.0
    max_vol = frontier_vols.max() * 1.1
    min_ret = 0.0
    max_ret = frontier_returns.max() * 1.1
    
    # Extend if MSMP or optimal CRRA is outside current range
    if msmp_summary is not None:
        max_vol = max(max_vol, msmp_summary['volatility'] * 1.1)
        max_ret = max(max_ret, msmp_summary['expected_return_gross'] * 1.1)
        min_ret = min(min_ret, msmp_summary['expected_return_gross'] * 1.1)
    
    if optimal_crra_summary is not None:
        max_vol = max(max_vol, optimal_crra_summary['volatility'] * 1.1)
        max_ret = max(max_ret, optimal_crra_summary['expected_return_gross'] * 1.1)
        min_ret = min(min_ret, optimal_crra_summary['expected_return_gross'] * 1.1)
    
    ax.set_xlim(min_vol, max_vol)
    ax.set_ylim(min_ret, max_ret)
    
    plt.tight_layout()
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Plot Investment Opportunity Set and MSMP together'
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
        '--no-short-selling', action='store_true',
        help='Use long-only portfolios (default: free portfolio with short selling)'
    )
    parser.add_argument(
        '--rra', type=float, default=4.0,
        help='Relative Risk Aversion coefficient for optimal CRRA portfolio (default: 4.0)'
    )
    parser.add_argument(
        '--output-suffix', type=str, default='',
        help='Suffix to add to output filename (e.g., "_test")'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Plotting Investment Opportunity Set and MSMP")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    allow_short = not args.no_short_selling
    print(f"Short selling: {'ALLOWED (free portfolio)' if allow_short else 'NOT ALLOWED (long-only)'}")
    print()
    
    # Load IOS data
    print("Loading IOS data...")
    print(f"  IOS summary file: {RESULTS_DIR / 'ios_summary.csv'}")
    try:
        ios_df, ios_summary = load_ios_data(
            args.portfolio_type, args.start_year, args.end_year, allow_short
        )
        print(f"  Loaded {len(ios_df)} frontier points")
    except Exception as e:
        print(f"  Error: {e}")
        return
    
    # Load MSMP data
    print("\nLoading MSMP data...")
    print(f"  MSMP summary file: {RESULTS_DIR / 'msmp_summary.csv'}")
    msmp_summary = load_msmp_data(
        args.portfolio_type, args.start_year, args.end_year, allow_short
    )
    if msmp_summary is not None:
        print(f"  Found MSMP: R={msmp_summary['expected_return_gross']:.6f}, "
              f"σ={msmp_summary['volatility']:.4%}")
    else:
        print("  No MSMP data found")
    
    # Load optimal CRRA data
    print("\nLoading optimal CRRA data...")
    print(f"  Optimal CRRA summary file: {RESULTS_DIR / 'optimal_crra_summary.csv'}")
    optimal_crra_summary = load_optimal_crra_data(
        args.portfolio_type, args.start_year, args.end_year, args.rra, allow_short
    )
    if optimal_crra_summary is not None:
        print(f"  Found optimal CRRA (RRA={args.rra}): R={optimal_crra_summary['expected_return_gross']:.6f}, "
              f"σ={optimal_crra_summary['volatility']:.4%}")
    else:
        print(f"  No optimal CRRA data found (RRA={args.rra})")
    
    # Plot
    print("\nGenerating plot...")
    fig, ax = plot_ios_and_msmp(
        ios_df, ios_summary, msmp_summary, optimal_crra_summary,
        args.portfolio_type, args.start_year, args.end_year
    )
    
    # Save plot
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    filename = PLOTS_DIR / f'ios_and_msmp{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {filename}")
    
    filename_pdf = PLOTS_DIR / f'ios_and_msmp{suffix}.pdf'
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"  Saved plot to {filename_pdf}")
    
    plt.close(fig)
    
    print("\n✓ Plotting complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

