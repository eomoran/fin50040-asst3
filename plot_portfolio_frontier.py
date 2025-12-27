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


def load_zbp_data(portfolio_type, start_year, end_year, rra=4.0, allow_short_selling=True):
    """
    Load Zero-Beta Portfolio data from summary CSV
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    rra : float
        Relative Risk Aversion coefficient for optimal portfolio (default: 4.0)
    allow_short_selling : bool
        Whether short selling was allowed
    
    Returns:
    --------
    zbp_summary : pd.Series
        Summary row from zbp_summary.csv, or None if not found
    """
    summary_file = RESULTS_DIR / "zbp_summary.csv"
    if not summary_file.exists():
        print(f"  ZBP summary file not found: {summary_file}")
        return None
    
    print(f"  Loading ZBP summary from: {summary_file}")
    summary_df = pd.read_csv(summary_file)
    
    # Drop 'on_frontier' column if it exists (backward compatibility)
    if 'on_frontier' in summary_df.columns:
        summary_df = summary_df.drop(columns=['on_frontier'])
    
    # Find matching entry
    mask = (
        (summary_df['portfolio_type'] == portfolio_type) &
        (summary_df['start_year'] == start_year) &
        (summary_df['end_year'] == end_year) &
        (summary_df['rra'] == rra) &
        (summary_df['allow_short_selling'] == allow_short_selling)
    )
    
    if not mask.any():
        print(f"  No matching ZBP entry found in summary")
        return None
    
    return summary_df[mask].iloc[0]


def load_risk_free_rate(start_year, end_year):
    """
    Load risk-free rate from Fama-French factors
    
    Parameters:
    -----------
    start_year : int
        Start year
    end_year : int
        End year
    
    Returns:
    --------
    rf_mean : float
        Mean risk-free rate (gross return R_f = 1 + r_f)
    """
    from pathlib import Path
    DATA_DIR = Path("data/processed")
    
    # Find factors file
    files = list(DATA_DIR.glob("*F-F_Research_Data_Factors*Annual*.csv"))
    if not files:
        return None
    
    factors = pd.read_csv(files[0], index_col=0, parse_dates=True)
    factors = factors.loc[f'{start_year}-01-01':f'{end_year}-01-01']
    factors = factors.dropna()
    
    if 'RF' not in factors.columns:
        return None
    
    # RF is in gross returns (R_f = 1 + r_f)
    rf_mean = factors['RF'].mean()
    return rf_mean


def plot_line_from_rf(ax, rf, portfolio_vol, portfolio_return, max_vol, reflect=False, 
                      color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=None, zorder=3):
    """
    Plot a line from risk-free rate (on y-axis) to a portfolio point
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    rf : float
        Risk-free rate (gross return, on y-axis at x=0)
    portfolio_vol : float
        Portfolio volatility (x-coordinate)
    portfolio_return : float
        Portfolio return (y-coordinate)
    reflect : bool
        If True, reflect the line along y=rf (for MSMP)
    color : str
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    alpha : float
        Transparency
    label : str
        Label for legend
    """
    if reflect:
        # Reflect: if portfolio is at (σ, R), reflect to (σ, 2*R_f - R)
        # The line goes from (0, R_f) to (σ, 2*R_f - R), then extend to frame edge
        reflected_return = 2 * rf - portfolio_return
        # Calculate slope from (0, R_f) to (σ, reflected_return)
        if portfolio_vol > 1e-10:
            slope = (reflected_return - rf) / portfolio_vol
        else:
            slope = 0
        # Extend to frame edge
        x_line = np.array([0.0, max_vol])
        y_line = np.array([rf, rf + slope * max_vol])
    else:
        # Normal line from (0, R_f) to (σ, R), then extend to frame edge
        if portfolio_vol > 1e-10:
            slope = (portfolio_return - rf) / portfolio_vol
        else:
            slope = 0
        # Extend to frame edge
        x_line = np.array([0.0, max_vol])
        y_line = np.array([rf, rf + slope * max_vol])
    
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, 
           linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)


def plot_portfolio_frontier(ios_df, ios_summary, msmp_summary=None, optimal_crra_summary=None,
                      zbp_summary=None, portfolio_type=None, start_year=None, end_year=None,
                      figsize=(12, 8)):
    """
    Plot Investment Opportunity Set curve with key portfolios and tangent lines
    
    Parameters:
    -----------
    ios_df : pd.DataFrame
        IOS curve data with return_gross, return_net, volatility columns
    ios_summary : pd.Series
        IOS summary row
    msmp_summary : pd.Series or None
        MSMP summary row (optional)
    optimal_crra_summary : pd.Series or None
        Optimal CRRA summary row (optional)
    zbp_summary : pd.Series or None
        Zero-Beta Portfolio summary row (optional)
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
    # Inefficient: return < MVP return AND volatility > MVP volatility
    # Efficient: everything else (including MVP itself)
    inefficient_mask = (ios_df['return_gross'] < mvp_return) & (ios_df['volatility'] > mvp_vol)
    efficient_mask = ~inefficient_mask
    
    # Sort each limb separately
    # Inefficient: sort by volatility DESCENDING (from highest vol down to MVP)
    inefficient_df = ios_df[inefficient_mask].sort_values('volatility', ascending=False)
    # Efficient: sort by volatility ASCENDING (from MVP up)
    efficient_df = ios_df[efficient_mask].sort_values('volatility', ascending=True)
    
    # Combine: inefficient -> MVP -> efficient
    # This creates the full U-shape with smooth connections
    if len(inefficient_df) > 0:
        # Ensure smooth connection: inefficient limb should end close to MVP
        # The last inefficient point should have volatility just above MVP
        # We'll include MVP in the concatenation to ensure smooth connection
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
        print(f"  Plotting IOS: {len(inefficient_df)} inefficient + 1 MVP + {len(efficient_df)} efficient = {len(frontier_vols)} total points")
        print(f"  Inefficient limb: vol range [{inefficient_df['volatility'].min():.4f}, {inefficient_df['volatility'].max():.4f}], return range [{inefficient_df['return_gross'].min():.4f}, {inefficient_df['return_gross'].max():.4f}]")
    else:
        # If no inefficient limb found, just plot efficient limb with MVP
        print(f"  Warning: No inefficient limb found! Only plotting efficient limb.")
        frontier_vols = np.concatenate([
            [mvp_vol],
            efficient_df['volatility'].values
        ])
        frontier_returns = np.concatenate([
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
    
    # Plot Zero-Beta Portfolio if available
    if zbp_summary is not None:
        zbp_return = zbp_summary['expected_return_gross']
        zbp_vol = zbp_summary['volatility']
        ax.scatter([zbp_vol], [zbp_return],
                  c='orange', s=200, marker='s', edgecolors='black', linewidths=1.5,
                  label=f'ZBP (R={zbp_return:.4f}, σ={zbp_vol:.2%})', zorder=5)
    
    # Calculate axis limits first (needed for tangent lines)
    min_vol = 0.0
    max_vol = frontier_vols.max() * 1.1
    min_ret = 0.0
    max_ret = frontier_returns.max() * 1.1
    
    # Extend if MSMP, optimal CRRA, or ZBP is outside current range
    if msmp_summary is not None:
        max_vol = max(max_vol, msmp_summary['volatility'] * 1.1)
        max_ret = max(max_ret, msmp_summary['expected_return_gross'] * 1.1)
        min_ret = min(min_ret, msmp_summary['expected_return_gross'] * 1.1)
    
    if optimal_crra_summary is not None:
        max_vol = max(max_vol, optimal_crra_summary['volatility'] * 1.1)
        max_ret = max(max_ret, optimal_crra_summary['expected_return_gross'] * 1.1)
        min_ret = min(min_ret, optimal_crra_summary['expected_return_gross'] * 1.1)
    
    if zbp_summary is not None:
        max_vol = max(max_vol, zbp_summary['volatility'] * 1.1)
        max_ret = max(max_ret, zbp_summary['expected_return_gross'] * 1.1)
        min_ret = min(min_ret, zbp_summary['expected_return_gross'] * 1.1)
    
    # Load risk-free rate
    rf = None
    if start_year and end_year:
        rf = load_risk_free_rate(start_year, end_year)
        if rf is not None:
            print(f"  Risk-free rate (mean): R_f = {rf:.6f} (net: {(rf-1)*100:.4f}%)")
    
    # Plot lines from R_f to portfolios
    # These represent the convex space from allocating between long/short positions
    # in the portfolio and the risk-free asset
    
    # Line from R_f to Optimal CRRA Portfolio (black dashed)
    if optimal_crra_summary is not None and rf is not None:
        opt_return = optimal_crra_summary['expected_return_gross']
        opt_vol = optimal_crra_summary['volatility']
        plot_line_from_rf(ax, rf, opt_vol, opt_return, max_vol, reflect=False, 
                          color='black', linestyle='--', label='R_f ↔ Optimal CRRA', zorder=3)
    
    # Line from R_f to ZBP (black dashed)
    if zbp_summary is not None and rf is not None:
        zbp_return = zbp_summary['expected_return_gross']
        zbp_vol = zbp_summary['volatility']
        plot_line_from_rf(ax, rf, zbp_vol, zbp_return, max_vol, reflect=False,
                          color='black', linestyle='--', label='R_f ↔ ZBP', zorder=3)
    
    # Lines from R_f to MSMP: both normal and reflected (red solid)
    if msmp_summary is not None and rf is not None:
        msmp_return = msmp_summary['expected_return_gross']
        msmp_vol = msmp_summary['volatility']
        # Normal line from R_f to MSMP
        plot_line_from_rf(ax, rf, msmp_vol, msmp_return, max_vol, reflect=False,
                          color='r', linestyle='-', label='R_f ↔ MSMP', zorder=3)
        # Reflected line from R_f to MSMP (reflected along y=R_f)
        plot_line_from_rf(ax, rf, msmp_vol, msmp_return, max_vol, reflect=True,
                          color='r', linestyle='-', label='R_f ↔ MSMP (reflected)', zorder=3)
    
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
        if zbp_summary is not None:
            title_parts.append('ZBP')
        title = f"{', '.join(title_parts)}\n{portfolio_type.upper()} Portfolios ({start_year}-{end_year})"
    else:
        title = 'Investment Opportunity Set'
        if msmp_summary is not None:
            title += ' and MSMP'
        if optimal_crra_summary is not None:
            title += ' and Optimal CRRA'
        if zbp_summary is not None:
            title += ' and ZBP'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Format axes: x-axis as percentages, y-axis as decimals (gross returns)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Set axis limits (already calculated above)
    ax.set_xlim(min_vol, max_vol)
    ax.set_ylim(min_ret, max_ret)
    
    # Label R_f on y-axis if available
    if rf is not None:
        ax.axhline(y=rf, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)
        ax.text(-0.02 * max_vol, rf, 'R_f', fontsize=10, ha='right', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Plot Investment Opportunity Set with key portfolios and tangent lines'
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
    
    # Load ZBP data
    print("\nLoading ZBP data...")
    print(f"  ZBP summary file: {RESULTS_DIR / 'zbp_summary.csv'}")
    zbp_summary = load_zbp_data(
        args.portfolio_type, args.start_year, args.end_year, args.rra, allow_short
    )
    if zbp_summary is not None:
        print(f"  Found ZBP: R={zbp_summary['expected_return_gross']:.6f}, "
              f"σ={zbp_summary['volatility']:.4%}")
    else:
        print(f"  No ZBP data found")
    
    # Plot
    print("\nGenerating plot...")
    fig, ax = plot_portfolio_frontier(
        ios_df, ios_summary, msmp_summary, optimal_crra_summary, zbp_summary,
        args.portfolio_type, args.start_year, args.end_year
    )
    
    # Save plot
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}{args.output_suffix}"
    filename = PLOTS_DIR / f'portfolio_frontier{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {filename}")
    
    filename_pdf = PLOTS_DIR / f'portfolio_frontier{suffix}.pdf'
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"  Saved plot to {filename_pdf}")
    
    plt.close(fig)
    
    print("\n✓ Plotting complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

