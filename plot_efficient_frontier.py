#!/usr/bin/env python3
"""
Plot Efficient Frontier with Key Portfolios

This script visualizes:
1. The efficient frontier (mean-variance frontier)
2. Individual asset portfolios
3. MSMP portfolio
4. Optimal CRRA portfolio (RRA=4)
5. Zero-beta portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add parent directory to path to import portfolio_analysis functions
sys.path.insert(0, str(Path(__file__).parent))

from portfolio_analysis import (
    load_portfolio_data, load_factors, compute_moments,
    construct_mv_frontier, find_msmp, find_optimal_crra_portfolio,
    find_zero_beta_portfolio, exclude_small_caps, recentre_returns
)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def plot_efficient_frontier(portfolio_type="size", start_year=1927, end_year=2013,
                           exclude_small_caps_flag=False, recentre_flag=False,
                           show_individual_assets=True, figsize=(12, 8)):
    """
    Plot efficient frontier with key portfolios
    
    Parameters:
    -----------
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    exclude_small_caps_flag : bool
        Whether small caps were excluded
    recentre_flag : bool
        Whether data was recentred
    show_individual_assets : bool
        Whether to show individual asset portfolios
    figsize : tuple
        Figure size
    """
    # Load data
    returns = load_portfolio_data(portfolio_type, start_year, end_year)
    factors = load_factors(start_year, end_year, prefer_annual=True)
    
    # Apply filters
    if exclude_small_caps_flag and portfolio_type == "size":
        returns = exclude_small_caps(returns, portfolio_type)
    
    if recentre_flag:
        if 'RF' in factors.columns:
            rf = factors['RF']
        else:
            rf = pd.Series(0, index=factors.index)
        returns = recentre_returns(returns, factors, rf)
    
    # Compute moments
    mu, Sigma, portfolio_names = compute_moments(returns)
    
    # Construct investment opportunity set (both efficient and inefficient limbs)
    frontier = construct_mv_frontier(mu, Sigma, num_portfolios=200)  # More points for smoother curve
    
    # Sort frontier by volatility for proper plotting (U-shape)
    sort_idx = np.argsort(frontier['volatilities'])
    frontier['volatilities'] = frontier['volatilities'][sort_idx]
    frontier['returns'] = frontier['returns'][sort_idx]
    
    # Find minimum variance portfolio explicitly
    n = len(mu)
    ones = np.ones(n)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.linalg.pinv(Sigma)
    w_mvp = inv_Sigma @ ones / (ones.T @ inv_Sigma @ ones)
    mvp_return = mu.T @ w_mvp
    mvp_vol = np.sqrt(w_mvp.T @ Sigma @ w_mvp)
    
    # Find key portfolios
    w_msmp, msmp_return, msmp_vol = find_msmp(mu, Sigma)
    w_opt, opt_return, opt_vol = find_optimal_crra_portfolio(mu, Sigma, rra=4)
    w_z, z_return, z_vol = find_zero_beta_portfolio(mu, Sigma, w_opt)
    
    # Verify zero-beta constraint
    zero_beta_check = abs(w_z.T @ Sigma @ w_opt)
    if zero_beta_check > 1e-4:
        print(f"  Warning: Zero-beta constraint check: {zero_beta_check:.2e} (should be ~0)")
    
    # Convert gross returns to net returns for display
    frontier_returns_net = frontier['returns'] - 1
    mvp_return_net = mvp_return - 1
    msmp_return_net = msmp_return - 1
    opt_return_net = opt_return - 1
    z_return_net = z_return - 1
    
    # Individual asset returns and volatilities
    if show_individual_assets:
        asset_returns_net = mu - 1
        asset_vols = np.sqrt(np.diag(Sigma))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot efficient frontier
    ax.plot(frontier['volatilities'], frontier_returns_net, 
            'b-', linewidth=2, label='Efficient Frontier', alpha=0.7, zorder=2)
    
    # Plot minimum variance portfolio
    ax.scatter([mvp_vol], [mvp_return_net], 
              c='purple', s=150, marker='o', edgecolors='black', linewidths=1.5,
              label=f'MVP (R={mvp_return_net:.2%}, σ={mvp_vol:.2%})', zorder=4)
    
    # Plot individual assets (if requested)
    if show_individual_assets:
        ax.scatter(asset_vols, asset_returns_net, 
                  c='gray', s=50, alpha=0.5, label='Individual Assets', zorder=1)
    
    # Plot key portfolios with distinct markers
    ax.scatter([msmp_vol], [msmp_return_net], 
              c='red', s=200, marker='*', edgecolors='black', linewidths=1.5,
              label=f'MSMP (R={msmp_return_net:.2%}, σ={msmp_vol:.2%})', zorder=5)
    
    ax.scatter([opt_vol], [opt_return_net], 
              c='green', s=200, marker='D', edgecolors='black', linewidths=1.5,
              label=f'Optimal CRRA (R={opt_return_net:.2%}, σ={opt_vol:.2%})', zorder=5)
    
    ax.scatter([z_vol], [z_return_net], 
              c='orange', s=200, marker='s', edgecolors='black', linewidths=1.5,
              label=f'Zero-β Portfolio (R={z_return_net:.2%}, σ={z_vol:.2%})', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('Volatility (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (r)', fontsize=12, fontweight='bold')
    
    # Title
    title_parts = [f'{portfolio_type.upper()} Portfolios']
    title_parts.append(f'{start_year}-{end_year}')
    if exclude_small_caps_flag:
        title_parts.append('(No Small Caps)')
    if recentre_flag:
        title_parts.append('(Recentred)')
                ax.set_title('Investment Opportunity Set with Key Portfolios\n' + ' '.join(title_parts), 
                            fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    # Set axis limits to show full frontier with some padding
    # X-axis: from slightly below MVP vol to slightly above max vol
    x_min = min(frontier['volatilities'].min(), mvp_vol) * 0.95
    x_max = max(frontier['volatilities'].max(), 
                msmp_vol if not np.isnan(msmp_vol) else 0,
                opt_vol if not np.isnan(opt_vol) else 0,
                z_vol if not np.isnan(z_vol) else 0) * 1.05
    ax.set_xlim(x_min, x_max)
    
    # Y-axis: from slightly below min return to slightly above max return
    y_min = min(frontier_returns_net.min(), 
                mvp_return_net,
                msmp_return_net if not np.isnan(msmp_return_net) else 0,
                z_return_net if not np.isnan(z_return_net) else 0) * 1.1  # Extra padding for negative returns
    y_max = max(frontier_returns_net.max(), 
                opt_return_net if not np.isnan(opt_return_net) else 0) * 1.05
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save figure
    suffix_parts = [portfolio_type, str(start_year), str(end_year)]
    if exclude_small_caps_flag:
        suffix_parts.append('no_small_caps')
    if recentre_flag:
        suffix_parts.append('recentred')
    suffix = '_'.join(suffix_parts)
    
    filename = PLOTS_DIR / f'efficient_frontier_{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {filename}")
    
    # Also save as PDF for high quality
    filename_pdf = PLOTS_DIR / f'efficient_frontier_{suffix}.pdf'
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"  Saved plot to {filename_pdf}")
    
    return fig, ax


def plot_all_configurations():
    """Plot efficient frontiers for all main configurations"""
    configs = [
        ('size', 1927, 2013, False, False),
        ('size', 1927, 2024, False, False),
        ('value', 1927, 2013, False, False),
        ('value', 1927, 2024, False, False),
    ]
    
    print("=" * 70)
    print("Generating Efficient Frontier Plots")
    print("=" * 70)
    
    for portfolio_type, start_year, end_year, exclude_small, recentre in configs:
        print(f"\nPlotting: {portfolio_type.upper()} {start_year}-{end_year}")
        try:
            plot_efficient_frontier(
                portfolio_type=portfolio_type,
                start_year=start_year,
                end_year=end_year,
                exclude_small_caps_flag=exclude_small,
                recentre_flag=recentre
            )
            plt.close()  # Close figure to free memory
        except Exception as e:
            print(f"  Error plotting {portfolio_type} {start_year}-{end_year}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All plots generated!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot efficient frontier with key portfolios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot size portfolios, 1927-2013 (default)
  python plot_efficient_frontier.py
  
  # Plot value portfolios, 1927-2024
  python plot_efficient_frontier.py --portfolio-type value --start-year 1927 --end-year 2024
  
  # Plot all main configurations
  python plot_efficient_frontier.py --all
        """
    )
    
    parser.add_argument('--portfolio-type', type=str, default='size',
                       choices=['size', 'value'],
                       help='Portfolio type (default: size)')
    parser.add_argument('--start-year', type=int, default=1927,
                       help='Start year (default: 1927)')
    parser.add_argument('--end-year', type=int, default=2013,
                       help='End year (default: 2013)')
    parser.add_argument('--exclude-small-caps', action='store_true',
                       help='Exclude small caps')
    parser.add_argument('--recentre', action='store_true',
                       help='Use recentred data')
    parser.add_argument('--all', action='store_true',
                       help='Plot all main configurations')
    parser.add_argument('--no-individual-assets', action='store_true',
                       help='Hide individual asset portfolios')
    
    args = parser.parse_args()
    
    if args.all:
        plot_all_configurations()
    else:
        fig, ax = plot_efficient_frontier(
            portfolio_type=args.portfolio_type,
            start_year=args.start_year,
            end_year=args.end_year,
            exclude_small_caps_flag=args.exclude_small_caps,
            recentre_flag=args.recentre,
            show_individual_assets=not args.no_individual_assets
        )
        plt.show()

