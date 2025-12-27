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
    Plot a simple line from risk-free rate (on y-axis) to a portfolio point, extending to frame edge.
    This is NOT a tangent line, just a straight connection.
    
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
    max_vol : float
        Maximum volatility to extend line to (frame edge)
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


def calculate_frontier_slope(ios_df, portfolio_vol, portfolio_return, efficient_df, inefficient_df):
    """
    Calculate the slope of the frontier at a given portfolio point
    
    Parameters:
    -----------
    ios_df : pd.DataFrame
        Full IOS curve data
    portfolio_vol : float
        Portfolio volatility
    portfolio_return : float
        Portfolio return
    efficient_df : pd.DataFrame
        Efficient limb data
    inefficient_df : pd.DataFrame
        Inefficient limb data
    
    Returns:
    --------
    slope : float
        Slope of the frontier at the portfolio point
    """
    # Determine which limb the portfolio is on
    mvp_idx = ios_df['volatility'].idxmin()
    mvp_return = ios_df.loc[mvp_idx, 'return_gross']
    mvp_vol = ios_df.loc[mvp_idx, 'volatility']
    
    is_efficient = portfolio_return >= mvp_return or portfolio_vol <= mvp_vol
    
    if is_efficient:
        # On efficient limb - use efficient_df
        vols = efficient_df['volatility'].values
        returns = efficient_df['return_gross'].values
    else:
        # On inefficient limb - use inefficient_df
        vols = inefficient_df['volatility'].values
        returns = inefficient_df['return_gross'].values
    
    if len(vols) < 2:
        # Fallback: use simple slope from MVP
        if portfolio_vol > 1e-10:
            return (portfolio_return - mvp_return) / (portfolio_vol - mvp_vol)
        else:
            return 0
    
    # Find closest point on the appropriate limb
    dist = np.abs(vols - portfolio_vol)
    closest_idx = np.argmin(dist)
    
    # Estimate slope using nearby points
    if closest_idx > 0 and closest_idx < len(vols) - 1:
        # Use points before and after to estimate slope
        vol_before = vols[closest_idx - 1]
        vol_after = vols[closest_idx + 1]
        ret_before = returns[closest_idx - 1]
        ret_after = returns[closest_idx + 1]
        
        # Average slope from both sides
        slope1 = (returns[closest_idx] - ret_before) / (vols[closest_idx] - vol_before) if abs(vols[closest_idx] - vol_before) > 1e-10 else 0
        slope2 = (ret_after - returns[closest_idx]) / (vol_after - vols[closest_idx]) if abs(vol_after - vols[closest_idx]) > 1e-10 else 0
        
        if slope1 != 0 and slope2 != 0:
            slope = (slope1 + slope2) / 2
        elif slope1 != 0:
            slope = slope1
        elif slope2 != 0:
            slope = slope2
        else:
            slope = 0
    elif closest_idx == 0 and len(vols) > 1:
        # At start of limb, use forward difference
        slope = (returns[1] - returns[0]) / (vols[1] - vols[0]) if abs(vols[1] - vols[0]) > 1e-10 else 0
    elif closest_idx == len(vols) - 1 and len(vols) > 1:
        # At end of limb, use backward difference
        slope = (returns[-1] - returns[-2]) / (vols[-1] - vols[-2]) if abs(vols[-1] - vols[-2]) > 1e-10 else 0
    else:
        # Fallback
        slope = (portfolio_return - mvp_return) / (portfolio_vol - mvp_vol) if (portfolio_vol - mvp_vol) > 1e-10 else 0
    
    return slope


def plot_tangent_line_from_rf(ax, rf, portfolio_vol, portfolio_return, max_vol, 
                               ios_df, efficient_df, inefficient_df,
                               reflect=False, color='r', linestyle='-', 
                               linewidth=1.5, alpha=0.7, label=None, zorder=3):
    """
    Plot a tangent line to the frontier at a portfolio point, extending to frame edge.
    The line passes through the portfolio point and is tangent to the frontier there.
    It extends back to the y-axis (where it may or may not intersect at R_f).
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    rf : float
        Risk-free rate (gross return, on y-axis at x=0) - for reference only
    portfolio_vol : float
        Portfolio volatility (x-coordinate)
    portfolio_return : float
        Portfolio return (y-coordinate)
    max_vol : float
        Maximum volatility to extend line to (frame edge)
    ios_df : pd.DataFrame
        Full IOS curve data
    efficient_df : pd.DataFrame
        Efficient limb data
    inefficient_df : pd.DataFrame
        Inefficient limb data
    reflect : bool
        If True, reflect the portfolio return across y=rf for the reflected line
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
    # Calculate the slope of the frontier at the portfolio point
    frontier_slope = calculate_frontier_slope(ios_df, portfolio_vol, portfolio_return, 
                                             efficient_df, inefficient_df)
    
    if reflect:
        # For reflected line: reflect the portfolio return across R_f
        reflected_return = 2 * rf - portfolio_return
        # The reflected line passes through (portfolio_vol, reflected_return)
        # Use negative of frontier slope for reflection
        slope = -frontier_slope
        # Calculate y-intercept: R = reflected_return - slope * portfolio_vol (at σ=0)
        y_intercept = reflected_return - slope * portfolio_vol
        # The line passes through (portfolio_vol, reflected_return)
        point_vol = portfolio_vol
        point_return = reflected_return
    else:
        # Normal tangent line: passes through portfolio point with frontier slope
        slope = frontier_slope
        # Calculate y-intercept: R = portfolio_return - slope * portfolio_vol (at σ=0)
        y_intercept = portfolio_return - slope * portfolio_vol
        # The line passes through (portfolio_vol, portfolio_return)
        point_vol = portfolio_vol
        point_return = portfolio_return
    
    # Verify the line passes through the point
    # R_point should equal y_intercept + slope * point_vol
    check = abs(point_return - (y_intercept + slope * point_vol))
    if check > 1e-6:
        print(f"  Warning: Tangent line check failed (error: {check:.2e})")
    
    # Extend line from y-axis (σ=0) to frame edge
    x_line = np.array([0.0, max_vol])
    y_line = np.array([y_intercept, y_intercept + slope * max_vol])
    
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, 
           linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)


def find_tangency_portfolio(rf, efficient_df):
    """
    Find the tangency portfolio (TP) - the portfolio on the efficient frontier
    that maximizes Sharpe ratio: (R - R_f) / σ
    
    Parameters:
    -----------
    rf : float
        Risk-free rate (gross return)
    efficient_df : pd.DataFrame
        Efficient frontier points with 'return_gross' and 'volatility' columns
    
    Returns:
    --------
    tp_return : float or None
        Tangency portfolio return
    tp_vol : float or None
        Tangency portfolio volatility
    """
    if rf is None or len(efficient_df) == 0:
        return None, None
    
    # Find portfolio on efficient frontier that maximizes Sharpe ratio
    # Sharpe ratio = (R - R_f) / σ
    best_sharpe = -np.inf
    tp_return = None
    tp_vol = None
    
    for idx, row in efficient_df.iterrows():
        ret = row['return_gross']
        vol = row['volatility']
        if vol > 1e-10:
            sharpe = (ret - rf) / vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                tp_return = ret
                tp_vol = vol
    
    return tp_return, tp_vol


def plot_asymptote_through_msmp(ax, msmp_vol, msmp_return, slope, max_vol, 
                                 color='r', linestyle='-', linewidth=1.5, 
                                 alpha=0.7, label=None, zorder=3):
    """
    Plot an asymptote that passes through MSMP and is tangent to the IOS.
    
    The asymptote passes through (msmp_vol, msmp_return) and has the given slope.
    It extends from the y-axis (σ=0) to the frame edge.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    msmp_vol : float
        MSMP volatility (x-coordinate)
    msmp_return : float
        MSMP return (gross, y-coordinate)
    slope : float
        Slope of the asymptote (tangent to IOS at MSMP)
    max_vol : float
        Maximum volatility to extend line to (frame edge)
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
    zorder : int
        Z-order for plotting
    """
    # Asymptote: R = msmp_return + slope * (σ - msmp_vol)
    # At σ=0: R = msmp_return - slope * msmp_vol
    # At σ=max_vol: R = msmp_return + slope * (max_vol - msmp_vol)
    y_intercept = msmp_return - slope * msmp_vol
    
    # Extend from y-axis (σ=0) to frame edge
    x_line = np.array([0.0, max_vol])
    y_line = np.array([y_intercept, y_intercept + slope * max_vol])
    
    # Verify it passes through MSMP
    check = abs(msmp_return - (y_intercept + slope * msmp_vol))
    if check > 1e-6:
        print(f"  Warning: Asymptote check failed (error: {check:.2e})")
    
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, 
           linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)


def plot_tangency_line(ax, portfolio_return, msmp_return, portfolio_vol, max_vol,
                       ios_df, efficient_df, inefficient_df,
                       color='r', linestyle='-', linewidth=1.5, alpha=0.7, 
                       label=None, zorder=3):
    """
    Plot a tangency line that is tangent to the IOS at the portfolio point.
    
    The tangency line uses the formula: y_intercept = 2 * R_portfolio - R_MSMP
    BUT the line must be tangent to the frontier, so we use the frontier slope
    at the portfolio point and ensure the line passes through the portfolio.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    portfolio_return : float
        Expected return of the portfolio
    msmp_return : float
        Expected return of MSMP
    portfolio_vol : float
        Volatility of the portfolio (line passes through this point)
    max_vol : float
        Maximum volatility to extend line to (frame edge)
    ios_df : pd.DataFrame
        Full IOS curve data
    efficient_df : pd.DataFrame
        Efficient limb data
    inefficient_df : pd.DataFrame
        Inefficient limb data
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
    # Calculate the slope of the frontier at the portfolio point
    frontier_slope = calculate_frontier_slope(ios_df, portfolio_vol, portfolio_return, 
                                             efficient_df, inefficient_df)
    
    # The tangency line must:
    # 1. Pass through (portfolio_vol, portfolio_return)
    # 2. Have slope = frontier_slope (to be tangent)
    # 3. Have y-intercept = 2 * R_portfolio - R_MSMP (from formula)
    
    # Calculate y-intercept from formula
    y_intercept_formula = 2 * portfolio_return - msmp_return
    
    # But for tangency, the line must pass through portfolio with frontier slope
    # So: portfolio_return = y_intercept + frontier_slope * portfolio_vol
    # Therefore: y_intercept = portfolio_return - frontier_slope * portfolio_vol
    
    y_intercept_tangent = portfolio_return - frontier_slope * portfolio_vol
    
    # Use the tangent y-intercept (ensures tangency)
    # The formula gives us a reference, but tangency requires using frontier slope
    y_intercept = y_intercept_tangent
    
    # Verify the line passes through the portfolio point
    check = abs(portfolio_return - (y_intercept + frontier_slope * portfolio_vol))
    if check > 1e-6:
        print(f"  Warning: Tangency line check failed (error: {check:.2e})")
    
    # Extend line from y-axis (σ=0) to frame edge
    x_line = np.array([0.0, max_vol])
    y_line = np.array([y_intercept, y_intercept + frontier_slope * max_vol])
    
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, 
           linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)


def plot_portfolio_frontier(ios_df, ios_summary, msmp_summary=None, optimal_crra_summary=None,
                      zbp_summary=None, portfolio_type=None, start_year=None, end_year=None,
                      closed_form=False, figsize=(12, 8)):
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
    
    # Calculate axis limits first (needed for tangent lines and circle)
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
    
    # Extend x-axis to at least 1.2 to show the circle from origin to MSMP
    if msmp_summary is not None:
        msmp_vol = msmp_summary['volatility']
        msmp_return = msmp_summary['expected_return_gross']
        # Circle radius = sqrt(σ² + R²)
        circle_radius = np.sqrt(msmp_vol**2 + msmp_return**2)
        # Ensure x-axis extends to show the circle (at least 1.2)
        max_vol = max(max_vol, circle_radius * 1.1, 1.2)
    else:
        # Even without MSMP, extend to 1.2 for consistency
        max_vol = max(max_vol, 1.2)
    
    # Load risk-free rate
    rf = None
    if start_year and end_year:
        rf = load_risk_free_rate(start_year, end_year)
        if rf is not None:
            print(f"  Risk-free rate (mean): R_f = {rf:.6f} (net: {(rf-1)*100:.4f}%)")
    
    # Plot lines from R_f to portfolios
    # These represent the convex space from allocating between long/short positions
    # in the portfolio and the risk-free asset
    
    # Horizontal line through ZBP - extend to end of frame
    if zbp_summary is not None:
        zbp_return = zbp_summary['expected_return_gross']
        zbp_vol = zbp_summary['volatility']
        # Draw horizontal line at ZBP return level, extending to max_vol
        ax.plot([0.0, max_vol], [zbp_return, zbp_return], 
               color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        # Line from CRRA portfolio to intersection of ZBP horizontal line with y-axis (vol=0)
        # Then continue in the same direction to end of frame
        if optimal_crra_summary is not None:
            opt_return = optimal_crra_summary['expected_return_gross']
            opt_vol = optimal_crra_summary['volatility']
            # Calculate slope from (0, zbp_return) to (opt_vol, opt_return)
            if opt_vol > 1e-10:
                slope = (opt_return - zbp_return) / opt_vol
            else:
                slope = 0
            # Line from (0, zbp_return) to (opt_vol, opt_return), then continue to (max_vol, ...)
            # At max_vol: y = zbp_return + slope * max_vol
            y_at_max_vol = zbp_return + slope * max_vol
            x_line = np.array([0.0, opt_vol, max_vol])
            y_line = np.array([zbp_return, opt_return, y_at_max_vol])
            ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='CRRA ↔ ZBP horizontal', zorder=3)
    
    # Calculate and plot tangency portfolio (TP) - distinct from CRRA
    tp_return = None
    tp_vol = None
    if rf is not None and len(efficient_df) > 0:
        tp_return, tp_vol = find_tangency_portfolio(rf, efficient_df)
        if tp_return is not None and tp_vol is not None:
            # Plot tangency portfolio
            ax.scatter([tp_vol], [tp_return],
                      c='blue', s=200, marker='D', edgecolors='black', linewidths=1.5,
                      label=f'Tangency Portfolio (TP) (R={tp_return:.4f}, σ={tp_vol:.2%})', zorder=5)
    
    # Red lines: from (0, R_mvp) to TP and from (0, R_mvp) to MSMP
    # These are not asymptotes but lines connecting MVP return on y-axis to key portfolios
    if msmp_summary is not None:
        msmp_return = msmp_summary['expected_return_gross']
        msmp_vol = msmp_summary['volatility']
        
        # Line from (0, R_mvp) to MSMP
        # Slope = (msmp_return - mvp_return) / (msmp_vol - 0) = (msmp_return - mvp_return) / msmp_vol
        if msmp_vol > 1e-10:
            slope_msmp = (msmp_return - mvp_return) / msmp_vol
        else:
            slope_msmp = 0
        
        # Plot line from y-axis (σ=0) at R_mvp to frame edge
        x_line = np.array([0.0, max_vol])
        y_line = np.array([mvp_return, mvp_return + slope_msmp * max_vol])
        ax.plot(x_line, y_line, color='r', linestyle='-', linewidth=1.5, alpha=0.7,
               label='R_MVP → MSMP', zorder=3)
    
    # Line from (0, R_mvp) to Tangency Portfolio (TP)
    if tp_return is not None and tp_vol is not None:
        # Slope = (tp_return - mvp_return) / (tp_vol - 0) = (tp_return - mvp_return) / tp_vol
        if tp_vol > 1e-10:
            slope_tp = (tp_return - mvp_return) / tp_vol
        else:
            slope_tp = 0
        
        # Plot line from y-axis (σ=0) at R_mvp to frame edge
        x_line = np.array([0.0, max_vol])
        y_line = np.array([mvp_return, mvp_return + slope_tp * max_vol])
        ax.plot(x_line, y_line, color='r', linestyle='-', linewidth=1.5, alpha=0.7,
               label='R_MVP → TP', zorder=3)
    
    # Circle from origin to MSMP (centered at origin)
    if msmp_summary is not None:
        msmp_return = msmp_summary['expected_return_gross']
        msmp_vol = msmp_summary['volatility']
        
        # Circle from origin to MSMP (centered at origin)
        # Radius = distance from origin to MSMP in (σ, R) space
        radius = np.sqrt(msmp_vol**2 + msmp_return**2)
        circle = plt.Circle((0, 0), radius, fill=False, color='black', 
                           linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.add_patch(circle)
    
    # Labels and formatting
    ax.set_xlabel('Volatility (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (R)', fontsize=12, fontweight='bold')
    
    # Title
    method_label = ' (Closed-Form)' if closed_form else ''
    if portfolio_type and start_year and end_year:
        title_parts = ['Investment Opportunity Set']
        if msmp_summary is not None:
            title_parts.append('MSMP')
        if optimal_crra_summary is not None:
            title_parts.append('Optimal CRRA')
        if zbp_summary is not None:
            title_parts.append('ZBP')
        title = f"{', '.join(title_parts)}{method_label}\n{portfolio_type.upper()} Portfolios ({start_year}-{end_year})"
    else:
        title = 'Investment Opportunity Set'
        if msmp_summary is not None:
            title += ' and MSMP'
        if optimal_crra_summary is not None:
            title += ' and Optimal CRRA'
        if zbp_summary is not None:
            title += ' and ZBP'
        title += method_label
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Format axes: x-axis as decimals (not percentages), y-axis as decimals (gross returns)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    # Set axis limits (already calculated above)
    ax.set_xlim(min_vol, max_vol)
    ax.set_ylim(min_ret, max_ret)
    
    # Mark and label R_f on y-axis if available
    if rf is not None:
        # Mark R_f with a point on the y-axis
        ax.scatter([0.0], [rf], c='gray', s=150, marker='o', edgecolors='black', 
                  linewidths=1.5, zorder=5, label=f'R_f = {rf:.4f}')
        # Add text label
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
    parser.add_argument(
        '--closed-form', action='store_true',
        help='Indicate that closed-form solutions were used (for filename/title)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Plotting Investment Opportunity Set and MSMP")
    print("=" * 70)
    print(f"Portfolio type: {args.portfolio_type}")
    print(f"Period: {args.start_year}-{args.end_year}")
    allow_short = not args.no_short_selling
    print(f"Short selling: {'ALLOWED (free portfolio)' if allow_short else 'NOT ALLOWED (long-only)'}")
    print(f"Method: {'CLOSED-FORM' if args.closed_form else 'OPTIMIZATION'}")
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
        ios_df, ios_summary, msmp_summary=msmp_summary, 
        optimal_crra_summary=optimal_crra_summary, zbp_summary=zbp_summary,
        portfolio_type=args.portfolio_type, start_year=args.start_year, end_year=args.end_year,
        closed_form=args.closed_form
    )
    
    # Save plot
    suffix = f"_{args.portfolio_type}_{args.start_year}_{args.end_year}"
    if args.closed_form:
        suffix += "_closed_form"
    if args.output_suffix:
        suffix += args.output_suffix
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

