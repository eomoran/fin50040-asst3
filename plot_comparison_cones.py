#!/usr/bin/env python3
"""
Plot Comparison Cones

This script plots comparison lines (linear combinations) for comparing against
the lecturer's red lines. These include:
- R_f ↔ MSMP (cyan) - both upper and lower lines
- R_ZBP ↔ MSMP (orange) - both upper and lower lines
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Directories
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/processed")


def load_portfolio_weights(portfolio_type, start_year, end_year, weights_file):
    """Load portfolio weights from CSV file"""
    weights_path = RESULTS_DIR / weights_file
    if not weights_path.exists():
        return None, None
    
    weights_df = pd.read_csv(weights_path)
    weights = weights_df['weight'].values
    portfolio_names = weights_df['portfolio'].tolist()
    
    return weights, portfolio_names


def compute_portfolio_return_series(returns, weights, portfolio_names):
    """Compute return series for a portfolio given weights"""
    available_names = [name for name in portfolio_names if name in returns.columns]
    if len(available_names) != len(portfolio_names):
        print(f"  Warning: {len(portfolio_names) - len(available_names)} portfolios not found in returns")
    
    weight_dict = dict(zip(portfolio_names, weights))
    available_weights = np.array([weight_dict[name] for name in available_names])
    
    if abs(available_weights.sum() - 1.0) > 1e-6:
        available_weights = available_weights / available_weights.sum()
    
    portfolio_returns = (returns[available_names] @ available_weights)
    return portfolio_returns


def plot_anchored_lines(ax, anchor_return, portfolio_vol, portfolio_return, max_vol, 
                       color='blue', label=None, zorder=2, alpha_upper=2.0, plot_upper=True):
    """
    Plot lines anchored at (0, anchor_return) representing linear combinations.
    
    For a zero-volatility anchor point and a risky portfolio, plots:
    - Lower line: from (0, anchor_return) through (portfolio_vol, portfolio_return) - represents α ≤ 1
    - Upper line (if plot_upper=True): from (0, anchor_return) through a point where α = alpha_upper - represents α > 1
    """
    # Lower line
    if portfolio_vol > 1e-10:
        slope_lower = (portfolio_return - anchor_return) / portfolio_vol
    else:
        slope_lower = 0
    
    # Upper line
    upper_return = alpha_upper * anchor_return + (1 - alpha_upper) * portfolio_return
    upper_vol = abs(1 - alpha_upper) * portfolio_vol
    
    if upper_vol > 1e-10:
        slope_upper = (upper_return - anchor_return) / upper_vol
    else:
        slope_upper = 0
    
    # Plot lower line
    x_lower = np.array([0.0, max_vol])
    y_lower = np.array([anchor_return, anchor_return + slope_lower * max_vol])
    ax.plot(x_lower, y_lower, color=color, linestyle='-', linewidth=1.5, 
           alpha=0.7, zorder=zorder, label=label)
    
    # Plot upper line (if requested)
    if plot_upper:
        x_upper = np.array([0.0, max_vol])
        y_upper = np.array([anchor_return, anchor_return + slope_upper * max_vol])
        ax.plot(x_upper, y_upper, color=color, linestyle='-', linewidth=1.5, 
               alpha=0.7, zorder=zorder)  # No label - single label for both lines
    
    # Return combined points for intersection finding
    vol_vals = np.linspace(0, max_vol, 500)
    return_vals_lower = anchor_return + slope_lower * vol_vals
    if plot_upper:
        return_vals_upper = anchor_return + slope_upper * vol_vals
        line_vols = np.concatenate([vol_vals, vol_vals])
        line_returns = np.concatenate([return_vals_lower, return_vals_upper])
    else:
        line_vols = vol_vals
        line_returns = return_vals_lower
    
    return line_vols, line_returns


def find_line_ios_intersections(line_vols, line_returns, ios_vols, ios_returns, tolerance=0.01):
    """Find intersections between a line and the IOS frontier"""
    if len(line_vols) < 2 or len(ios_vols) == 0:
        return np.array([]), np.array([])
    
    intersections_vols = []
    intersections_returns = []
    
    line_vol_min = line_vols.min()
    line_vol_max = line_vols.max()
    
    for ios_vol, ios_return in zip(ios_vols, ios_returns):
        if line_vol_min <= ios_vol <= line_vol_max:
            if len(line_vols) >= 2:
                vol1, ret1 = line_vols[0], line_returns[0]
                vol2, ret2 = line_vols[-1], line_returns[-1]
                
                if abs(vol2 - vol1) > 1e-10:
                    slope = (ret2 - ret1) / (vol2 - vol1)
                    line_return_at_ios_vol = ret1 + slope * (ios_vol - vol1)
                    dist = abs(ios_return - line_return_at_ios_vol)
                    
                    if dist < tolerance:
                        intersections_vols.append(ios_vol)
                        intersections_returns.append(ios_return)
    
    # Remove duplicates
    if len(intersections_vols) > 0:
        intersections_vols = np.array(intersections_vols)
        intersections_returns = np.array(intersections_returns)
        
        if len(intersections_vols) > 1:
            sort_idx = np.argsort(intersections_vols)
            intersections_vols = intersections_vols[sort_idx]
            intersections_returns = intersections_returns[sort_idx]
            
            keep = [True]
            for i in range(1, len(intersections_vols)):
                dist = np.sqrt((intersections_vols[i] - intersections_vols[i-1])**2 + 
                             (intersections_returns[i] - intersections_returns[i-1])**2)
                if dist > tolerance * 3:
                    keep.append(True)
                else:
                    keep.append(False)
            intersections_vols = intersections_vols[keep]
            intersections_returns = intersections_returns[keep]
    
    return np.array(intersections_vols), np.array(intersections_returns)


def plot_comparison_cones(ax, portfolio_type, start_year, end_year, 
                         rf, msmp_summary, zbp_summary, tp_return, tp_vol,
                         frontier_vols, frontier_returns, max_vol):
    """
    Plot comparison cones (linear combination lines) on an existing axes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    portfolio_type : str
        'size' or 'value'
    start_year : int
        Start year
    end_year : int
        End year
    rf : float
        Risk-free rate (gross return)
    msmp_summary : pd.Series
        MSMP summary
    zbp_summary : pd.Series
        ZBP summary
    tp_return : float
        Tangency portfolio return
    tp_vol : float
        Tangency portfolio volatility
    frontier_vols : np.array
        IOS frontier volatilities
    frontier_returns : np.array
        IOS frontier returns
    max_vol : float
        Maximum volatility for plotting
    """
    try:
        # Load portfolio returns
        from construct_investment_opportunity_set import load_portfolio_returns
        returns = load_portfolio_returns(portfolio_type, start_year, end_year)
        
        msmp_return = msmp_summary['expected_return_gross']
        msmp_vol = msmp_summary['volatility']
        
        # Line 1: R_f and MSMP (cyan) - plot both upper and lower
        if rf is not None and msmp_summary is not None:
            rf_return = rf
            
            print(f"  Generating R_f ↔ MSMP line (all linear combinations)...")
            print(f"    R_f: (0.0000, {rf_return:.4f})")
            print(f"    MSMP: ({msmp_vol:.4f}, {msmp_return:.4f})")
            
            # Use plot_anchored_lines with plot_upper=True to show both lines
            line_vols, line_returns = plot_anchored_lines(
                ax, anchor_return=rf_return, portfolio_vol=msmp_vol,
                portfolio_return=msmp_return, max_vol=max_vol * 1.2,
                color='cyan', label='Line: R_f ↔ MSMP', zorder=2, alpha_upper=2.0, plot_upper=True
            )
            
            print(f"    Generated line with {len(line_vols)} points")
            print(f"    Vol range: [{line_vols.min():.4f}, {line_vols.max():.4f}]")
            print(f"    Return range: [{line_returns.min():.4f}, {line_returns.max():.4f}]")
            
            # Find intersections with IOS
            if len(line_vols) > 0:
                intersection_vols, intersection_returns = find_line_ios_intersections(
                    line_vols, line_returns, 
                    frontier_vols, frontier_returns, tolerance=0.01
                )
                if len(intersection_vols) > 0:
                    ax.scatter(intersection_vols, intersection_returns, 
                             c='cyan', s=50, marker='o', edgecolors='darkcyan', 
                             linewidths=1, zorder=6, alpha=0.8)
            
            # Check if TP lies on the R_f ↔ MSMP line
            if tp_return is not None and tp_vol is not None and len(line_vols) > 0:
                if abs(rf_return - msmp_return) > 1e-10:
                    alpha_for_tp_return = (tp_return - msmp_return) / (rf_return - msmp_return)
                    vol_from_alpha = abs(1 - alpha_for_tp_return) * msmp_vol
                    
                    line_vols_sorted = np.sort(line_vols)
                    line_returns_sorted = line_returns[np.argsort(line_vols)]
                    closest_idx = np.argmin(np.abs(line_vols_sorted - tp_vol))
                    closest_vol = line_vols_sorted[closest_idx]
                    closest_return = line_returns_sorted[closest_idx]
                    dist = np.sqrt((tp_vol - closest_vol)**2 + (tp_return - closest_return)**2)
                    
                    tolerance = 0.01
                    if dist < tolerance:
                        ax.scatter([tp_vol], [tp_return], 
                                 c='cyan', s=300, marker='*', edgecolors='darkcyan', 
                                 linewidths=2, zorder=7, alpha=0.9,
                                 label=f'TP on R_f↔MSMP line (α={alpha_for_tp_return:.3f})')
                        print(f"  ✓ TP lies on R_f ↔ MSMP line!")
                        print(f"    α for TP return: {alpha_for_tp_return:.4f}")
                        print(f"    TP actual vol: {tp_vol:.4f}, vol from α: {vol_from_alpha:.4f}")
                        print(f"    Distance from line: {dist:.6f}")
                    else:
                        print(f"  TP is NOT on R_f ↔ MSMP line")
                        print(f"    α for TP return: {alpha_for_tp_return:.4f}")
                        print(f"    TP actual vol: {tp_vol:.4f}, vol from α: {vol_from_alpha:.4f}")
                        print(f"    Distance from line: {dist:.6f}")
                        
                        closest_return_idx = np.argmin(np.abs(line_returns_sorted - tp_return))
                        hyperbola_vol_at_tp_return = line_vols_sorted[closest_return_idx]
                        hyperbola_return_at_tp_return = line_returns_sorted[closest_return_idx]
                        
                        ax.scatter([hyperbola_vol_at_tp_return], [hyperbola_return_at_tp_return], 
                                 c='cyan', s=200, marker='x', linewidths=3, zorder=7, alpha=0.7,
                                 label=f'R_f↔MSMP at TP return (α={alpha_for_tp_return:.3f})')
        
        # Line 2: ZBP and MSMP (orange) - plot both upper and lower
        if zbp_summary is not None and msmp_summary is not None:
            zbp_return = zbp_summary['expected_return_gross']
            
            # Use plot_anchored_lines to plot both upper and lower lines
            line_vols, line_returns = plot_anchored_lines(
                ax, anchor_return=zbp_return, portfolio_vol=msmp_vol, 
                portfolio_return=msmp_return, max_vol=max_vol,
                color='orange', label='Line: R_ZBP ↔ MSMP', zorder=2, alpha_upper=2.0, plot_upper=True
            )
            
            # Find intersections with IOS
            if len(line_vols) > 0:
                intersection_vols, intersection_returns = find_line_ios_intersections(
                    line_vols, line_returns, 
                    frontier_vols, frontier_returns, tolerance=0.01
                )
                if len(intersection_vols) > 0:
                    ax.scatter(intersection_vols, intersection_returns, 
                             c='orange', s=50, marker='o', edgecolors='darkorange', 
                             linewidths=1, zorder=6, alpha=0.8)
    
    except Exception as e:
        print(f"  Warning: Could not plot comparison cones: {e}")
        import traceback
        traceback.print_exc()

