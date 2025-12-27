#!/usr/bin/env python3
"""
Diagnostic script to trace Zero-Beta Portfolio calculation
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from portfolio_analysis import (
    load_portfolio_data, compute_moments, 
    find_optimal_crra_portfolio, find_zero_beta_portfolio,
    construct_mv_frontier
)

print("=" * 70)
print("Zero-Beta Portfolio Calculation Diagnostic")
print("=" * 70)

# Load data
returns = load_portfolio_data('size', 1927, 2013)
mu, Sigma, portfolio_names = compute_moments(returns)

print(f"\nData: {len(portfolio_names)} portfolios, {len(returns)} observations")
print(f"Mean returns range: [{mu.min():.4f}, {mu.max():.4f}] (gross)")
print(f"Volatility range: [{np.sqrt(np.diag(Sigma)).min():.4f}, {np.sqrt(np.diag(Sigma)).max():.4f}]")

# Find optimal portfolio
print(f"\n{'='*70}")
print("Step 1: Find Optimal CRRA Portfolio (RRA=4)")
print(f"{'='*70}")
w_opt, opt_return, opt_vol = find_optimal_crra_portfolio(mu, Sigma, rra=4)
print(f"Optimal portfolio:")
print(f"  Return (gross): {opt_return:.6f} (net: {opt_return-1:.6f})")
print(f"  Volatility: {opt_vol:.6f}")
print(f"  Weight sum: {w_opt.sum():.6f}")

# Find MVP
print(f"\n{'='*70}")
print("Step 2: Find Minimum Variance Portfolio (MVP)")
print(f"{'='*70}")
n = len(mu)
ones = np.ones(n)
inv_Sigma = np.linalg.pinv(Sigma)
w_mvp = inv_Sigma @ ones / (ones.T @ inv_Sigma @ ones)
mvp_return = mu.T @ w_mvp
mvp_vol = np.sqrt(w_mvp.T @ Sigma @ w_mvp)
print(f"MVP:")
print(f"  Return (gross): {mvp_return:.6f} (net: {mvp_return-1:.6f})")
print(f"  Volatility: {mvp_vol:.6f}")

# Check covariance between optimal and MVP
cov_opt_mvp = w_opt.T @ Sigma @ w_mvp
print(f"\nCovariance between optimal and MVP: {cov_opt_mvp:.6f}")

# Find ZBP
print(f"\n{'='*70}")
print("Step 3: Find Zero-Beta Portfolio")
print(f"{'='*70}")
print("Constraint: Cov(R_z, R_opt) = w_z'Σw_opt = 0")
print(f"Current optimal portfolio covariance with itself: {w_opt.T @ Sigma @ w_opt:.6f}")

# Calculate search range
mu_optimal = opt_return
search_min = mvp_return * 0.95
search_max = max(mu_optimal, mvp_return * 1.1)
print(f"\nSearch range:")
print(f"  MVP return: {mvp_return:.6f}")
print(f"  Optimal return: {mu_optimal:.6f}")
print(f"  Search min (5% below MVP): {search_min:.6f}")
print(f"  Search max: {search_max:.6f}")

# Try finding ZBP
print(f"\nAttempting to find ZBP on frontier...")
w_z, z_return, z_vol = find_zero_beta_portfolio(mu, Sigma, w_opt, on_frontier=True)

print(f"\nZero-Beta Portfolio:")
print(f"  Return (gross): {z_return:.6f} (net: {z_return-1:.6f})")
print(f"  Volatility: {z_vol:.6f}")
print(f"  Weight sum: {w_z.sum():.6f}")

# Verify constraints
zero_beta_check = abs(w_z.T @ Sigma @ w_opt)
print(f"\nConstraint verification:")
print(f"  Zero-beta constraint (w_z'Σw_opt): {w_z.T @ Sigma @ w_opt:.2e} (should be ~0)")
print(f"  Budget constraint (sum(w_z)): {w_z.sum():.6f} (should be 1.0)")

# Check if ZBP is on frontier
print(f"\n{'='*70}")
print("Step 4: Check if ZBP is on Efficient Frontier")
print(f"{'='*70}")
frontier = construct_mv_frontier(mu, Sigma, num_portfolios=200)

# Find closest frontier point to ZBP
z_return_net = z_return - 1
z_vol_net = z_vol

frontier_returns_net = frontier['returns'] - 1
distances = np.sqrt((frontier['volatilities'] - z_vol)**2 + (frontier_returns_net - z_return_net)**2)
closest_idx = np.argmin(distances)
closest_dist = distances[closest_idx]

print(f"ZBP location: Return={z_return_net:.4f}, Vol={z_vol:.4f}")
print(f"Closest frontier point: Return={frontier_returns_net[closest_idx]:.4f}, Vol={frontier['volatilities'][closest_idx]:.4f}")
print(f"Distance from frontier: {closest_dist:.6f}")

if closest_dist < 0.01:
    print("  ✓ ZBP is on or very close to frontier")
else:
    print(f"  ✗ ZBP is NOT on frontier (distance: {closest_dist:.4f})")

# Check position relative to MVP and optimal
print(f"\n{'='*70}")
print("Step 5: Position Analysis")
print(f"{'='*70}")
print(f"MVP: Return={mvp_return-1:.4f}, Vol={mvp_vol:.4f}")
print(f"ZBP: Return={z_return-1:.4f}, Vol={z_vol:.4f}")
print(f"Optimal: Return={opt_return-1:.4f}, Vol={opt_vol:.4f}")

if z_return < mvp_return:
    print(f"\n✓ ZBP is on inefficient limb (return < MVP return)")
    print(f"  Difference from MVP: {mvp_return - z_return:.6f} (gross), {mvp_return-1 - (z_return-1):.6f} (net)")
else:
    print(f"\n✗ ZBP is NOT on inefficient limb (return >= MVP return)")

# Check if ZBP is between MVP and optimal
if mvp_return <= z_return <= opt_return or opt_return <= z_return <= mvp_return:
    print(f"✓ ZBP return is between MVP and Optimal")
else:
    print(f"✗ ZBP return is outside MVP-Optimal range")

# Try alternative: find ZBP without frontier constraint
print(f"\n{'='*70}")
print("Step 6: Compare with ZBP NOT on frontier")
print(f"{'='*70}")
w_z2, z_return2, z_vol2 = find_zero_beta_portfolio(mu, Sigma, w_opt, on_frontier=False)
zero_beta_check2 = abs(w_z2.T @ Sigma @ w_opt)
print(f"ZBP (not on frontier):")
print(f"  Return (gross): {z_return2:.6f} (net: {z_return2-1:.6f})")
print(f"  Volatility: {z_vol2:.6f}")
print(f"  Zero-beta constraint: {zero_beta_check2:.2e}")
print(f"  Distance from frontier: {np.min(np.sqrt((frontier['volatilities'] - z_vol2)**2 + ((frontier['returns']-1) - (z_return2-1))**2)):.6f}")

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"ZBP on frontier: Return={z_return-1:.4f}, Vol={z_vol:.4f}, Cov check={zero_beta_check:.2e}")
print(f"ZBP off frontier: Return={z_return2-1:.4f}, Vol={z_vol2:.4f}, Cov check={zero_beta_check2:.2e}")
print(f"\nExpected (from professor's plot): ZBP on inefficient limb near MVP")
print(f"  Approximate: Return ~0.07 (7%), Vol ~0.20 (20%)")

