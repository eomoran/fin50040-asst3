# Zero-Beta Portfolio (ZBP) Calculation Explanation

## Overview

The Zero-Beta Portfolio (ZBP) is calculated to have **zero covariance** with the optimal portfolio. This document explains the current implementation and identifies potential issues.

## Current Implementation

### Function: `find_zero_beta_portfolio(mu, Sigma, w_m, on_frontier=True)`

**Inputs:**
- `mu`: Mean gross returns (E[R]) - shape (n,)
- `Sigma`: Covariance matrix of gross returns (Cov(R)) - shape (n, n)
- `w_m`: Optimal portfolio weights - shape (n,)
- `on_frontier`: If True, find ZBP on the efficient frontier

**Mathematical Constraint:**
The ZBP must satisfy:
```
Cov(R_z, R_m) = 0
w_z'Σw_m = 0
```

Where:
- `R_z = w_z'R` is the ZBP return
- `R_m = w_m'R` is the optimal portfolio return
- `w_z` are the ZBP weights

### Current Algorithm (when `on_frontier=True`)

1. **Calculate MVP (Minimum Variance Portfolio)**:
   ```python
   w_mvp = inv_Sigma @ ones / (ones.T @ inv_Sigma @ ones)
   mu_mvp = mu.T @ w_mvp
   ```

2. **Determine Search Range**:
   ```python
   mu_optimal = mu.T @ w_m  # Optimal portfolio return
   search_min = mu_minvar * 0.95  # 5% below MVP (inefficient part)
   search_max = max(mu_optimal, mu_minvar * 1.1)  # To optimal or 10% above MVP
   target_returns = np.linspace(search_min, search_max, 400)
   ```

3. **For Each Target Return**:
   - Minimize variance subject to:
     - Budget constraint: `sum(w) = 1`
     - Return constraint: `mu'w = target_return` (ensures on frontier)
     - Zero-beta constraint: `w'Σw_m = 0`
   
   ```python
   constraints = [
       {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
       {'type': 'eq', 'fun': lambda w: mu.T @ w - target_return},
       {'type': 'eq', 'fun': lambda w: w.T @ Sigma @ w_m}
   ]
   ```

4. **Select Best Solution**:
   - Among all successful optimizations, pick the one with smallest `|w'Σw_m|` (closest to zero)

## Potential Issues

### Issue 1: Search Range May Be Too Restrictive

**Current range**: `[mvp_return * 0.95, max(optimal_return, mvp_return * 1.1)]`

**Problem**: If the optimal return is very high, this might miss the ZBP which should be on the inefficient limb (below MVP).

**From professor's plot**: ZBP is at approximately:
- Return: ~0.07 (7% net, 1.07 gross)
- Volatility: ~0.20 (20%)
- On inefficient limb (below MVP)

**Fix**: The search should definitely include the inefficient part. The current `search_min = mu_minvar * 0.95` should work, but maybe we need to extend further below MVP.

### Issue 2: Optimization May Be Failing

The optimization has **3 equality constraints**:
1. Budget: `sum(w) = 1`
2. Return: `mu'w = target_return`
3. Zero-beta: `w'Σw_m = 0`

With bounds `[-1, 1]` for each weight, this might be infeasible for some return levels.

**Check**: Are optimizations succeeding? Are we getting valid solutions?

### Issue 3: Initial Guess May Be Poor

**Current**: Uses `w0 = np.ones(n) / n` (equal weights) for all target returns.

**Problem**: For inefficient part (below MVP), equal weights might not be a good starting point.

**Fix**: Use better initial guesses:
- For inefficient part: try negative of MVP (scaled)
- For efficient part: try MVP or optimal portfolio

### Issue 4: Constraint Tolerance

**Current check**: `cov_check < 1e-4` (0.0001)

**Problem**: This might be too strict, causing valid solutions to be rejected.

## Expected Behavior (from Professor's Plot)

Based on the professor's plot for SIZE portfolios (1927-2013):
- **MVP**: Return ~0.105 (10.5%), Vol ~0.15 (15%)
- **ZBP**: Return ~0.07 (7%), Vol ~0.20 (20%) - **on inefficient limb**
- **Optimal**: Return ~0.20 (20%), Vol ~0.28 (28%) - **on efficient limb**

The ZBP should be:
1. **On the inefficient limb** (below MVP in return)
2. **Near the MVP** (similar volatility, slightly higher)
3. **Have zero covariance** with optimal portfolio

## Recommended Fixes

1. **Extend search range below MVP**:
   ```python
   # Search from well below MVP to optimal
   search_min = mu_minvar * 0.85  # 15% below MVP (more room for inefficient part)
   search_max = max(mu_optimal, mu_minvar * 1.2)  # Extend above MVP too
   ```

2. **Better initial guesses**:
   ```python
   if target_return < mu_minvar:
       # Inefficient part: try negative MVP or scaled MVP
       w0 = -w_minvar.copy() / np.sum(-w_minvar)  # Normalized negative MVP
   else:
       # Efficient part: try MVP or optimal
       w0 = w_minvar.copy()  # Start from MVP
   ```

3. **Relax constraint tolerance**:
   ```python
   if budget_check < 1e-5 and return_check < 1e-5 and cov_check < 1e-3:  # More lenient
   ```

4. **Verify ZBP is actually on frontier**:
   After finding ZBP, check if it's actually on the frontier by finding the closest frontier point.

## Alternative Approach

If the 3-constraint optimization is too difficult, we could:
1. First find all portfolios on the frontier
2. Then search through those portfolios for the one with smallest `|w'Σw_m|`

This ensures the ZBP is definitely on the frontier.

