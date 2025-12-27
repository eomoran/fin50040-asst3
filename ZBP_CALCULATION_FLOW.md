# Zero-Beta Portfolio (ZBP) Calculation Flow

## Overview

The Zero-Beta Portfolio (ZBP) is a portfolio that has **zero covariance** with the optimal portfolio and lies on the **inefficient limb** of the efficient frontier (below the Minimum Variance Portfolio).

## Mathematical Definition

For a portfolio with weights `w_z`, the ZBP satisfies:

```
Cov(R_z, R_m) = 0
```

Where:
- `R_z = w_z'R` is the ZBP return (gross)
- `R_m = w_m'R` is the optimal portfolio return (gross)
- `R` is the vector of asset gross returns

In matrix form:
```
w_z'Σw_m = 0
```

Where `Σ` is the covariance matrix of gross returns.

## Current Implementation Flow

### Step 1: Calculate Key Portfolios

1. **Minimum Variance Portfolio (MVP)**:
   ```
   w_mvp = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
   μ_mvp = μ'w_mvp
   ```

2. **Optimal CRRA Portfolio** (RRA=4):
   ```
   w_opt = argmax_w [μ'w - (γ/2) * w'Σw]
   subject to: sum(w) = 1
   μ_opt = μ'w_opt
   ```

### Step 2: Determine Search Range

The ZBP should be on the **inefficient limb** (below MVP), so we search:

```
search_min = μ_mvp * 0.80  (20% below MVP)
search_max = μ_mvp * 1.10  (10% above MVP)
```

**Rationale**: Based on professor's plot, ZBP is on inefficient limb near MVP.

### Step 3: Grid Search Along Frontier

For each target return `μ_target` in `[search_min, search_max]`:

1. **Optimization Problem**:
   ```
   minimize: w'Σw  (variance)
   subject to:
     - sum(w) = 1  (budget constraint)
     - μ'w = μ_target  (return constraint - ensures on frontier)
     - w'Σw_m = 0  (zero-beta constraint)
   ```

2. **Initial Guesses**:
   - **If μ_target < μ_mvp** (inefficient part):
     - Try negative MVP: `w0 = -w_mvp / sum(-w_mvp)`
     - Try scaled MVP: `w0 = 0.5 * w_mvp / sum(0.5 * w_mvp)`
   - **If μ_target ≥ μ_mvp** (efficient part):
     - Try MVP: `w0 = w_mvp`
   - Always include equal weights as fallback

3. **Constraint Verification**:
   ```
   |sum(w) - 1| < 1e-5  (budget)
   |μ'w - μ_target| < 1e-5  (return)
   |w'Σw_m| < 1e-3  (zero-beta, more lenient)
   ```

### Step 4: Select Best Solution

Among all successful optimizations, select the one with:
- Smallest `|w'Σw_m|` (closest to zero covariance)

## Expected Results (from Professor's Plot)

For SIZE portfolios (1927-2013):
- **MVP**: Return ≈ 0.105 (10.5%), Vol ≈ 0.15 (15%)
- **ZBP**: Return ≈ 0.07 (7%), Vol ≈ 0.20 (20%) ← **On inefficient limb**
- **Optimal**: Return ≈ 0.20 (20%), Vol ≈ 0.28 (28%) ← **On efficient limb**

## Key Points

1. **ZBP is on inefficient limb**: Return < MVP return
2. **ZBP has zero covariance** with optimal portfolio
3. **ZBP is on frontier**: Minimum variance for its return level
4. **ZBP is near MVP**: Similar volatility, slightly higher

## Potential Issues and Fixes

### Issue: ZBP Not on Frontier
**Check**: After finding ZBP, verify it's actually on the frontier by finding the closest frontier point.

### Issue: Optimization Failing
**Fix**: Use better initial guesses (negative MVP for inefficient part).

### Issue: Search Range Too Narrow
**Fix**: Extend search to 20% below MVP (was 5% before).

### Issue: Constraint Tolerance Too Strict
**Fix**: Use more lenient tolerance for zero-beta constraint (1e-3 instead of 1e-4).

## Code Location

Function: `find_zero_beta_portfolio()` in `portfolio_analysis.py` (lines 488-720)

Called from:
- `portfolio_analysis.py` main() → line 1094
- `plot_efficient_frontier.py` → line 97

