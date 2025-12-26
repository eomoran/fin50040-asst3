# Ex-Post Pricing Kernel and MSMP Analysis

## Overview

This document analyzes the ex-post pricing kernel (stochastic discount factor) and its relationship to the Minimum Second Moment Portfolio (MSMP) for both the 1927-2013 subsample and the 1927-2024 full sample periods.

## The Pricing Kernel and MSMP Relationship

### Theoretical Relationship

The **pricing kernel** (stochastic discount factor, SDF) is related to the MSMP portfolio return by:

\[
m = \frac{R_{msmp}}{E[R_{msmp}^2]}
\]

where:
- \( R_{msmp} = 1 + r_{msmp} \) is the **gross return** on the MSMP portfolio
- \( r_{msmp} \) is the net return on the MSMP portfolio
- \( E[R_{msmp}^2] \) is the **second moment** of the MSMP return

The **price of the SDF** is:

\[
p(m) = E[m] = \frac{E[R_{msmp}]}{E[R_{msmp}^2]}
\]

### Why This Relationship?

The MSMP minimizes the second moment \( E[R^2] = E[R]^2 + Var(R) \) subject to the budget constraint. The pricing kernel constructed from MSMP has the property that it prices all assets correctly in the sense that:

\[
E[m \cdot R_i] = 1
\]

for all asset returns \( R_i \), where \( R_i = 1 + r_i \) is the gross return on asset \( i \).

## Why 52 Portfolio-Period Combinations?

For each period (1927-2013 and 1927-2024), we have:

- **Size portfolios (full)**: 19 portfolios
- **Value portfolios (full)**: 19 portfolios  
- **Size portfolios (no small caps)**: 14 portfolios (excludes 5 smallest: `<= 0`, `Lo 10`, `Lo 20`, `Lo 30`, `2-Dec`)

**Total**: 19 + 19 + 14 = **52 portfolio-period combinations** per time period

This explains why both periods show 52 observations in the non-recentred, risk-free CAPM results.

## MSMP Weight Comparison: 1927-2013 vs 1927-2024

### Size Portfolios

- **Weight correlation**: 0.9394 (high correlation, but not perfect)
- **Mean absolute difference**: 0.1392
- **Max absolute difference**: 0.4265 (for `Qnt 4` portfolio)

**Key observations**:
- MSMP weights are relatively stable across periods (94% correlation)
- However, some portfolios show significant weight changes:
  - `Qnt 4`: 0.6816 → 0.2552 (decrease of 0.4265)
  - `3-Dec`: -0.5125 → -0.1087 (less negative)
  - `9-Dec`: -0.4098 → -0.0380 (less negative)

### Value Portfolios

- **Weight correlation**: 0.9544 (slightly higher than size portfolios)
- **Mean absolute difference**: 0.1222
- **Max absolute difference**: 0.3369 (for `7-Dec` portfolio)

**Key observations**:
- Value portfolios show even higher stability (95% correlation)
- Largest changes:
  - `7-Dec`: -0.1226 → 0.2143 (sign change!)
  - `Qnt 2`: 0.7766 → 0.4590 (decrease)
  - `4-Dec`: -0.3701 → -0.0804 (less negative)

## Ex-Post Pricing Kernel Performance

### Computing the Pricing Kernel

To compute the ex-post pricing kernel, we:

1. **Load portfolio returns** for the period
2. **Compute MSMP weights** (already saved in results)
3. **Calculate MSMP return series**: \( r_{msmp,t} = \sum_i w_{msmp,i} \cdot r_{i,t} \)
4. **Compute gross returns**: \( R_{msmp,t} = 1 + r_{msmp,t} \)
5. **Calculate second moment**: \( E[R_{msmp}^2] = \frac{1}{T} \sum_t R_{msmp,t}^2 \)
6. **Construct pricing kernel**: \( m_t = \frac{R_{msmp,t}}{E[R_{msmp}^2]} \)
7. **Price of SDF**: \( p(m) = E[m] = \frac{E[R_{msmp}]}{E[R_{msmp}^2]} \)

### Expected Results

For the **1927-2013 subsample**:
- We can compute the ex-post pricing kernel and verify its properties
- Compare MSMP weights and returns with the full sample

For the **1927-2024 full sample**:
- Compute the ex-post pricing kernel
- Compare with the subsample to see how extending the period affects the pricing kernel

### Key Questions to Address

1. **Does the pricing kernel price assets correctly?**
   - Check: \( E[m \cdot R_i] \approx 1 \) for all portfolios \( i \)

2. **How does the pricing kernel change between periods?**
   - Compare \( p(m) \), \( E[m] \), \( Var(m) \) between 1927-2013 and 1927-2024

3. **Is comparing MSMP weights sufficient?**
   - MSMP weights tell us about portfolio composition
   - Pricing kernel tells us about asset pricing performance
   - Both are important: weights show composition stability, pricing kernel shows pricing performance

## Implementation Notes

The pricing kernel analysis requires:
1. Loading portfolio return data
2. Applying MSMP weights to compute MSMP return series
3. Computing the pricing kernel from MSMP returns
4. Verifying pricing properties: \( E[m \cdot R_i] = 1 \)

This analysis complements the MSMP weight comparison by providing insight into the **pricing performance** of the ex-post pricing kernel in both periods.

## Files

- **MSMP weights**: `results/combined/msmp_weights_combined.csv`
- **Analysis script**: `analyze_expost_pricing_kernel.py`
- **This document**: `PRICING_KERNEL_ANALYSIS.md`

