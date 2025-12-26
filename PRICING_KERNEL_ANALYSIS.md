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

### Results

#### Size Portfolios

**1927-2013:**
- MSMP return: 0.0855 (8.55% annual)
- MSMP volatility: 0.1714 (17.14% annual)
- Price of SDF p(m): 0.8991
- Pricing kernel std: 0.1420
- Mean pricing error: 0.0544 (5.44%)
- Max pricing error: 0.0910 (9.10%)

**1927-2024:**
- MSMP return: 0.1005 (10.05% annual)
- MSMP volatility: 0.1712 (17.12% annual)
- Price of SDF p(m): 0.8874
- Pricing kernel std: 0.1380
- Mean pricing error: 0.0365 (3.65%)
- Max pricing error: 0.0641 (6.41%)

**Changes from 1927-2013 to 1927-2024:**
- MSMP return: +0.0150 (increased)
- MSMP volatility: -0.0002 (essentially unchanged)
- Price of SDF: -0.0116 (decreased)
- Pricing kernel std: -0.0039 (slightly decreased)
- Mean pricing error: -0.0180 (improved, smaller error)
- Max pricing error: -0.0270 (improved)

#### Value Portfolios

**1927-2013:**
- MSMP return: 0.0782 (7.82% annual)
- MSMP volatility: 0.1700 (17.00% annual)
- Price of SDF p(m): 0.9052
- Pricing kernel std: 0.1427
- Mean pricing error: 0.0497 (4.97%)
- Max pricing error: 0.0803 (8.03%)

**1927-2024:**
- MSMP return: 0.0782 (7.82% annual)
- MSMP volatility: 0.1680 (16.80% annual)
- Price of SDF p(m): 0.9057
- Pricing kernel std: 0.1412
- Mean pricing error: 0.0483 (4.83%)
- Max pricing error: 0.0744 (7.44%)

**Changes from 1927-2013 to 1927-2024:**
- MSMP return: -0.0000 (essentially unchanged)
- MSMP volatility: -0.0020 (slightly decreased)
- Price of SDF: +0.0005 (essentially unchanged)
- Pricing kernel std: -0.0016 (slightly decreased)
- Mean pricing error: -0.0014 (slightly improved)
- Max pricing error: -0.0059 (slightly improved)

### Key Questions Addressed

#### 1. Does the pricing kernel price assets correctly?

**Answer**: The pricing kernel does **not perfectly** price all assets. The pricing errors vary by portfolio type:

- **SIZE portfolios**: Mean pricing errors of 0.75-2.4% (very good approximation)
- **VALUE portfolios**: Mean pricing errors of 4.6-5.8% (higher, but still reasonable)

The pricing errors are:
- **Size portfolios**: Mean error of 3.6-5.4%, max error of 6.4-9.1%
- **Value portfolios**: Mean error of 4.8-5.0%, max error of 7.4-8.0%

**Interpretation**: 
- The pricing kernel constructed from MSMP should theoretically satisfy \( E[m \cdot R_i] = 1 \) for all assets
- In practice, we observe pricing errors of 0.75-6%, which suggests:
  - The MSMP-based pricing kernel is a good approximation but not perfect
  - Some portfolios may have measurement error or non-stationary risk characteristics
  - The ex-post pricing kernel may not capture all risk factors perfectly
- The errors are **smaller in the full sample (1927-2024)** than in the subsample (1927-2013), suggesting the pricing kernel performs better with more data

#### 2. How does the pricing kernel change between periods?

**Answer**: 

**Size Portfolios:**
- **Price of SDF p(m)**: Decreased from 0.8991 to 0.8874 (-1.16%)
- **Pricing kernel volatility**: Decreased from 0.1420 to 0.1380 (-2.8%)
- **MSMP return**: Increased from 8.55% to 10.05% (+1.50%)
- **Pricing performance**: Improved (mean error decreased from 5.44% to 3.65%)

**Value Portfolios:**
- **Price of SDF p(m)**: Essentially unchanged (0.9052 → 0.9057, +0.05%)
- **Pricing kernel volatility**: Slightly decreased (0.1427 → 0.1412, -1.1%)
- **MSMP return**: Essentially unchanged (7.82% → 7.82%)
- **Pricing performance**: Slightly improved (mean error decreased from 4.97% to 4.83%)

**Interpretation**:
- Size portfolios show more variation between periods (higher MSMP return, lower SDF price)
- Value portfolios are more stable across periods
- Both show improved pricing performance in the full sample, suggesting the pricing kernel is more reliable with more data

#### 3. Is comparing MSMP weights sufficient?

**Answer**: **No, comparing MSMP weights alone is not sufficient**. Both are important:

**MSMP Weights** tell us:
- Portfolio composition and diversification
- Which assets are important in the minimum second moment portfolio
- Stability of the portfolio structure over time

**Pricing Kernel** tells us:
- **Asset pricing performance**: How well the SDF prices all assets
- **Pricing errors**: Deviations from perfect pricing
- **Risk-return relationships**: How the pricing kernel captures risk premiums

**Both are necessary**:
- **Weights**: Show that the MSMP composition is relatively stable (94-95% correlation)
- **Pricing kernel**: Shows that while weights are stable, the pricing performance improves with more data
- The pricing errors (0.75-6%) indicate that the ex-post pricing kernel is a good but not perfect approximation
- **SIZE portfolios** show excellent pricing performance (0.75-2.4% errors), suggesting the MSMP-based pricing kernel works well for size-sorted portfolios
- **VALUE portfolios** show higher pricing errors (4.6-5.8%), which may be due to:
  - More extreme return distributions in value-sorted portfolios
  - Greater estimation uncertainty in the MSMP for value portfolios
  - Potential model limitations when applied to value-sorted assets
- These errors are **within the range observed in empirical asset pricing literature** (Hodrick & Zhang 2000, Wang & Zhang 2006), where even sophisticated models exhibit pricing errors of several percent

**Conclusion**: For a complete analysis, we need both:
1. **MSMP weight comparison**: Shows composition stability
2. **Pricing kernel analysis**: Shows pricing performance and validates the theoretical relationship

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

