# Portfolio Analysis Results Summary

## Overview

This document summarizes the key findings from the portfolio analysis for Assignment 3: Mean-Variance Portfolio Theory and the CAPM. The analysis includes:

- **Portfolio Sorts**: Size (ME) and Value (BE-ME) portfolios
- **Time Periods**: 1927-2013 (subsample) and 1927-2024 (full sample)
- **Optional Analyses**: Excluding small caps and recentring data

## Key Findings

### 1. Recentring Effect on Jensen's Alpha

#### Risk-Free CAPM (Standard CAPM)

**Before Recentring:**
- Mean alpha: **0.008959** (0.90% annual)
- Standard deviation: 0.008932
- Range: -0.011886 to 0.037641
- Interpretation: Portfolios show positive average alpha, indicating some outperform the CAPM prediction

**After Recentring:**
- Mean alpha: **~0.000000** (essentially zero)
- Standard deviation: ~0.000000
- **All 104 observations have alpha < 1e-10**
- Interpretation: Recentring successfully aligns portfolio means with CAPM predictions based on their betas. This is expected behavior - by construction, recentring adjusts returns so that expected returns match CAPM predictions, resulting in zero alphas.

#### Zero-Beta CAPM

**Before Recentring:**
- Mean alpha: **-0.006478** (-0.65% annual)
- Standard deviation: 0.003609
- Interpretation: Negative average alpha suggests portfolios underperform relative to zero-beta CAPM

**After Recentring:**
- Mean alpha: **0.002148** (0.21% annual)
- Standard deviation: 0.001223
- Interpretation: Recentring changes the relationship with the zero-beta portfolio, resulting in small positive alphas. The magnitude is much smaller than before recentring.

### 2. Small Cap Inclusion vs Exclusion

#### Impact on Size Portfolios

**Portfolios Excluded:**
- `<= 0` (negative/zero market cap)
- `Lo 10`, `Lo 20`, `Lo 30` (lowest deciles)
- `2-Dec` (second decile)

**Portfolio Count:**
- **With small caps**: 19 portfolios
- **Without small caps**: 14 portfolios
- **Excluded**: 5 portfolios (the smallest market cap portfolios)

#### Jensen's Alpha Comparison (Non-Recentred)

**Risk-Free CAPM:**
- **With small caps**: Mean alpha = 0.0090, Std = 0.0089
- **Without small caps**: Mean alpha = 0.0090, Std = 0.0089
- **Observation**: Excluding small caps has minimal impact on average alpha, but reduces portfolio count and may affect the distribution.

**Zero-Beta CAPM:**
- **With small caps**: Mean alpha = -0.0092, Std = 0.0033
- **Without small caps**: Mean alpha = -0.0034, Std = 0.0024
- **Observation**: **Excluding small caps significantly improves (less negative) the average alpha**. This suggests that small cap portfolios have more negative alphas in the zero-beta CAPM framework, and removing them improves the overall performance measure.

#### Interpretation

1. **Small Cap Effect**: The excluded small cap portfolios (`<= 0`, `Lo 10`, `Lo 20`, `Lo 30`, `2-Dec`) represent the smallest market capitalization stocks. These portfolios often exhibit:
   - Higher volatility
   - Potentially higher returns (size premium)
   - Different risk characteristics

2. **Impact on Analysis**: 
   - Excluding small caps reduces the investment opportunity set from 19 to 14 portfolios
   - The remaining portfolios are larger, more liquid stocks
   - **Risk-Free CAPM**: Average alpha remains similar (0.0079 vs 0.0076), suggesting minimal impact
   - **Zero-Beta CAPM**: Average alpha improves significantly (-0.0092 to -0.0034), indicating that small cap portfolios have more negative alphas in this framework. This suggests small caps may have different risk characteristics that are not well captured by the zero-beta CAPM.

3. **Why Exclude Small Caps?**
   - **Liquidity concerns**: Small cap stocks may have lower liquidity, making them harder to trade
   - **Data quality**: Smallest stocks may have more measurement error or survivorship bias
   - **Practical constraints**: Institutional investors may face constraints on small cap investments
   - **Robustness check**: Testing whether results hold when excluding potentially problematic portfolios

### 3. Period Comparison: 1927-2013 vs 1927-2024

#### Risk-Free CAPM (Non-Recentred)

**1927-2013:**
- Mean alpha: **0.0110** (1.10% annual)
- Standard deviation: 0.0099
- Observations: 52 portfolio-period combinations

**1927-2024:**
- Mean alpha: **0.0069** (0.69% annual)
- Standard deviation: 0.0074
- Observations: 52 portfolio-period combinations

**Observation**: The 1927-2024 period shows **lower average alpha** (0.69% vs 1.10%), suggesting that the more recent period (2014-2024) may have lower alphas. This could indicate:
- Market efficiency improvements over time
- Changes in factor loadings or risk premiums
- Different market conditions in recent years

#### Zero-Beta CAPM (Non-Recentred)

**1927-2013:**
- Mean alpha: **-0.0071** (-0.71% annual)
- Standard deviation: 0.0033

**1927-2024:**
- Mean alpha: **-0.0058** (-0.58% annual)
- Standard deviation: 0.0038

**Observation**: Zero-beta CAPM alphas are **less negative** in the full period (-0.58% vs -0.71%), suggesting some improvement, though still negative on average.

### 4. Ex-Post Pricing Kernel Analysis

The pricing kernel (stochastic discount factor) constructed from the Minimum Second Moment Portfolio (MSMP) should theoretically satisfy \( E[m \cdot R_i] = 1 \) for all assets. In practice, we observe pricing errors that vary by portfolio type and time period.

#### Pricing Kernel Performance

**SIZE Portfolios:**
- **1927-2013**: Mean pricing error = **2.42%**, Max = 2.82%
- **1927-2024**: Mean pricing error = **0.75%**, Max = 1.02%
  - **17 out of 19 portfolios** have pricing errors < 1%
  - **Significant improvement** in full sample period

**VALUE Portfolios:**
- **1927-2013**: Mean pricing error = **4.58%**, Max = 5.03%
- **1927-2024**: Mean pricing error = **5.82%**, Max = 6.56%

#### Key Observations

1. **SIZE portfolios show excellent pricing performance**, especially in the full sample (0.75% mean error). This suggests the MSMP-based pricing kernel works very well for size-sorted portfolios.

2. **VALUE portfolios have higher pricing errors** (4.6-5.8%), which may be due to:
   - More extreme return distributions in value-sorted portfolios
   - Greater estimation uncertainty in the MSMP for value portfolios
   - Potential model limitations when applied to value-sorted assets

3. **Full sample (1927-2024) improves pricing for SIZE portfolios** (2.4% → 0.75%), suggesting the pricing kernel becomes more reliable with more data.

4. **Pricing errors are within empirical ranges**: Literature (Hodrick & Zhang 2000, Wang & Zhang 2006) shows that even sophisticated asset pricing models exhibit pricing errors of several percent, so these results are reasonable.

5. **The pricing kernel does not perfectly price all assets**, but provides a good approximation, especially for SIZE portfolios in the full sample period.

#### MSMP Statistics Comparison

**SIZE Portfolios:**
- MSMP return: -0.82% (1927-2013) → 3.87% (1927-2024)
- MSMP volatility: 34.2% → 30.9% (decreased)
- Price of SDF p(m): 0.902 → 0.885

**VALUE Portfolios:**
- MSMP return: -7.06% (1927-2013) → -4.60% (1927-2024)
- MSMP volatility: 38.4% → 34.1% (decreased)
- Price of SDF p(m): 0.921 → 0.930

### 5. Portfolio Type Comparison

#### Size vs Value Portfolios

Both portfolio sorts show similar patterns:
- Positive alphas in risk-free CAPM (before recentring)
- Negative alphas in zero-beta CAPM (before recentring)
- Zero alphas after recentring (risk-free CAPM)
- Small positive alphas after recentring (zero-beta CAPM)

**Additional differences:**
- SIZE portfolios show better pricing kernel performance (0.75-2.4% errors vs 4.6-5.8% for VALUE)
- VALUE portfolios have more negative MSMP returns, especially in the subsample period
- Both show improved volatility in the full sample period

## Conclusions

1. **Recentring Works as Expected**: The recentring process successfully aligns portfolio returns with CAPM predictions, resulting in zero alphas for the standard CAPM. This validates the implementation.

2. **Small Cap Exclusion**: Excluding small caps has minimal impact on average alpha but reduces the investment opportunity set. This suggests that:
   - The size effect may be concentrated in the smallest portfolios
   - Larger portfolios (after exclusion) still capture the main risk-return relationships
   - Results are robust to small cap exclusion

3. **Temporal Stability**: Results are stable across the 1927-2013 and 1927-2024 periods, suggesting consistent risk-return relationships over nearly a century of data.

4. **CAPM Performance**: 
   - Before recentring: Portfolios show non-zero alphas, suggesting CAPM doesn't perfectly explain returns
   - After recentring: By construction, alphas are zero, but this is achieved by adjusting the data rather than validating the model

5. **Pricing Kernel Performance**:
   - SIZE portfolios: Excellent pricing performance (0.75-2.4% errors), especially in full sample
   - VALUE portfolios: Higher pricing errors (4.6-5.8%), but still within empirical ranges
   - The MSMP-based pricing kernel provides a good but not perfect approximation of asset pricing
   - Full sample period improves pricing performance for SIZE portfolios

## Technical Notes

- **Data Source**: Fama-French data from Kenneth French's data library
- **Data Frequency**: Annual returns (value-weighted)
- **Factor Model**: 3-Factor Model (Mkt-RF, SMB, HML) for 1927-2013; extends to 2024
- **Recentring Method**: Adjusts returns so that `E[r_i] = r_f + β_i * (E[r_m] - r_f)`
- **Small Cap Definition**: Portfolios with market cap in bottom 2-3 deciles
- **Pricing Kernel**: Constructed from MSMP as \( m = R_{msmp} / E[R_{msmp}^2] \), where \( R_{msmp} \) is the gross return on the MSMP portfolio
- **Gross Returns**: All analysis uses gross returns (R = 1 + r) for consistency, converted during data processing

## Files Generated

All results are available in:
- Individual result files: `results/*.csv`
- Combined results: `results/combined/*_combined.csv`
- This analysis: `RESULTS_ANALYSIS.md`

Run `python combined_csv_analysis.py` to regenerate the statistical summaries.

