# Assignment Compliance Check

This document compares our implementation against the assignment requirements.

## Required Tasks (1927-2013 and 1927-2024)

### ✅ 1. Construct MV Frontier
**Requirement**: Construct the (ex-post) MV frontier for the case of free portfolio. No riskless asset, budget constraint only.

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `construct_mv_frontier()`
- ✅ Generates 100 portfolios along the efficient frontier
- ✅ Uses only budget constraint (sum of weights = 1)
- ✅ No riskless asset included
- ✅ Results saved for both periods and both portfolio types

**Output**: `mv_frontier_*.csv` files in `results/` directory

---

### ✅ 2. Find MSMP Portfolio
**Requirement**: Find Minimum Second Moment Portfolio (MSMP)

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `find_msmp()`
- ✅ Minimizes E[(w'R)²] = w'(Σ_R + μ_R μ_R')w subject to budget constraint
- ✅ Uses gross returns (R = 1 + r) as required
- ✅ Results saved for both periods and both portfolio types

**Output**: `msmp_weights_*.csv` files in `results/` directory

**Key Results**:
- SIZE (1927-2013): MSMP return = -0.82%, Volatility = 34.2%
- SIZE (1927-2024): MSMP return = 3.87%, Volatility = 30.9%
- VALUE (1927-2013): MSMP return = -7.06%, Volatility = 38.4%
- VALUE (1927-2024): MSMP return = -4.60%, Volatility = 34.1%

---

### ✅ 3. Find Optimal CRRA Portfolio
**Requirement**: Find optimal portfolio for CRRA investor with RRA=4

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `find_optimal_crra_portfolio()`
- ✅ Maximizes E[U(w'R)] = E[(w'R)^(1-γ)/(1-γ)] where γ = RRA = 4
- ✅ Uses gross returns throughout
- ✅ Results saved for both periods and both portfolio types

**Output**: `optimal_weights_*.csv` files in `results/` directory

---

### ✅ 4. Find Zero-Beta Portfolio
**Requirement**: Find Zero-β portfolio for optimal portfolio above

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `find_zero_beta_portfolio()`
- ✅ Finds portfolio w_z such that Cov(w_z'R, w_opt'R) = 0
- ✅ Uses robust optimization with multiple initial guesses
- ✅ Results saved for both periods and both portfolio types

**Output**: `zero_beta_weights_*.csv` files in `results/` directory

---

### ✅ 5. Estimate Jensen's Alpha
**Requirement**: Estimate Jensen's α for each portfolio in:
- Zero-β CAPM version
- R_f version (using Kenneth French risk-free rate data)

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `estimate_jensens_alpha()`
- ✅ Zero-β CAPM: r_i - r_z = α_i + β_i(r_m_opt - r_z)
  - Uses optimal portfolio as market proxy (not Mkt-RF)
  - Uses zero-beta portfolio return
- ✅ Standard CAPM: r_i - r_f = α_i + β_i(r_m - r_f)
  - Uses Mkt-RF as market proxy
  - Uses risk-free rate from Fama-French factors
- ✅ Results saved for both periods and both portfolio types

**Output**: 
- `jensens_alpha_zerobeta_*.csv` files
- `jensens_alpha_rf_*.csv` files

**Key Results** (Non-Recentred):
- Risk-Free CAPM: Mean alpha = 0.69-1.10% (varies by period)
- Zero-Beta CAPM: Mean alpha = -0.58% to -0.71% (negative, as expected)

---

## Optional Tasks

### ✅ 6. Exclude Small Caps
**Requirement**: Find Jensen's α after excluding small caps

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `exclude_small_caps()`
- ✅ Excludes: `<= 0`, `Lo 10`, `Lo 20`, `Lo 30`, `2-Dec`
- ✅ Reduces size portfolios from 19 to 14
- ✅ Results show improved zero-beta CAPM alphas (-0.0092 → -0.0034)

**Output**: Results with `_no_small_caps` suffix

---

### ✅ 7. Recentre Data
**Requirement**: Recentre data set, align means with CAPM and betas, then find Jensen's α again

**Our Implementation**:
- ✅ Implemented in `portfolio_analysis.py` → `recentre_returns()`
- ✅ Adjusts returns so that E[r_i] = r_f + β_i * (E[r_m] - r_f)
- ✅ By construction, risk-free CAPM alphas become zero
- ✅ Zero-beta CAPM alphas become small positive values (0.21% mean)

**Output**: Results with `_recentred` suffix

**Key Results**:
- Risk-Free CAPM (Recentred): Mean alpha ≈ 0.000000 (all < 1e-10) ✓
- Zero-Beta CAPM (Recentred): Mean alpha = 0.21% (small positive)

---

## Data Requirements

### ✅ Portfolio Data
- ✅ 10 Size Portfolios (Portfolios Formed on ME)
- ✅ Second Portfolio Sort: Book-to-Market (BE-ME)
- ✅ Data extended from 1927 to 2024 (most recent available)
- ✅ Using Annual Value-Weighted Returns (as specified)

### ✅ Factor Data
- ✅ 3-Factor Model (F-F_Research_Data_Factors) for 1927-2013
- ✅ 5-Factor Model downloaded (for potential future use)
- ✅ Risk-free rate included in factors file
- ✅ Data extended to 2024

### ✅ Data Processing
- ✅ Automatic ZIP file extraction
- ✅ Multi-section CSV parsing (handles monthly/annual, value/equal weighted)
- ✅ Date parsing for various formats (YYYYMM, YYYY, YYYY-MM-DD)
- ✅ **Gross returns conversion**: All portfolio returns and RF converted to gross returns (R = 1 + r) during processing
- ✅ Consistent use of gross returns throughout all analysis

---

## Results Organization

### ✅ Consolidated Results
- ✅ `jensens_alpha_rf_combined.csv` - All risk-free CAPM alphas
- ✅ `jensens_alpha_zerobeta_combined.csv` - All zero-beta CAPM alphas
- ✅ `msmp_weights_combined.csv` - All MSMP weights
- ✅ `optimal_weights_combined.csv` - All optimal CRRA portfolio weights
- ✅ `zero_beta_weights_combined.csv` - All zero-beta portfolio weights

### ✅ Analysis Documents
- ✅ `RESULTS_ANALYSIS.md` - Comprehensive results summary
- ✅ `PRICING_KERNEL_ANALYSIS.md` - Ex-post pricing kernel analysis (extension)
- ✅ `combined_csv_analysis.py` - Statistical analysis of combined results

---

## Extensions Beyond Requirements

### ✅ Ex-Post Pricing Kernel Analysis
**Not explicitly required, but relevant to theory**:
- ✅ Computed pricing kernel from MSMP: m = R_msmp / E[R_msmp^2]
- ✅ Verified pricing properties: E[m * R_i] ≈ 1
- ✅ Pricing errors: SIZE (0.75-2.4%), VALUE (4.6-5.8%)
- ✅ Documented in `PRICING_KERNEL_ANALYSIS.md`

**Rationale**: The pricing kernel is theoretically related to MSMP and provides insight into asset pricing performance.

---

## Comparison with Professor's Original Solution

### Expected Differences
1. **Updated Data**: Fama-French data for 1927-2013 has been revised/updated since original assignment
2. **Extended Period**: Data now extends to 2024 (vs. 2013 in original)
3. **Data Processing**: May have minor differences in date parsing or data handling
4. **Rounding**: Minor numerical differences due to floating-point precision

### Our Approach
- ✅ All results saved to CSV for easy comparison
- ✅ Combined results files for systematic comparison
- ✅ Documentation of methodology for reproducibility
- ✅ Note in README about expected differences

---

## Code Quality & Reproducibility

### ✅ Scripts
- ✅ Modular design with clear function separation
- ✅ Command-line arguments for flexibility
- ✅ Comprehensive error handling
- ✅ Progress reporting and logging
- ✅ Parallel execution support for speed

### ✅ Documentation
- ✅ README with setup instructions
- ✅ Inline code comments
- ✅ Analysis markdown files
- ✅ LLM disclosure statement

### ✅ Reproducibility
- ✅ All scripts use relative paths
- ✅ Data processing is deterministic
- ✅ Results can be regenerated by running scripts in order
- ✅ Requirements.txt for dependency management

---

## Summary

**All Required Tasks**: ✅ Complete
- MV Frontier construction
- MSMP portfolio
- Optimal CRRA portfolio (RRA=4)
- Zero-beta portfolio
- Jensen's alpha (both versions)

**All Optional Tasks**: ✅ Complete
- Excluding small caps
- Recentring data

**Data Requirements**: ✅ Complete
- Two portfolio sorts (Size and BE-ME)
- Factor models and risk-free rate
- Extended to 2024

**Results Organization**: ✅ Complete
- Individual result files
- Consolidated CSV files
- Analysis documents

**Extensions**: ✅ Complete
- Ex-post pricing kernel analysis
- Comprehensive statistical analysis
- Period comparisons

---

## Verification Checklist

- [x] All required analyses completed for 1927-2013
- [x] All required analyses completed for 1927-2024
- [x] Both portfolio types analyzed (Size and Value)
- [x] Optional sections completed
- [x] Results saved and organized
- [x] Documentation complete
- [x] Code is reproducible
- [x] LLM disclosure included

**Status**: ✅ **FULLY COMPLIANT** with assignment requirements

