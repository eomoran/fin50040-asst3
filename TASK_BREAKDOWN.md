# Task Breakdown - Assignment 3

## Current Status

### ✅ Completed Setup
- `download_famafrench_data.py` - Downloads Fama-French data
- `process_famafrench_data.py` - Processes and converts to gross returns
- Data is ready in `data/processed/` directory

### 📋 Tasks to Implement (from PTAP 50040 Assignment.md)

For each portfolio sort (Size and Value) and each period (1927-2013 and 1927-2024):

1. **Construct Investment Opportunity Set**
   - Mean-variance frontier (both efficient and inefficient limbs)
   - No riskless asset, budget constraint only
   - This is the full "U-shape" curve

2. **Find MSMP Portfolio**
   - Minimum Second Moment Portfolio
   - Minimizes E[(w'R)²] = w'(Σ_R + μ_R μ_R')w

3. **Find Optimal CRRA Portfolio**
   - For CRRA investor with RRA=4
   - Maximizes E[U(w'R)] = E[(w'R)^(1-γ)/(1-γ)]

4. **Find Zero-β Portfolio**
   - Zero covariance with optimal portfolio
   - Must be on the investment opportunity set (inefficient limb)

5. **Estimate Jensen's α**
   - Zero-β CAPM version: r_i - r_z = α_i + β_i(r_m - r_z)
   - Risk-free version: r_i - r_f = α_i + β_i(r_m - r_f)

## Data Format Summary

See `DATA_FORMAT_SUMMARY.md` for details.

**Key Points:**
- Portfolio returns: **GROSS returns** (R = 1 + r) in `data/processed/`
- Risk-free rate: **GROSS returns** (R_f = 1 + r_f)
- Factor returns: **EXCESS returns** (r, not R)
- Dates: Annual frequency, YYYY-MM-DD format

## Next Steps

We will create separate scripts for each task:
1. `construct_investment_opportunity_set.py` - Task 1
2. `find_msmp.py` - Task 2
3. `find_optimal_crra.py` - Task 3
4. `find_zero_beta.py` - Task 4
5. `estimate_jensens_alpha.py` - Task 5

Each script will:
- Load data from `data/processed/`
- Accept parameters: portfolio_type, start_year, end_year
- Save results to `results/` directory
- Work with gross returns consistently

