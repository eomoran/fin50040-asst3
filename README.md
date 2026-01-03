# Assignment 3: Mean-Variance Portfolio Theory and the CAPM

## LLM Disclosure

This assignment was completed with the assistance of an AI coding assistant (Claude via Cursor). The assistant helped with:
- Script development for data downloading and processing
- Implementation of portfolio optimization algorithms
- Data analysis and result generation
- Code debugging and error handling

All analysis results, interpretations, and conclusions are my own work.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if using conda:
   ```bash
   conda activate fin50040
   pip install -r requirements.txt
   ```

2. **Download Fama-French data:**
   ```bash
   python download_famafrench_data.py
   ```
   
   This will download:
   - 10 Size Portfolios (Portfolios Formed on ME)
   - Second Portfolio Sort (default: Book-to-Market, can be changed in script)
   - Fama-French 3-Factor and 5-Factor Models
   - Risk-free rate data (included in factors file)

3. **Process the downloaded data:**
   ```bash
   python process_famafrench_data.py
   ```
   
   This will extract ZIP files and convert them to CSV format for easier analysis.
   **Note**: Portfolio returns are automatically converted to gross returns (R = 1 + r)
   during processing, and RF is converted to gross return (R_f = 1 + r_f) in factor files.
   All analysis scripts work with gross returns throughout.

## Data Files

All data will be downloaded to the `data/` directory:
- `Portfolios_Formed_on_ME.zip` - 10 Size Portfolios
- `Portfolios_Formed_on_BE-ME.zip` - Book-to-Market Portfolios (or other sort)
- `F-F_Research_Data_Factors.zip` - 3-Factor Model (includes RF)
- `F-F_Research_Data_5_Factors_2x3.zip` - 5-Factor Model

Processed data is saved to `data/processed/` directory.

## Customizing Portfolio Sort

To change the second portfolio sort, edit `download_famafrench_data.py` and modify the `sort_type` parameter in the `download_second_portfolio_sort()` call. Available options:
- `"value"` - Book-to-Market (BE/ME)
- `"momentum"` - Prior (2-12) Returns
- `"profitability"` - Operating Profitability
- `"investment"` - Investment
- `"beta"` - Beta

## Running Analyses

### Quick Start: Run All Analyses

**Run all analyses for both portfolio types and both periods:**
```bash
python run_all_portfolio_analyses.py --all --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0
```

**With closed-form solutions:**
```bash
python run_all_portfolio_analyses.py --all --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0 --closed-form
```

**With comparison cones in plots:**
```bash
python run_all_portfolio_analyses.py --all --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0 --plot-alt-cones
```

### Individual Scripts

The analysis consists of 6 steps:

1. **Construct Investment Opportunity Set (IOS):**
   ```bash
   python construct_investment_opportunity_set.py --portfolio-type size --start-year 1927 --end-year 2024
   ```
   Use `--closed-form` for analytical solution, or omit for optimization-based method.

2. **Find MSMP Portfolio:**
   ```bash
   python find_msmp_portfolio.py --portfolio-type size --start-year 1927 --end-year 2024
   ```
   Use `--closed-form` for analytical solution using RR' matrix.

3. **Find Optimal CRRA Portfolio:**
   ```bash
   python find_optimal_crra_portfolio.py --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0
   ```
   Use `--closed-form` for analytical solution using Lagrange multipliers.

4. **Find Zero-Beta Portfolio:**
   ```bash
   python find_zero_beta_portfolio.py --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0
   ```
   By default, uses optimization with `on_frontier=True` (ensures ZBP is on the frontier).
   Use `--closed-form` for analytical formula (also on frontier by construction).

5. **Estimate Jensen's Alpha:**
   ```bash
   python estimate_jensens_alpha.py --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0
   ```
   Estimates alpha for both Zero-Beta CAPM and Risk-Free CAPM models.

6. **Plot Portfolio Frontier:**
   ```bash
   python plot_portfolio_frontier.py --portfolio-type size --start-year 1927 --end-year 2024 --rra 4.0
   ```
   Use `--closed-form` to match closed-form calculations.
   Use `--plot-alt-cones` to include comparison lines (R_f ↔ MSMP and R_ZBP ↔ MSMP).

### Command-Line Options

**Common options:**
- `--portfolio-type {size,value}` - Portfolio sort type
- `--start-year YYYY` - Start year (e.g., 1927)
- `--end-year YYYY` - End year (e.g., 2013 or 2024)
- `--rra FLOAT` - Relative Risk Aversion coefficient (default: 4.0)
- `--closed-form` - Use analytical/closed-form solutions instead of optimization
- `--plot-alt-cones` - Plot alternative comparison lines (for plotting script)

**Wrapper script options:**
- `--all` - Run all steps (makes other arguments optional)
- `--skip-plot` - Skip the plotting step
- `--skip-jensens` - Skip Jensen's alpha estimation

## Results

All results are saved to the `results/` directory:

### Portfolio Weights
- `ios_summary.csv` - Investment Opportunity Set summary
- `investment_opportunity_set_{type}_{start}_{end}.csv` - Full IOS curve
- `msmp_weights_{type}_{start}_{end}.csv` - MSMP portfolio weights
- `optimal_crra_weights_{type}_{start}_{end}.csv` - Optimal CRRA portfolio weights
- `zbp_weights_{type}_{start}_{end}.csv` - Zero-Beta Portfolio weights

### Summary Files
- `msmp_summary.csv` - MSMP statistics (return, volatility)
- `optimal_crra_summary.csv` - Optimal CRRA portfolio statistics
- `zbp_summary.csv` - Zero-Beta Portfolio statistics
- `jensens_alpha_summary.csv` - Jensen's alpha results for all portfolios

### Individual Alpha Files
- `jensens_alpha_zerobeta_{type}_{start}_{end}.csv` - Zero-Beta CAPM alphas
- `jensens_alpha_riskfree_{type}_{start}_{end}.csv` - Risk-Free CAPM alphas

### Plots
Plots are saved to the `plots/` directory:
- `portfolio_frontier_{type}_{start}_{end}.png` - High-resolution PNG (300 DPI)
- `portfolio_frontier_{type}_{start}_{end}.pdf` - Vector PDF
- `portfolio_frontier_{type}_{start}_{end}_closed_form.png` - When using `--closed-form` flag

Plots show:
- Investment Opportunity Set (IOS) - the full mean-variance frontier
- Minimum Variance Portfolio (MVP)
- Minimum Second Moment Portfolio (MSMP)
- Optimal CRRA Portfolio (RRA=4.0)
- Zero-Beta Portfolio (ZBP)
- Tangency Portfolio (TP)
- Capital Market Line (CAL)
- Asymptotes and comparison lines (when enabled)

## Assignment Requirements

This implementation addresses all required tasks from the assignment:

1. ✅ **Download data** (1927-2020, extended to 2024) for two portfolio sorts
2. ✅ **Download benchmark factors** (Fama-French 3-Factor Model)
3. ✅ **Construct MV frontier** for free portfolio formation (budget constraint only)
4. ✅ **Find Minimum Second Moment Portfolio (MSMP)**
5. ✅ **Find optimal portfolio** for CRRA investor with RRA=4
6. ✅ **Find Zero-Beta Portfolio** of the optimal portfolio
7. ✅ **Estimate Jensen's alpha** for each portfolio:
   - Zero-Beta CAPM version
   - Risk-Free CAPM version (using Kenneth French risk-free rate)
8. ✅ **Optional: Exclude small caps** and repeat alpha estimation
9. ✅ **Optional: Recentre data** and repeat alpha estimation

All tasks completed for both periods:
- 1927-2013 (to compare with professor's solution)
- 1927-2024 (full sample, extended from 2020)

## Results Comparison

The results for the 1927-2013 subsample should be compared with the professor's original solution (see `Assignment 3 - Day 1 MPT CAPM.xlsx`). Any differences are likely due to:
- **Updates to underlying Fama-French data**: The historical data for the 1927-2013 period has been revised/updated by Fama-French since the original assignment was created (not just extended to 2024)
- Dataset extension: Fama-French data has been extended to 2024
- Minor differences in data processing or rounding
- Method differences: Closed-form vs optimization methods may yield slightly different results

## Implementation Notes

### Closed-Form vs Optimization

All calculations support both methods:
- **Closed-form**: Analytical solutions using matrix algebra (faster, exact)
- **Optimization**: Numerical optimization using `scipy.optimize` (more flexible)

Use the `--closed-form` flag consistently across all scripts for comparable results.

### Benchmark Factor

For Jensen's alpha estimation, the **optimal CRRA portfolio** is used as the benchmark/market portfolio (not a market index). This is the portfolio that maximizes CRRA utility with RRA=4.

### Zero-Beta Portfolio

The Zero-Beta Portfolio is calculated with respect to the **optimal CRRA portfolio** (not the Tangency Portfolio). It satisfies:
- `Cov(R_z, R_CRRA) = 0`
- Lies on the frontier (hyperbola boundary)

## Project Structure

```
asst3/
├── data/                    # Raw and processed data
│   ├── processed/          # Processed CSV files
│   └── *.zip               # Downloaded ZIP files
├── results/                # Analysis results (CSV files)
├── plots/                  # Generated plots (PNG/PDF)
├── scripts_hidden/         # Legacy/alternative scripts
├── construct_investment_opportunity_set.py
├── find_msmp_portfolio.py
├── find_optimal_crra_portfolio.py
├── find_zero_beta_portfolio.py
├── estimate_jensens_alpha.py
├── plot_portfolio_frontier.py
├── plot_comparison_cones.py
├── run_all_portfolio_analyses.py
└── README.md
```

## Additional Documentation

- `PTAP 50040 Assignment.md` - Task tracking checklist
- `ASSIGNMENT_COMPLIANCE.md` - Detailed compliance check
- `TASK_BREAKDOWN.md` - Implementation breakdown
- `DATA_FORMAT_SUMMARY.md` - Data format details
- `RESULTS_ANALYSIS.md` - Results analysis and interpretation
- `PRICING_KERNEL_ANALYSIS.md` - Extension: Pricing kernel analysis

## References

- See `Assignment 3 - Day 1 MPT CAPM.xlsx` for the professor's original solution
- Fama-French data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
