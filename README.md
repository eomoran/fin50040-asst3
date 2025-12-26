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

## Customizing Portfolio Sort

To change the second portfolio sort, edit `download_famafrench_data.py` and modify the `sort_type` parameter in the `download_second_portfolio_sort()` call. Available options:
- `"value"` - Book-to-Market (BE/ME)
- `"momentum"` - Prior (2-12) Returns
- `"profitability"` - Operating Profitability
- `"investment"` - Investment
- `"beta"` - Beta

## Running Analyses

4. **Run all analyses:**
   ```bash
   python run_all_analyses.py
   ```
   
   This will run all required and optional analyses:
   - Size and Value portfolios for 1927-2013 and 1927-2024
   - Optional: Excluding small caps
   - Optional: Recentring data

5. **Combine results:**
   ```bash
   python combine_results.py
   ```
   
   This creates consolidated CSV files in `results/combined/` that combine all configurations:
   - `jensens_alpha_rf_combined.csv` - Jensen's alpha (risk-free version)
   - `jensens_alpha_zerobeta_combined.csv` - Jensen's alpha (zero-beta version)
   - `msmp_weights_combined.csv` - MSMP portfolio weights
   - `optimal_weights_combined.csv` - Optimal CRRA portfolio weights
   - `zero_beta_weights_combined.csv` - Zero-beta portfolio weights

## Results Comparison

The results for the 1927-2013 subsample should be compared with the professor's original solution (see `Assignment 3 - Day 1 MPT CAPM.xlsx`). Any differences are likely due to:
- **Updates to underlying Fama-French data**: The historical data for the 1927-2013 period has been revised/updated by Fama-French since the original assignment was created (not just extended to 2024)
- Dataset extension: Fama-French data has been extended to 2024
- Minor differences in data processing or rounding

See `PTAP 50040 Assignment.md` for the complete task list.

