# Assignment 3: Mean-Variance Portfolio Theory and the CAPM

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

## Next Steps

After downloading and processing the data, you can proceed with:
1. Constructing MV frontiers
2. Finding MSMP portfolios
3. Finding optimal portfolios for CRRA investors
4. Estimating Jensen's alpha

See `PTAP 50040 Assignment.md` for the complete task list.

