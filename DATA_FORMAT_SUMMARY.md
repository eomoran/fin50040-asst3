# Data Format Summary

## Processed Data Location
All processed data is in `data/processed/` directory.

## Portfolio Returns Data

### File Naming Pattern
- Size portfolios: `Portfolios_Formed_on_ME_Portfolios_Formed_on_ME_Value_Weight_Returns___Annual_from_January_to_December.csv`
- Value portfolios: `Portfolios_Formed_on_BE-ME_Portfolios_Formed_on_BE-ME_Value_Weight_Returns___Annual_from_January_to_December.csv`

### Format
- **Index**: Dates (YYYY-MM-DD format, annual frequency)
  - Example: `1927-01-01`, `1928-01-01`, ..., `2024-01-01`
- **Columns**: Portfolio names (e.g., `'<= 0'`, `'Lo 30'`, `'Med 40'`, `'Hi 30'`, `'Lo 20'`, `'Qnt 2'`, etc.)
- **Values**: **GROSS RETURNS** (R = 1 + r)
  - Example: 1.05 means 5% net return
  - Example: 0.95 means -5% net return
- **Shape**: (98, 19) for full sample (1927-2024)
  - 98 years of data
  - 19 portfolios (including `'<= 0'` which may have NaN values)

### Important Notes
- Data is already converted to **gross returns** during processing
- Some portfolios may have NaN values for early years
- Use annual value-weighted returns as specified in assignment

## Factor Data

### File Naming Pattern
- 3-Factor Model: `F-F_Research_Data_Factors_F-F_Research_Data_Factors_Annual_Factors_January-December.csv`
- 5-Factor Model: `F-F_Research_Data_5_Factors_2x3_F-F_Research_Data_5_Factors_2x3_Annual_Factors_January-December.csv`

### Format
- **Index**: Dates (YYYY-MM-DD format, annual frequency)
- **Columns**:
  - `Mkt-RF`: Market excess return (excess return, not gross)
  - `SMB`: Small Minus Big (excess return)
  - `HML`: High Minus Low (excess return)
  - `RMW`: Robust Minus Weak (excess return, 5-factor only)
  - `CMA`: Conservative Minus Aggressive (excess return, 5-factor only)
  - `RF`: Risk-free rate (**GROSS RETURN**, R = 1 + r)

### Important Notes
- **`RF` is in GROSS returns** (R_f = 1 + r_f)
  - Values are > 1 (e.g., 1.0312, 1.0356)
  - This is correct: risk-free rate as gross return
- **All other factors are EXCESS returns** (r, not R)
  - `Mkt-RF` = r_m - r_f (can be negative, e.g., -0.1958)
  - `SMB`, `HML`, `RMW`, `CMA` are also excess returns (can be negative)
  - This is standard: factors are excess returns, not gross returns
- For 1927-2013 period, use 3-factor model (covers full period)
- For 1927-2024 period, can use either 3-factor or 5-factor model

**Summary:**
- Portfolio returns: **R** (gross returns, R = 1 + r)
- Risk-free rate (RF): **R_f** (gross returns, R_f = 1 + r_f)
- Factor returns (Mkt-RF, SMB, HML, etc.): **r** (excess returns, not gross)

## Data Loading Example

```python
import pandas as pd
from pathlib import Path

# Load portfolio returns
portfolio_file = Path('data/processed/Portfolios_Formed_on_ME_Portfolios_Formed_on_ME_Value_Weight_Returns___Annual_from_January_to_December.csv')
returns = pd.read_csv(portfolio_file, index_col=0, parse_dates=True)

# Filter by date range
returns = returns.loc['1927-01-01':'2013-01-01']

# Remove portfolios with all NaN
returns = returns.dropna(axis=1, how='all')

# Load factors
factor_file = Path('data/processed/F-F_Research_Data_Factors_F-F_Research_Data_Factors_Annual_Factors_January-December.csv')
factors = pd.read_csv(factor_file, index_col=0, parse_dates=True)
factors = factors.loc['1927-01-01':'2013-01-01']

# Extract risk-free rate (already in gross returns)
rf = factors['RF']  # Gross return: R_f = 1 + r_f

# Extract excess returns
mkt_rf = factors['Mkt-RF']  # Excess return: r_m - r_f
```

## Key Points for Analysis

1. **All portfolio returns are in GROSS returns** (R = 1 + r)
   - When computing moments: Use directly
   - When displaying: Convert to net returns (r = R - 1)

2. **Risk-free rate is in GROSS returns** (R_f = 1 + r_f)
   - When computing excess returns: `R_i - R_f` (gross excess)
   - Or convert to net: `(R_i - 1) - (R_f - 1) = r_i - r_f`

3. **Factor returns are EXCESS returns** (r, not R)
   - `Mkt-RF` = r_m - r_f (excess return)
   - Use directly in regressions

4. **Date filtering**: Use pandas date indexing
   - `returns.loc['1927-01-01':'2013-01-01']`
   - Ensure dates align between returns and factors

