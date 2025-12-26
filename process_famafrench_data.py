"""
Process and extract Fama-French data from downloaded ZIP files
"""

import pandas as pd
import zipfile
from pathlib import Path
import re

DATA_DIR = Path("data")

def extract_zip_file(zip_path, output_dir=None):
    """Extract a ZIP file"""
    if output_dir is None:
        output_dir = zip_path.parent / zip_path.stem
    output_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted {zip_path.name} to {output_dir}")
    return output_dir

def parse_famafrench_txt(txt_path, skip_rows=None):
    """
    Parse a Fama-French text file
    Fama-French files typically have:
    - Header rows to skip
    - Date column (YYYYMM format)
    - Multiple return columns
    """
    # Try to auto-detect skip rows
    if skip_rows is None:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20]):
                if re.match(r'^\d{6}', line.strip()):  # Date format YYYYMM
                    skip_rows = i
                    break
    
    if skip_rows is None:
        skip_rows = 0
    
    # Read the file
    df = pd.read_csv(txt_path, skiprows=skip_rows, sep=r'\s+', 
                     na_values=['-99.99', '-999'], engine='python')
    
    # Try to parse date column
    if df.columns[0].isdigit() or len(str(df.iloc[0, 0])) == 6:
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce')
        df = df.rename(columns={date_col: 'Date'})
        df = df.set_index('Date')
    
    return df

def process_size_portfolios():
    """Process the 10 size portfolios data"""
    zip_path = DATA_DIR / "Portfolios_Formed_on_ME.zip"
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found. Please download it first.")
        return None
    
    # Extract ZIP
    extract_dir = extract_zip_file(zip_path)
    
    # Find the main data file (usually ends with .txt)
    txt_files = list(extract_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {extract_dir}")
        return None
    
    # Process the main file (usually the largest or first one)
    main_file = txt_files[0]
    print(f"Processing {main_file.name}...")
    
    df = parse_famafrench_txt(main_file)
    
    # Save processed data
    output_path = DATA_DIR / "size_portfolios_processed.csv"
    df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")
    
    return df

def process_factors():
    """Process Fama-French factors (3-factor model)"""
    zip_path = DATA_DIR / "F-F_Research_Data_Factors.zip"
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found. Please download it first.")
        return None
    
    # Extract ZIP
    extract_dir = extract_zip_file(zip_path)
    
    # Find the main data file
    txt_files = list(extract_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {extract_dir}")
        return None
    
    main_file = txt_files[0]
    print(f"Processing {main_file.name}...")
    
    df = parse_famafrench_txt(main_file)
    
    # Save processed data
    output_path = DATA_DIR / "factors_processed.csv"
    df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")
    
    return df

def filter_date_range(df, start_year=1927, end_year=2024):
    """Filter dataframe to specific date range"""
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask]

def main():
    """Main processing function"""
    print("=" * 60)
    print("Processing Fama-French Data")
    print("=" * 60)
    print()
    
    # Process size portfolios
    print("\n[1/2] Processing Size Portfolios...")
    size_df = process_size_portfolios()
    if size_df is not None:
        print(f"Shape: {size_df.shape}")
        print(f"Date range: {size_df.index.min()} to {size_df.index.max()}")
        print(f"Columns: {list(size_df.columns)}")
    
    # Process factors
    print("\n[2/2] Processing Factors...")
    factors_df = process_factors()
    if factors_df is not None:
        print(f"Shape: {factors_df.shape}")
        print(f"Date range: {factors_df.index.min()} to {factors_df.index.max()}")
        print(f"Columns: {list(factors_df.columns)}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

