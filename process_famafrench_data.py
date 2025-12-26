"""
Process and extract Fama-French data from downloaded ZIP files
Automatically scans the data folder for all .zip files
"""

import pandas as pd
import zipfile
from pathlib import Path
import re

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_zip_file(zip_path, output_dir=None):
    """Extract a ZIP file"""
    if output_dir is None:
        output_dir = zip_path.parent / zip_path.stem
    output_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"  Extracted {zip_path.name} to {output_dir}")
    return output_dir


def find_data_start_row(csv_path):
    """
    Find the row index where the header row is (to skip everything before it)
    Returns the number of rows to skip so pandas reads the header row correctly
    """
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for header row (starts with comma and has multiple columns)
        if stripped.startswith(',') and stripped.count(',') > 2:
            # This is the header row, skip everything before it
            return i
        # Or look for date-like pattern (YYYYMM followed by comma) - data row
        if re.match(r'^\d{6},', stripped):
            # Found data row, look backwards for header
            for j in range(i-1, max(-1, i-10), -1):
                if j >= 0 and lines[j].strip().startswith(','):
                    return j
            # No header found, skip everything before data
            return i
    
    return 0


def parse_famafrench_csv(csv_path):
    """
    Parse a Fama-French CSV file
    Fama-French CSV files typically have:
    - Header comments/metadata at the top (need to skip)
    - Date column (YYYYMM format)
    - Multiple return columns
    - Spaces after commas
    """
    try:
        # Find where header row is
        header_row = find_data_start_row(csv_path)
        
        # Read CSV, skipping rows before header, using header row
        # Use skipinitialspace to handle spaces after commas
        df = pd.read_csv(
            csv_path, 
            skiprows=header_row,
            header=0,  # First row after skipping is the header
            skipinitialspace=True,
            na_values=['-99.99', '-999', 'NA', '', 'NaN'],
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        if df.empty:
            print(f"    Warning: Empty dataframe after parsing")
            return None
        
        # Check if first column looks like dates (YYYYMM format)
        first_col = df.columns[0]
        
        # If first column is empty or unnamed, it might be the index
        if first_col == '' or first_col.startswith('Unnamed'):
            # Check if first data value looks like a date
            if len(df) > 0:
                first_val = str(df.iloc[0, 0])
                if len(first_val) == 6 and first_val.isdigit():
                    # This is the date column, set it as index
                    df = df.set_index(df.columns[0])
                    df.index.name = 'Date'
                else:
                    # Try to find date column
                    for col in df.columns:
                        if df[col].dtype == 'object' and len(df) > 0:
                            sample = str(df[col].iloc[0])
                            if len(sample) == 6 and sample.isdigit():
                                df = df.set_index(col)
                                df.index.name = 'Date'
                                break
        else:
            # Check if first column contains dates
            sample_val = str(df[first_col].iloc[0]) if len(df) > 0 else ''
            if len(sample_val) == 6 and sample_val.isdigit():
                # Parse as date and set as index
                df[first_col] = pd.to_datetime(df[first_col].astype(str), format='%Y%m', errors='coerce')
                df = df.rename(columns={first_col: 'Date'})
                df = df.set_index('Date')
        
        # Convert numeric columns (skip date/index)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows where date is NaT (failed parsing)
        if df.index.name == 'Date':
            # Filter out NaT dates from index
            df = df[df.index.notna()]
        
        return df
    except Exception as e:
        print(f"    Error parsing CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_zip_file(zip_path):
    """
    Process a single ZIP file - extract and convert to CSV
    
    Parameters:
    -----------
    zip_path : Path
        Path to the ZIP file
    """
    print(f"\nProcessing: {zip_path.name}")
    
    # Extract ZIP
    extract_dir = extract_zip_file(zip_path)
    
    # Find CSV files (prefer CSV over TXT)
    csv_files = list(extract_dir.glob("*.csv"))
    txt_files = list(extract_dir.glob("*.txt")) if not csv_files else []
    
    data_files = csv_files if csv_files else txt_files
    
    if not data_files:
        print(f"  ⚠ No CSV or TXT files found in {extract_dir}")
        return None
    
    # Process all data files found
    results = {}
    for data_file in data_files:
        print(f"  Processing {data_file.name}...")
        
        if data_file.suffix.lower() == '.csv':
            df = parse_famafrench_csv(data_file)
        else:
            # For TXT files, use the old method (though we expect CSV)
            print(f"    Note: Found TXT file, but expecting CSV")
            continue
        
        if df is not None and not df.empty:
            # Create output filename
            zip_stem = zip_path.stem.replace('_CSV', '').replace('_TXT', '')
            output_filename = f"{zip_stem}_{data_file.stem}.csv"
            output_path = OUTPUT_DIR / output_filename
            
            df.to_csv(output_path)
            print(f"    ✓ Saved to {output_path}")
            print(f"      Shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
            
            results[data_file.name] = df
        else:
            print(f"    ⚠ Could not process {data_file.name}")
    
    return results


def scan_and_process_all():
    """
    Scan data directory for all .zip files and process them
    """
    if not DATA_DIR.exists():
        print(f"Error: Data directory '{DATA_DIR}' does not exist.")
        return
    
    # Find all ZIP files
    zip_files = sorted(DATA_DIR.glob("*.zip"))
    
    if not zip_files:
        print(f"No .zip files found in {DATA_DIR}")
        return
    
    print("=" * 70)
    print("Processing Fama-French Data")
    print("=" * 70)
    print(f"\nFound {len(zip_files)} ZIP file(s) in {DATA_DIR}:")
    for i, zip_file in enumerate(zip_files, 1):
        print(f"  {i}. {zip_file.name}")
    print()
    
    all_results = {}
    
    for zip_file in zip_files:
        results = process_zip_file(zip_file)
        if results:
            all_results[zip_file.name] = results
    
    # Summary
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"\nProcessed {len(all_results)} ZIP file(s)")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nProcessed files:")
    for zip_name, file_results in all_results.items():
        print(f"  {zip_name}:")
        for file_name, df in file_results.items():
            print(f"    - {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return all_results


def filter_date_range(df, start_year=1927, end_year=2024):
    """Filter dataframe to specific date range"""
    if df.index.name != 'Date' and 'Date' not in df.columns:
        print("Warning: No date column found for filtering")
        return df
    
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask]


def main():
    """Main processing function"""
    results = scan_and_process_all()
    
    if results:
        print("\n" + "=" * 70)
        print("✓ Processing complete!")
        print("=" * 70)
        print(f"\nAll processed files are in: {OUTPUT_DIR}")
        print("\nYou can now load the processed CSV files for analysis.")
    else:
        print("\n⚠ No files were processed.")


if __name__ == "__main__":
    main()
