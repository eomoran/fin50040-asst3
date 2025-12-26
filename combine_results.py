#!/usr/bin/env python3
"""
Combine all analysis results into single CSV files

This script combines all individual result CSVs (across different portfolio sorts,
time periods, and flags) into consolidated CSV files for easier comparison.
"""

import pandas as pd
from pathlib import Path
import re

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/combined")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_filename(filename):
    """
    Parse filename to extract configuration parameters
    
    Example: jensens_alpha_rf_size_1927_2013_no_small_caps_recentred.csv
    Returns: {
        'metric': 'jensens_alpha_rf',
        'portfolio_type': 'size',
        'start_year': '1927',
        'end_year': '2013',
        'exclude_small_caps': True,
        'recentred': True
    }
    """
    name = filename.stem  # Remove .csv extension
    
    # Known metric prefixes
    metric_prefixes = [
        'jensens_alpha_rf',
        'jensens_alpha_zerobeta',
        'msmp_weights',
        'optimal_weights',
        'zero_beta_weights'
    ]
    
    # Find the metric by matching known prefixes
    metric = None
    portfolio_idx = None
    
    for prefix in metric_prefixes:
        if name.startswith(prefix):
            metric = prefix
            # Find where portfolio type starts (after metric)
            remaining = name[len(prefix):].lstrip('_')
            parts_remaining = remaining.split('_')
            
            # Find portfolio type in remaining parts
            for i, part in enumerate(parts_remaining):
                if part in ['size', 'value']:
                    portfolio_idx = len(prefix.split('_')) + i
                    break
            break
    
    # If no metric found, try to extract from pattern
    if metric is None:
        parts = name.split('_')
        # Find where portfolio type starts (size or value)
        for i, part in enumerate(parts):
            if part in ['size', 'value']:
                portfolio_idx = i
                metric = '_'.join(parts[:i])
                break
        
        if portfolio_idx is None:
            # Old format without portfolio type (e.g., jensens_alpha_rf.csv)
            # Try to match known metrics
            for prefix in metric_prefixes:
                if prefix in name:
                    metric = prefix
                    break
            
            if metric is None:
                metric = name  # Use full name as metric
            
            return {
                'metric': metric,
                'portfolio_type': 'unknown',
                'start_year': None,
                'end_year': None,
                'exclude_small_caps': False,
                'recentred': False
            }
    
    parts = name.split('_')
    if portfolio_idx is not None and portfolio_idx < len(parts):
        portfolio_type = parts[portfolio_idx]
        
        # Extract years (should be after portfolio type)
        start_year = None
        end_year = None
        if portfolio_idx + 1 < len(parts):
            # Check if next part is a year
            year_match = re.match(r'^(\d{4})$', parts[portfolio_idx + 1])
            if year_match:
                start_year = parts[portfolio_idx + 1]
                if portfolio_idx + 2 < len(parts):
                    year_match2 = re.match(r'^(\d{4})$', parts[portfolio_idx + 2])
                    if year_match2:
                        end_year = parts[portfolio_idx + 2]
    else:
        portfolio_type = 'unknown'
        start_year = None
        end_year = None
    
    # Check for flags
    exclude_small_caps = 'no_small_caps' in name
    recentred = 'recentred' in name
    
    return {
        'metric': metric,
        'portfolio_type': portfolio_type,
        'start_year': start_year,
        'end_year': end_year,
        'exclude_small_caps': exclude_small_caps,
        'recentred': recentred
    }


def combine_results():
    """Combine all result files by metric type"""
    
    # Get all CSV files
    csv_files = list(RESULTS_DIR.glob("*.csv"))
    
    # Group by metric type
    metric_groups = {}
    
    for csv_file in csv_files:
        config = parse_filename(csv_file)
        metric = config['metric']
        
        if metric not in metric_groups:
            metric_groups[metric] = []
        
        metric_groups[metric].append((csv_file, config))
    
    # Combine each metric group
    for metric, files in metric_groups.items():
        print(f"\nCombining {metric}...")
        print(f"  Found {len(files)} files")
        
        combined_data = []
        
        for csv_file, config in files:
            try:
                # Read the CSV
                df = pd.read_csv(csv_file, index_col=0)
                
                # Add configuration columns
                df['portfolio_type'] = config['portfolio_type']
                df['start_year'] = config['start_year']
                df['end_year'] = config['end_year']
                df['exclude_small_caps'] = config['exclude_small_caps']
                df['recentred'] = config['recentred']
                
                # Add configuration identifier
                config_parts = [config['portfolio_type']]
                if config['start_year']:
                    config_parts.append(f"{config['start_year']}-{config['end_year']}")
                if config['exclude_small_caps']:
                    config_parts.append('no_small_caps')
                if config['recentred']:
                    config_parts.append('recentred')
                
                df['configuration'] = '_'.join(config_parts)
                
                # The first column should be the value (metric value)
                # Rename it to the metric name if it's not already
                value_col = df.columns[0]
                if value_col == '0' or value_col == 'Unnamed: 0':
                    # The actual values are in the first data column
                    if len(df.columns) > 1:
                        value_col = df.columns[1]
                
                # Rename value column to metric name
                df = df.rename(columns={value_col: metric})
                
                # Reset index to make portfolio name a column
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'portfolio'})
                
                combined_data.append(df)
                
            except Exception as e:
                print(f"  Warning: Error reading {csv_file.name}: {e}")
                continue
        
        if not combined_data:
            print(f"  No valid data to combine for {metric}")
            continue
        
        # Combine all DataFrames
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Reorder columns: portfolio, configuration details, then value
        cols = ['portfolio', 'portfolio_type', 'start_year', 'end_year', 
                'exclude_small_caps', 'recentred', 'configuration', metric]
        # Only include columns that exist
        cols = [c for c in cols if c in combined_df.columns]
        combined_df = combined_df[cols]
        
        # Sort by portfolio, then by configuration
        combined_df = combined_df.sort_values(['portfolio', 'configuration'])
        
        # Save combined file
        output_file = OUTPUT_DIR / f"{metric}_combined.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file.name} ({len(combined_df)} rows)")
    
    print(f"\n✓ Combined results saved to {OUTPUT_DIR}/")
    print(f"  Generated {len(metric_groups)} combined files")


if __name__ == "__main__":
    print("=" * 70)
    print("Combining Analysis Results")
    print("=" * 70)
    combine_results()
    print("\n" + "=" * 70)

