"""
Download Fama-French data for Assignment 3
Downloads portfolio sorts, benchmark factors, and risk-free rate data
Automatically discovers available files from the FTP directory
"""

import requests
import re
from pathlib import Path
from html.parser import HTMLParser

# Base URL for Kenneth French data
BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


class LinkParser(HTMLParser):
    """Parse HTML to extract links to .zip files"""
    def __init__(self):
        super().__init__()
        self.file_links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[0] == 'href':
                    href = attr[1]
                    # Extract .zip file links
                    if href.endswith('.zip'):
                        self.file_links.append(href)


def get_available_files():
    """
    Fetch file listings from the data library page
    The FTP directory listing is forbidden (403), so we scrape the data library page instead
    """
    data_library_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
    
    print("Fetching file listings from Fama-French data library page...")
    
    try:
        response = requests.get(data_library_url, timeout=30)
        response.raise_for_status()
        
        parser = LinkParser()
        parser.feed(response.text)
        
        # Extract .zip files from the links
        zip_files = []
        file_map = {}
        
        for link in parser.file_links:
            # Make absolute URL if needed
            if link.startswith('http'):
                full_url = link
            elif link.startswith('/'):
                full_url = "https://mba.tuck.dartmouth.edu" + link
            elif link.startswith('ftp/'):
                full_url = BASE_URL + link
            else:
                full_url = BASE_URL + "ftp/" + link
            
            filename = full_url.split('/')[-1]
            zip_files.append(filename)
            file_map[filename] = full_url
        
        zip_files = sorted(set(zip_files))
        print(f"Found {len(zip_files)} .zip files on data library page")
        return zip_files, file_map
        
    except Exception as e:
        print(f"Error fetching data library page: {e}")
        return [], {}


def find_file_by_keywords(available_files, keywords, prefer_csv=True):
    """
    Find a file matching keywords
    
    Parameters:
    -----------
    available_files : list
        List of available .zip file names
    keywords : list
        List of keywords to match (e.g., ['ME', 'Portfolios'])
    prefer_csv : bool
        If True, prefer files with '_CSV' in the name
    """
    matches = []
    
    for filename in available_files:
        filename_upper = filename.upper()
        # Check if all keywords are in the filename
        if all(keyword.upper() in filename_upper for keyword in keywords):
            matches.append(filename)
    
    if not matches:
        return None
    
    # Prefer CSV versions if requested
    if prefer_csv:
        csv_matches = [f for f in matches if '_CSV' in f.upper()]
        if csv_matches:
            return csv_matches[0]
    
    # Return first match
    return matches[0]


def download_famafrench_file(url, file_name, description):
    """
    Download a file from the Fama-French website
    
    Parameters:
    -----------
    url : str
        Full URL to the file
    file_name : str
        Name of the file (for saving)
    description : str
        Description for logging
    """
    output_path = DATA_DIR / file_name
    
    print(f"Downloading {description}...")
    print(f"  File: {file_name}")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        
        # Check status code explicitly
        if response.status_code == 404:
            print(f"  ✗ ERROR: File not found (404)")
            return None
        
        # Raise exception for other HTTP errors
        response.raise_for_status()
        
        # Check if response has content
        if not response.content:
            print(f"  ✗ ERROR: Empty response from server")
            return None
        
        # Write file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Verify file was written and has content
        if not output_path.exists():
            print(f"  ✗ ERROR: File was not created")
            return None
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            print(f"  ✗ ERROR: Downloaded file is empty (0 bytes)")
            output_path.unlink()
            return None
        
        print(f"  ✓ Successfully downloaded ({file_size:,} bytes)")
        return output_path
        
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP Error: {e}")
        if 'response' in locals():
            print(f"     Status code: {response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Network Error: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return None


def download_size_portfolios(available_files, file_map):
    """Download 10 size portfolios (Portfolios Formed on ME)"""
    file_name = find_file_by_keywords(available_files, ['Portfolios', 'ME'], prefer_csv=True)
    if not file_name:
        print("  ✗ Could not find size portfolios file")
        return None
    url = file_map.get(file_name, BASE_URL + file_name)
    return download_famafrench_file(url, file_name, "10 Size Portfolios")


def download_benchmark_factors(available_files, file_map):
    """Download Fama-French factors"""
    # 3-Factor Model
    file_3f = find_file_by_keywords(available_files, ['F-F', 'Research', 'Factors'], prefer_csv=False)
    if not file_3f:
        # Try alternative naming
        file_3f = find_file_by_keywords(available_files, ['Factors'], prefer_csv=False)
    
    result_3f = None
    if file_3f:
        url_3f = file_map.get(file_3f, BASE_URL + file_3f)
        result_3f = download_famafrench_file(url_3f, file_3f, "Fama-French 3-Factor Model")
    
    # 5-Factor Model
    file_5f = find_file_by_keywords(available_files, ['5_Factors'], prefer_csv=False)
    if not file_5f:
        file_5f = find_file_by_keywords(available_files, ['5', 'Factors'], prefer_csv=False)
    
    result_5f = None
    if file_5f:
        url_5f = file_map.get(file_5f, BASE_URL + file_5f)
        result_5f = download_famafrench_file(url_5f, file_5f, "Fama-French 5-Factor Model")
    
    return result_3f, result_5f


def download_second_portfolio_sort(available_files, file_map, sort_type="value"):
    """
    Download a second portfolio sort
    
    Parameters:
    -----------
    available_files : list
        List of available files
    file_map : dict
        Dictionary mapping filenames to full URLs
    sort_type : str
        Type of portfolio sort
    """
    sort_keywords = {
        "value": ['BE-ME', 'Portfolios'],
        "momentum": ['Prior', '12', '2'],
        "profitability": ['OP', 'Portfolios'],
        "investment": ['INV', 'Portfolios'],
        "beta": ['BETA', 'Portfolios']
    }
    
    if sort_type not in sort_keywords:
        print(f"  ✗ Unknown sort type: {sort_type}")
        print(f"     Available options: {list(sort_keywords.keys())}")
        return None
    
    keywords = sort_keywords[sort_type]
    file_name = find_file_by_keywords(available_files, keywords, prefer_csv=True)
    
    if not file_name:
        print(f"  ✗ Could not find {sort_type} portfolio file")
        print(f"     Searched for files containing: {keywords}")
        return None
    
    url = file_map.get(file_name, BASE_URL + file_name)
    description = f"Portfolio sort: {sort_type.title()}"
    return download_famafrench_file(url, file_name, description)


def main():
    """Main function to download all required data"""
    print("=" * 70)
    print("Fama-French Data Downloader for Assignment 3")
    print("=" * 70)
    print()
    
    # Get list of available files and file map
    available_files, file_map = get_available_files()
    
    if not available_files:
        print("\n✗ Could not fetch directory listing. Exiting.")
        return
    
    print(f"\nAvailable files (showing first 20):")
    for i, f in enumerate(available_files[:20], 1):
        print(f"  {i}. {f}")
    if len(available_files) > 20:
        print(f"  ... and {len(available_files) - 20} more")
    print()
    
    results = []
    
    # Download 10 size portfolios
    print("\n" + "=" * 70)
    print("[1/4] Downloading 10 Size Portfolios...")
    print("=" * 70)
    result = download_size_portfolios(available_files, file_map)
    results.append(("10 Size Portfolios", result is not None))
    
    # Download second portfolio sort
    print("\n" + "=" * 70)
    print("[2/4] Downloading Second Portfolio Sort...")
    print("=" * 70)
    print("Current selection: value (Book-to-Market)")
    print("To change, modify the sort_type parameter in download_second_portfolio_sort()")
    result = download_second_portfolio_sort(available_files, file_map, sort_type="value")
    results.append(("Second Portfolio Sort (value)", result is not None))
    
    # Download benchmark factors
    print("\n" + "=" * 70)
    print("[3/4] Downloading Benchmark Factors...")
    print("=" * 70)
    result_3f, result_5f = download_benchmark_factors(available_files, file_map)
    results.append(("3-Factor Model", result_3f is not None))
    results.append(("5-Factor Model", result_5f is not None))
    
    # Risk-free rate info
    print("\n" + "=" * 70)
    print("[4/4] Risk-Free Rate Data...")
    print("=" * 70)
    print("Note: Risk-free rate (RF) is included in the Fama-French factors file")
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary:")
    print("=" * 70)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {successful}/{total} downloads successful")
    
    if successful == total:
        print("\n✓ All downloads completed successfully!")
        print(f"Data saved to: {DATA_DIR.absolute()}")
        print("\nNext steps:")
        print("1. Extract the ZIP files")
        print("2. Load and process the data for your analysis")
        print("3. Check the data files for the exact format and date ranges")
    else:
        print(f"\n⚠ Warning: {total - successful} download(s) failed.")
        print("\nIf files were not found, you can:")
        print("1. Check the available files listed above")
        print("2. Manually download from:")
        print("   https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
        print("3. Modify the keywords in the script to match different file names")


if __name__ == "__main__":
    main()
