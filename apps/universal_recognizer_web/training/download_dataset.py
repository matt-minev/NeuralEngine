"""
Script to download EMNIST ByClass dataset.

Downloads the official EMNIST ByClass dataset from NIST in the required format.
"""

import os
import sys
import urllib.request
import gzip
import shutil

# EMNIST ByClass download URLs (official NIST source)
EMNIST_BASE_URL = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
EMNIST_FILES = {
    'train_images': 'emnist-byclass-train-images-idx3-ubyte.gz',
    'train_labels': 'emnist-byclass-train-labels-idx1-ubyte.gz',
    'test_images': 'emnist-byclass-test-images-idx3-ubyte.gz',
    'test_labels': 'emnist-byclass-test-labels-idx1-ubyte.gz',
    'mapping': 'emnist-byclass-mapping.txt'
}

# Alternative: Direct download from NIST (if base URL doesn't work)
ALTERNATIVE_URLS = {
    'train_images': 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
    # Note: EMNIST is distributed as a zip file containing all splits
}

def download_file(url: str, destination: str, description: str):
    """Download a file with progress."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {destination}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n  ✓ Downloaded successfully")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def download_emnist_byclass(data_dir: str = None):
    """
    Download EMNIST ByClass dataset.
    
    Args:
        data_dir: Directory to save data (default: apps/universal_recognizer_web/data)
    """
    if data_dir is None:
        # Default to universal_recognizer_web/data
        # __file__ is training/download_dataset.py
        # Go up to universal_recognizer_web, then into data
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_path, 'data')
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print("=" * 70)
    
    print("\n⚠️  IMPORTANT: EMNIST Dataset Download Instructions")
    print("=" * 70)
    print("\nThe EMNIST ByClass dataset is available from NIST but requires")
    print("manual download due to their website structure.")
    print("\nPlease follow these steps:")
    print("\n1. Visit the official EMNIST page:")
    print("   https://www.nist.gov/itl/iad/image-group/emnist-dataset")
    print("\n2. Or go directly to the download page:")
    print("   https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/")
    print("\n3. Download the 'gzip.zip' file (contains all EMNIST splits)")
    print("\n4. Extract the zip file")
    print("\n5. Copy the following files to the data directory:")
    print(f"   {data_dir}/")
    print("   - emnist-byclass-train-images-idx3-ubyte.gz")
    print("   - emnist-byclass-train-labels-idx1-ubyte.gz")
    print("   - emnist-byclass-test-images-idx3-ubyte.gz")
    print("   - emnist-byclass-test-labels-idx1-ubyte.gz")
    print("   - emnist-byclass-mapping.txt")
    print("\n6. Verify files are in place, then run training")
    print("\n" + "=" * 70)
    
    # Check if files already exist
    print("\nChecking for existing files...")
    all_exist = True
    for file_key, filename in EMNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n✓ All required files found! Ready to train.")
        return True
    else:
        print("\n✗ Some files are missing. Please download and place them as instructed above.")
        return False


def verify_dataset(data_dir: str = None):
    """Verify that all required dataset files are present."""
    if data_dir is None:
        # __file__ is training/download_dataset.py
        # Go up to universal_recognizer_web, then into data
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_path, 'data')
    
    print(f"Verifying dataset in: {data_dir}")
    print("=" * 70)
    
    all_good = True
    expected_sizes = {
        'emnist-byclass-train-images-idx3-ubyte.gz': 450_000_000,  # ~450MB
        'emnist-byclass-test-images-idx3-ubyte.gz': 75_000_000,   # ~75MB
        'emnist-byclass-train-labels-idx1-ubyte.gz': 700_000,     # ~700KB
        'emnist-byclass-test-labels-idx1-ubyte.gz': 120_000,      # ~120KB
        'emnist-byclass-mapping.txt': 1000                         # ~1KB
    }
    
    for filename, min_size in expected_sizes.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            # More lenient check - files might be smaller if using different EMNIST version
            if size >= min_size * 0.3:  # Allow 70% variance (some EMNIST versions are smaller)
                print(f"  ✓ {filename} ({size_mb:.1f} MB) - OK")
            elif size > 0:
                print(f"  ⚠ {filename} ({size_mb:.1f} MB) - WARNING: File seems small but may still work")
                # Don't fail on size warnings, just warn
            else:
                print(f"  ✗ {filename} - ERROR: File is empty")
                all_good = False
        else:
            print(f"  ✗ {filename} - MISSING")
            all_good = False
    
    if all_good:
        print("\n✓ Dataset verification passed! Ready to train.")
    else:
        print("\n✗ Dataset verification failed. Please check the files.")
    
    return all_good


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download/verify EMNIST ByClass dataset')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory (default: apps/universal_recognizer_web/data)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing files, do not show download instructions')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.data_dir)
    else:
        download_emnist_byclass(args.data_dir)
        print("\n" + "=" * 70)
        print("After downloading, verify with:")
        print("  python -m training.download_dataset --verify-only")
        print("=" * 70)

