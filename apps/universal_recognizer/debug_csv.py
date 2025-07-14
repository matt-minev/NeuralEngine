"""
Debug script to check EMNIST ByClass CSV structure
"""
import pandas as pd
import os

def check_csv_structure():
    """Check the structure of EMNIST ByClass CSV files."""
    print("🔍 Debugging EMNIST ByClass CSV Structure")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ 'data' directory not found!")
        print("Please ensure your CSV files are in a 'data' folder.")
        return
    
    # List files in data directory
    files = os.listdir('data')
    print(f"📁 Files in data directory: {files}")
    
    # Find EMNIST files
    train_files = [f for f in files if 'train' in f.lower()]
    test_files = [f for f in files if 'test' in f.lower()]
    
    print(f"🚂 Training files found: {train_files}")
    print(f"🧪 Test files found: {test_files}")
    
    # Check each file
    for filename in files:
        if filename.endswith('.csv'):
            filepath = os.path.join('data', filename)
            print(f"\n📊 Analyzing {filename}:")
            
            try:
                # Read just the first few rows
                df = pd.read_csv(filepath, nrows=5)
                
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   First column name: '{df.columns[0]}'")
                print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
                
                # Show first few values of first column
                print(f"   First column values: {df.iloc[:, 0].tolist()}")
                
                # Check if first column looks like labels (0-61 range)
                first_col_values = df.iloc[:, 0]
                print(f"   First column range: {first_col_values.min()} to {first_col_values.max()}")
                
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")

if __name__ == "__main__":
    check_csv_structure()
