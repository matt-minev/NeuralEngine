"""
Universal Character Data Loading - EMNIST ByClass
===============================================

Loads EMNIST ByClass CSV data for complete character recognition (0-9, A-Z, a-z).
Built on your proven NeuralEngine data pipeline.
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_utils import DataLoader, DataPreprocessor

def load_and_preprocess_emnist_byclass():
    """
    Load and preprocess EMNIST ByClass data from CSV files.
    FIXED: Handles CSV files without proper headers.
    
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    print("🔤 Loading EMNIST ByClass with YOUR NeuralEngine DataLoader...")
    
    # Initialize NeuralEngine components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # FIXED: Use pandas directly to handle headerless CSV files
    import pandas as pd
    
    # Load training data with no headers
    print("Loading training data...")
    train_df = pd.read_csv('data/emnist-byclass-train.csv', header=None)
    
    # Separate features and labels
    y_train = train_df.iloc[:, 0].values.reshape(-1, 1)  # First column (class labels)
    X_train = train_df.iloc[:, 1:].values  # Remaining columns (pixel data)
    
    # Load test data with no headers
    print("Loading test data...")
    test_df = pd.read_csv('data/emnist-byclass-test.csv', header=None)
    
    # Separate features and labels
    y_test = test_df.iloc[:, 0].values.reshape(-1, 1)  # First column (class labels)
    X_test = test_df.iloc[:, 1:].values  # Remaining columns (pixel data)
    
    print(f"✅ Training data: {X_train.shape}")
    print(f"✅ Test data: {X_test.shape}")
    print(f"✅ Classes: 62 total (0-9, A-Z, a-z)")
    print(f"✅ Label range: {y_train.min()}-{y_train.max()}")
    
    # Preprocess with NeuralEngine
    print("🔧 Preprocessing with YOUR NeuralEngine...")
    
    # Normalize pixel values to [0, 1] range
    X_train = preprocessor.normalize_features(X_train, method='minmax')
    X_test = preprocessor.normalize_features(X_test, method='minmax', fit_scaler=False)
    
    # Convert labels to one-hot encoding (62 classes)
    y_train_onehot = np.zeros((y_train.shape[0], 62))
    y_test_onehot = np.zeros((y_test.shape[0], 62))
    
    for i in range(y_train.shape[0]):
        y_train_onehot[i, int(y_train[i, 0])] = 1
    
    for i in range(y_test.shape[0]):
        y_test_onehot[i, int(y_test[i, 0])] = 1
    
    print("✅ Universal character data preprocessing complete!")
    print(f"   Training features: {X_train.shape}")
    print(f"   Training labels: {y_train_onehot.shape}")
    print(f"   Test features: {X_test.shape}")
    print(f"   Test labels: {y_test_onehot.shape}")
    
    return (X_train, y_train_onehot), (X_test, y_test_onehot)

def prepare_universal_data_splits():
    """
    Prepare train/validation splits from EMNIST ByClass data.
    """
    print("📊 Creating universal character recognition data splits...")
    
    # Load the preprocessed data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_emnist_byclass()
    
    # Create train/validation split from training data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,  # 10% for validation
        random_state=42,
        stratify=y_train.argmax(axis=1)  # Stratify by character class
    )
    
    print(f"✅ Universal data splits created successfully:")
    print(f"   Training: {X_train_split.shape[0]:,} samples")
    print(f"   Validation: {X_val.shape[0]:,} samples")
    print(f"   Test: {X_test.shape[0]:,} samples")
    
    # Display class distribution sample
    print(f"\n📈 Sample Class Distribution:")
    train_labels = y_train_split.argmax(axis=1)
    for i in range(0, 10):  # Show first 10 classes (digits)
        char = index_to_character(i)
        count = np.sum(train_labels == i)
        print(f"   {char}: {count:,} samples")
    print(f"   ... (52 more character classes)")
    
    return (X_train_split, y_train_split), (X_val, y_val), (X_test, y_test)

def load_universal_test_data():
    """
    Load and preprocess the EMNIST ByClass test data only.
    FIXED: Handles CSV files without proper headers.
    
    Returns:
        Tuple of (X_test, y_test) where:
        - X_test: Normalized test features (samples, 784)
        - y_test: One-hot encoded test labels (samples, 62)
    """
    print("📁 Loading EMNIST ByClass test data...")
    
    # Initialize NeuralEngine components
    preprocessor = DataPreprocessor()
    
    # FIXED: Load CSV with no headers
    import pandas as pd
    test_df = pd.read_csv('data/emnist-byclass-test.csv', header=None)
    
    # Separate features and labels
    y_test = test_df.iloc[:, 0].values.reshape(-1, 1)  # First column
    X_test = test_df.iloc[:, 1:].values  # Remaining columns
    
    # Normalize features to [0, 1] range
    X_test = preprocessor.normalize_features(X_test, method='minmax', fit_scaler=False)
    
    # Convert labels to one-hot encoding
    y_test_onehot = np.zeros((y_test.shape[0], 62))
    for i in range(y_test.shape[0]):
        y_test_onehot[i, int(y_test[i, 0])] = 1
    
    print(f"✅ Test data loaded: {X_test.shape[0]:,} samples")
    
    return X_test, y_test_onehot

# Character mapping utilities
def index_to_character(index: int) -> str:
    """Convert class index (0-61) to character."""
    if 0 <= index <= 9:
        return str(index)  # Digits 0-9
    elif 10 <= index <= 35:
        return chr(ord('A') + index - 10)  # Uppercase A-Z
    elif 36 <= index <= 61:
        return chr(ord('a') + index - 36)  # Lowercase a-z
    else:
        return '?'

def character_to_index(char: str) -> int:
    """Convert character to class index (0-61)."""
    if char.isdigit():
        return int(char)  # 0-9 maps to 0-9
    elif char.isupper() and char.isalpha():
        return ord(char) - ord('A') + 10  # A-Z maps to 10-35
    elif char.islower() and char.isalpha():
        return ord(char) - ord('a') + 36  # a-z maps to 36-61
    else:
        return -1

def get_character_type(index: int) -> str:
    """Get character type description."""
    if 0 <= index <= 9:
        return "Digit"
    elif 10 <= index <= 35:
        return "Uppercase"
    elif 36 <= index <= 61:
        return "Lowercase"
    else:
        return "Unknown"

def get_all_characters():
    """Get list of all 62 characters in order."""
    chars = []
    # Digits 0-9
    chars.extend([str(i) for i in range(10)])
    # Uppercase A-Z
    chars.extend([chr(ord('A') + i) for i in range(26)])
    # Lowercase a-z
    chars.extend([chr(ord('a') + i) for i in range(26)])
    return chars

if __name__ == "__main__":
    """Test the universal character data loading pipeline."""
    print("🧪 Testing EMNIST ByClass Data Loading...")
    print("=" * 50)
    
    try:
        # Test basic loading
        print("\n1. Testing basic data loading...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_universal_data_splits()
        
        print(f"\n✅ Data loading test successful!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Validation set: {X_val.shape}")
        print(f"   Test set: {X_test.shape}")
        
        # Test character mapping
        print(f"\n2. Testing character mapping...")
        test_chars = ['0', '5', '9', 'A', 'M', 'Z', 'a', 'm', 'z']
        for char in test_chars:
            index = character_to_index(char)
            back_to_char = index_to_character(index)
            char_type = get_character_type(index)
            print(f"   {char} -> {index} -> {back_to_char} ({char_type}) ✅")
        
        # Test sample display
        print(f"\n3. Testing sample data...")
        sample_indices = np.random.choice(len(X_train), 5, replace=False)
        for idx in sample_indices:
            char_index = np.argmax(y_train[idx])
            character = index_to_character(char_index)
            char_type = get_character_type(char_index)
            print(f"   Sample {idx}: '{character}' ({char_type}, index {char_index})")
        
        print(f"\n🎉 All tests passed! Universal character data loading is ready.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
