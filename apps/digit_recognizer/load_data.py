"""
MNIST data loading - fixed version.

Loads MNIST CSV data and prepares it for NeuralEngine training.
Fixed the DataSplitter issue that was causing unpacking errors.
"""

import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_utils import DataLoader, DataPreprocessor


def load_and_preprocess_mnist():
    """
    Load and preprocess MNIST data from CSV files.
    
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    print("Loading MNIST CSV with NeuralEngine DataLoader...")
    
    # initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # load training data
    print("Loading training data...")
    X_train, y_train = loader.load_csv(
        'data/mnist_train.csv',
        target_column='label'
    )
    
    # load test data
    print("Loading test data...")
    X_test, y_test = loader.load_csv(
        'data/mnist_test.csv', 
        target_column='label'
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # preprocess with neuralengine
    print("Preprocessing with NeuralEngine...")
    
    # normalize pixel values to [0, 1] range
    X_train = preprocessor.normalize_features(X_train, method='minmax')
    X_test = preprocessor.normalize_features(X_test, method='minmax', fit_scaler=False)
    
    # convert labels to one-hot encodeing
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    
    for i in range(y_train.shape[0]):
        y_train_onehot[i, int(y_train[i, 0])] = 1
        
    for i in range(y_test.shape[0]):
        y_test_onehot[i, int(y_test[i, 0])] = 1
    
    print("Data preprocessing complete!")
    print(f"  Training features: {X_train.shape}")
    print(f"  Training labels: {y_train_onehot.shape}")
    print(f"  Test features: {X_test.shape}")
    print(f"  Test labels: {y_test_onehot.shape}")
    
    return (X_train, y_train_onehot), (X_test, y_test_onehot)


def prepare_data_splits():
    """
    Prepare train/validation splits from loaded MNIST data.
    FIXED VERSION - No more unpacking errors!
    """
    print("Creating data splits with train_test_split...")
    
    # load the preprocessed data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
    
    # FIXED: use sklearn's train_test_split directly to avoid DataSplitter issues
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,  # 10% for validation
        random_state=42,
        stratify=y_train.argmax(axis=1)  # stratify by digit class
    )
    
    print(f"Data splits created succesfully:")
    print(f"  Training: {X_train_split.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    return (X_train_split, y_train_split), (X_val, y_val), (X_test, y_test)


def load_test_data():
    """
    Load and preprocess the MNIST test data only.
    
    Loads just the test data for evaluation purposes,
    seperate from the training pipeline.
    
    Returns:
        Tuple of (X_test, y_test) where:
        - X_test: normalized test features (samples, 784)
        - y_test: one-hot encoded test labels (samples, 10)
    """
    print("Loading MNIST test data...")
    
    # initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # load test data from CSV
    X_test, y_test = loader.load_csv(
        'data/mnist_test.csv',
        target_column='label'
    )
    
    # normalize features to [0, 1] range
    X_test = preprocessor.normalize_features(X_test, method='minmax', fit_scaler=False)
    
    # convert labels to one-hot encoding
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    for i in range(y_test.shape[0]):
        y_test_onehot[i, int(y_test[i, 0])] = 1
    
    print(f"Test data loaded: {X_test.shape[0]:,} samples")
    
    return X_test, y_test_onehot


if __name__ == "__main__":
    """Test the data loading pipeline."""
    print("Testing MNIST data loading...")
    
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits()
        
        print(f"\nData loading test successful!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Sample training labels: {y_train[:3].argmax(axis=1)}")
        
    except Exception as e:
        print(f"Error during data loading: {e}")
        raise
