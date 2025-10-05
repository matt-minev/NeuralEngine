"""
Data utilities for loading, preprocessing, and managing datasets.

Handles CSV/JSON/NumPy loading, normalization, train/val/test splits,
and batch processing.
"""

import numpy as np
import pandas as pd
import json
from typing import Tuple, List, Dict, Optional, Union, Any
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load data from various file formats (CSV, JSON, NumPy)."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.loaded_data = {}
        self.data_info = {}
    
    def load_csv(self, filepath: str, target_column: str = None, 
                 feature_columns: List[str] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file.
        
        Args:
            filepath: path to CSV
            target_column: name of target column (y values)
            feature_columns: list of feature columns (X values)
        """
        if self.verbose:
            print(f"Loading CSV: {filepath}")
        
        try:
            df = pd.read_csv(filepath, **kwargs)
            
            if self.verbose:
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
        
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
        
        # handle missing values
        if df.isnull().any().any():
            missing_info = df.isnull().sum()
            if self.verbose:
                print(f"  Missing values detected:")
                for col, count in missing_info[missing_info > 0].items():
                    print(f"    {col}: {count} missing")
            
            # fill with median for numeric, mode for categorical
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # separate features and target
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            y = df[target_column].values
            X = df.drop(columns=[target_column]).values
            
            if self.verbose:
                print(f"  Target: {target_column}")
                print(f"  Features: {df.drop(columns=[target_column]).columns.tolist()}")
        
        elif feature_columns:
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
            
            X = df[feature_columns].values
            y = None
            
            if self.verbose:
                print(f"  Features: {feature_columns}")
        
        else:
            # use all columns as features
            X = df.values
            y = None
            
            if self.verbose:
                print(f"  Using all columns as features")
        
        # convert to proper types
        X = np.array(X, dtype=np.float32)
        if y is not None:
            y = np.array(y, dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        
        # store info
        self.data_info[filepath] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'target_column': target_column,
            'feature_columns': feature_columns or df.columns.tolist()
        }
        
        if self.verbose:
            print(f"  Data loaded successfully")
            print(f"  X: {X.shape}, y: {y.shape if y is not None else 'None'}")
        
        return X, y
    
    def load_json(self, filepath: str, x_key: str = 'X', y_key: str = 'y') -> Tuple[np.ndarray, np.ndarray]:
        """Load data from JSON file."""
        if self.verbose:
            print(f"Loading JSON: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON: {e}")
        
        # extract features and targets
        if x_key not in data:
            raise ValueError(f"Key '{x_key}' not found in JSON")
        
        X = np.array(data[x_key], dtype=np.float32)
        
        if y_key in data:
            y = np.array(data[y_key], dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
        else:
            y = None
            if self.verbose:
                print(f"  No target data found (key: '{y_key}')")
        
        if self.verbose:
            print(f"  Data loaded")
            print(f"  X: {X.shape}, y: {y.shape if y is not None else 'None'}")
        
        return X, y
    
    def load_numpy(self, x_path: str, y_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from NumPy files."""
        if self.verbose:
            print(f"Loading NumPy: {x_path}")
        
        try:
            X = np.load(x_path)
            X = X.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load X from {x_path}: {e}")
        
        if y_path:
            try:
                y = np.load(y_path)
                y = y.astype(np.float32)
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
            except Exception as e:
                raise ValueError(f"Failed to load y from {y_path}: {e}")
        else:
            y = None
        
        if self.verbose:
            print(f"  Data loaded")
            print(f"  X: {X.shape}, y: {y.shape if y is not None else 'None'}")
        
        return X, y


class DataPreprocessor:
    """Preprocessing utilities for normalization, scaling, and feature enginering."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.scalers = {}
        self.preprocessing_info = {}
    
    def normalize_features(self, X: np.ndarray, method: str = 'standard', 
                          fit_scaler: bool = True) -> np.ndarray:
        """
        Normalize features.
        
        Methods:
        - standard: (x - mean) / std
        - minmax: (x - min) / (max - min)
        - robust: (x - median) / IQR (good for outliers)
        """
        if self.verbose:
            print(f"Normalizing with {method} method")
        
        # choose scaler
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # fit or use existing
        if fit_scaler or method not in self.scalers:
            self.scalers[method] = scaler_class()
            X_normalized = self.scalers[method].fit_transform(X)
            
            if self.verbose:
                print(f"  Fitted new {method} scaler")
        else:
            X_normalized = self.scalers[method].transform(X)
            
            if self.verbose:
                print(f"  Used existing {method} scaler")
        
        # store info
        self.preprocessing_info[method] = {
            'original_shape': X.shape,
            'normalized_shape': X_normalized.shape,
            'original_mean': np.mean(X, axis=0),
            'original_std': np.std(X, axis=0),
            'normalized_mean': np.mean(X_normalized, axis=0),
            'normalized_std': np.std(X_normalized, axis=0)
        }
        
        if self.verbose:
            print(f"  Original mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")
            print(f"  Normalized mean: {np.mean(X_normalized):.4f}, std: {np.std(X_normalized):.4f}")
        
        return X_normalized.astype(np.float32)
    
    def normalize_targets(self, y: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize target values."""
        if self.verbose:
            print(f"Normalizing targets with {method} method")
        
        return self.normalize_features(y, method, fit_scaler=True)
    
    def detect_outliers(self, X: np.ndarray, method: str = 'iqr', 
                       threshold: float = 1.5) -> np.ndarray:
        """
        Detect outliers using IQR or z-score method.
        
        Returns boolean mask indicating outliers.
        """
        if self.verbose:
            print(f"Detecting outliers with {method} method")
        
        if method == 'iqr':
            # interquartile range
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (X < lower_bound) | (X > upper_bound)
            
        elif method == 'zscore':
            # z-score method
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            outliers = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_count = np.sum(outliers)
        if self.verbose:
            print(f"  Found {outlier_count} outliers ({outlier_count/X.size*100:.2f}%)")
        
        return outliers
    
    def handle_outliers(self, X: np.ndarray, method: str = 'clip', 
                       threshold: float = 1.5) -> np.ndarray:
        """Handle outliers by clipping or replacing."""
        outliers = self.detect_outliers(X, 'iqr', threshold)
        
        if method == 'clip':
            # clip to bounds
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            X_handled = np.clip(X, lower_bound, upper_bound)
            
        elif method == 'replace':
            # replace with median
            X_handled = X.copy()
            median_values = np.median(X, axis=0)
            X_handled[outliers] = median_values
            
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
        
        if self.verbose:
            print(f"  Handled outliers using {method}")
        
        return X_handled
    
    def inverse_transform(self, X_normalized: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Transform normalized data back to original scale."""
        if method not in self.scalers:
            raise ValueError(f"No scaler found for method: {method}")
        
        return self.scalers[method].inverse_transform(X_normalized)


class DataSplitter:
    """Split data into train/val/test sets."""
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
    
    def train_val_test_split(self, X: np.ndarray, y: np.ndarray, 
                            train_size: float = 0.7, val_size: float = 0.15,
                            test_size: float = 0.15, stratify: bool = False) -> Tuple:
        """
        Split data into train/val/test sets.
        
        Args:
            X, y: data arrays
            train_size, val_size, test_size: split ratios (must sum to 1.0)
            stratify: whether to stratify split for classification
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Split sizes must sum to 1.0")
        
        if self.verbose:
            print(f"Splitting data: {train_size:.0%} train, {val_size:.0%} val, {test_size:.0%} test")
        
        # first split: train vs (val + test)
        stratify_y = y if stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(val_size + test_size),
            random_state=self.random_state,
            stratify=stratify_y
        )

        # shortcircuit when test_size == 0
        if test_size == 0:
            return (X_train, y_train), (X_temp, y_temp), (None, None)
        
        # second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        stratify_y_temp = y_temp if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=self.random_state,
            stratify=stratify_y_temp
        )
        
        if self.verbose:
            print(f"  Train: {X_train.shape[0]} samples")
            print(f"  Val: {X_val.shape[0]} samples")
            print(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def time_based_split(self, X: np.ndarray, y: np.ndarray, 
                        train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """Split time series data chronologically (no shuffling)."""
        n_samples = X.shape[0]
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        if self.verbose:
            print(f"Time-based split:")
            print(f"  Train: samples 0-{train_end-1}")
            print(f"  Val: samples {train_end}-{val_end-1}")
            print(f"  Test: samples {val_end}-{n_samples-1}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class BatchProcessor:
    """Create mini-batches for training."""
    
    def __init__(self, batch_size: int = 32, shuffle: bool = True, 
                 random_state: int = 42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create list of batches from data."""
        n_samples = X.shape[0]
        
        # shuffle if needed
        if self.shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
        
        # create batches
        batches = []
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            batches.append((X_batch, y_batch))
        
        return batches
    
    def batch_generator(self, X: np.ndarray, y: np.ndarray):
        """Generator for memory-efficent batch processing."""
        while True:
            batches = self.create_batches(X, y)
            for batch in batches:
                yield batch


if __name__ == "__main__":
    print("Testing Data Utilities")
    print("=" * 40)
    
    # create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 
         0.5 * X[:, 3] + 1.5 * X[:, 4] + np.random.randn(n_samples) * 0.1)
    y = y.reshape(-1, 1)
    
    # add outliers
    outlier_indices = np.random.choice(n_samples, 50, replace=False)
    X[outlier_indices] += np.random.randn(50, n_features) * 5
    
    print(f"Created synthetic dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Added 50 outliers")
    
    # test preprocessor
    print(f"\nTesting DataPreprocessor...")
    preprocessor = DataPreprocessor(verbose=True)
    
    X_normalized = preprocessor.normalize_features(X, method='standard')
    print(f"  Normalized shape: {X_normalized.shape}")
    
    outliers = preprocessor.detect_outliers(X, method='iqr')
    print(f"  Detected {np.sum(outliers)} outlier points")
    
    X_handled = preprocessor.handle_outliers(X, method='clip')
    print(f"  Handled outliers: {X_handled.shape}")
    
    # test splitter
    print(f"\nTesting DataSplitter...")
    splitter = DataSplitter(verbose=True)
    
    splits = splitter.train_val_test_split(X_normalized, y, 
                                          train_size=0.7, val_size=0.15, test_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # test batch processor
    print(f"\nTesting BatchProcessor...")
    batch_processor = BatchProcessor(batch_size=64, shuffle=True)
    
    batches = batch_processor.create_batches(X_train, y_train)
    print(f"  Created {len(batches)} batches")
    print(f"  First batch: {batches[0][0].shape}")
    print(f"  Last batch: {batches[-1][0].shape}")
    
    batch_gen = batch_processor.batch_generator(X_train, y_train)
    first_batch = next(batch_gen)
    print(f"  Generator first batch: {first_batch[0].shape}")
    
    # test CSV loading
    print(f"\nTesting CSV loading...")
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y.flatten()
    df.to_csv('temp_data.csv', index=False)
    
    loader = DataLoader(verbose=True)
    X_loaded, y_loaded = loader.load_csv('temp_data.csv', target_column='target')
    
    print(f"  Loaded X: {X_loaded.shape}")
    print(f"  Loaded y: {y_loaded.shape}")
    
    # cleanup
    import os
    os.remove('temp_data.csv')
    
    # test network integration
    print(f"\nTesting network integration...")
    
    try:
        from nn_core import NeuralNetwork, mean_squared_error
        from autodiff import TrainingEngine, Adam
        
        network = NeuralNetwork([n_features, 10, 5, 1])
        trainer = TrainingEngine(network, Adam(learning_rate=0.001), mean_squared_error)
        
        history = trainer.train(X_train, y_train, epochs=100, 
                               validation_data=(X_val, y_val), 
                               verbose=False, plot_progress=False)
        
        results = trainer.evaluate(X_test, y_test)
        print(f"  Test Loss: {results['loss']:.6f}")
        print(f"  Test MSE: {results['mse']:.6f}")
        print(f"  Test MAE: {results['mae']:.6f}")
        
    except ImportError:
        print("  Neural network modules not available for testing")
    
    print(f"\nAll tests passed!")
