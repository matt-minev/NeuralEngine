import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class QuadraticDataProcessor:
    """Handles data loading, preprocessing, and splitting for quadratic equations"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.data = None
        self.scalers = {}
        self.data_stats = {}
        
    def load_data(self, filepath: str) -> bool:
        """Load quadratic equation dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            # Validate data format
            if df.shape[1] != 5:
                raise ValueError("Dataset must have exactly 5 columns: a, b, c, x1, x2")
                
            self.data = df.values.astype(np.float32)
            
            # Add error column for verification scenario
            self._add_error_column()
            
            # Calculate statistics
            self._calculate_stats()
            
            if self.verbose:
                print(f"✅ Loaded {len(self.data)} quadratic equations")
                
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load data: {str(e)}")
            return False
    
    def _add_error_column(self):
        """Add error column for equation verification"""
        errors = []
        for row in self.data:
            a, b, c, x1, x2 = row
            # Calculate ax² + bx + c for both roots
            error1 = abs(a * x1**2 + b * x1 + c)
            error2 = abs(a * x2**2 + b * x2 + c)
            avg_error = (error1 + error2) / 2
            errors.append(avg_error)
        
        # Add error column
        error_col = np.array(errors).reshape(-1, 1)
        self.data = np.column_stack([self.data, error_col])
    
    def _calculate_stats(self):
        """Calculate dataset statistics"""
        if self.data is None:
            return
            
        data_5col = self.data[:, :5]  # Only first 5 columns for stats
        column_names = ['a', 'b', 'c', 'x1', 'x2']
        
        self.data_stats = {
            'total_equations': len(data_5col),
            'columns': {}
        }
        
        for i, name in enumerate(column_names):
            col_data = data_5col[:, i]
            self.data_stats['columns'][name] = {
                'mean': float(np.mean(col_data)),
                'std': float(np.std(col_data)),
                'min': float(np.min(col_data)),
                'max': float(np.max(col_data))
            }
        
        # Data quality metrics
        x1_whole = np.sum(np.abs(data_5col[:, 3] - np.round(data_5col[:, 3])) < 1e-6)
        x2_whole = np.sum(np.abs(data_5col[:, 4] - np.round(data_5col[:, 4])) < 1e-6)
        
        self.data_stats['quality'] = {
            'x1_whole_pct': float(x1_whole / len(data_5col) * 100),
            'x2_whole_pct': float(x2_whole / len(data_5col) * 100)
        }
    
    def prepare_scenario_data(self, scenario, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for a specific scenario"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Extract input and target data
        X = self.data[:, scenario.input_indices]
        y = self.data[:, scenario.target_indices]
        
        if normalize:
            # Create scenario-specific scalers
            scaler_key = f"{scenario.name}_input"
            target_scaler_key = f"{scenario.name}_target"
            
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                self.scalers[target_scaler_key] = StandardScaler()
                
            X = self.scalers[scaler_key].fit_transform(X)
            y = self.scalers[target_scaler_key].fit_transform(y)
        
        return X.astype(np.float32), y.astype(np.float32)
    
    def transform_input(self, scenario, input_data: np.ndarray) -> np.ndarray:
        """Transform input data using scenario-specific scaler"""
        scaler_key = f"{scenario.name}_input"
        
        if scaler_key not in self.scalers:
            raise ValueError(f"Scaler for scenario '{scenario.name}' not found. Train model first.")
            
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        return self.scalers[scaler_key].transform(input_data)
    
    def inverse_transform_output(self, scenario, output_data: np.ndarray) -> np.ndarray:
        """Inverse transform output data using scenario-specific scaler"""
        target_scaler_key = f"{scenario.name}_target"
        
        if target_scaler_key not in self.scalers:
            raise ValueError(f"Target scaler for scenario '{scenario.name}' not found. Train model first.")
            
        return self.scalers[target_scaler_key].inverse_transform(output_data)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_size: float = 0.7, val_size: float = 0.15, 
                   random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_size), random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + (1 - train_size - val_size))
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_sample_data(self, n_samples: int = 100) -> np.ndarray:
        """Get sample data for preview"""
        if self.data is None:
            return np.array([])
            
        sample_size = min(n_samples, len(self.data))
        return self.data[:sample_size, :5]  # Only first 5 columns
    
    def get_stats(self) -> dict:
        """Get dataset statistics"""
        return self.data_stats
