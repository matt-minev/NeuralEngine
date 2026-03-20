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

            # Canonicalize raw equations so learning targets are consistent.
            self._canonicalize_equations()
            
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

    def _canonicalize_equations(self):
        """
        Canonicalize root ordering in the base dataset.
        Enforces x1 <= x2, which removes label permutation ambiguity.
        """
        if self.data is None or self.data.shape[1] < 5:
            return

        x1 = self.data[:, 3]
        x2 = self.data[:, 4]
        left = np.minimum(x1, x2)
        right = np.maximum(x1, x2)
        self.data[:, 3] = left
        self.data[:, 4] = right

    def _scenario_uses_roots_as_input(self, scenario) -> bool:
        return set(scenario.input_features) == {'x1', 'x2'}

    def _scenario_predicts_two_roots(self, scenario) -> bool:
        return set(scenario.target_features) == {'x1', 'x2'}

    def _apply_scenario_input_constraints(self, scenario, X: np.ndarray) -> np.ndarray:
        """
        Apply scenario-specific input canonicalization to avoid equivalent-input ambiguity.
        """
        Xc = np.asarray(X, dtype=np.float32).copy()
        if self._scenario_uses_roots_as_input(scenario) and Xc.shape[1] >= 2:
            x1 = Xc[:, 0]
            x2 = Xc[:, 1]
            Xc[:, 0] = np.minimum(x1, x2)
            Xc[:, 1] = np.maximum(x1, x2)
        return Xc

    def _apply_scenario_target_constraints(self, scenario, y: np.ndarray) -> np.ndarray:
        """
        Apply scenario-specific target canonicalization.
        - root outputs are sorted (x1 <= x2)
        - roots->coeff uses monic canonical form (a=1, b/a, c/a)
        """
        yc = np.asarray(y, dtype=np.float32).copy()

        if self._scenario_predicts_two_roots(scenario) and yc.shape[1] >= 2:
            r1 = yc[:, 0]
            r2 = yc[:, 1]
            yc[:, 0] = np.minimum(r1, r2)
            yc[:, 1] = np.maximum(r1, r2)
            return yc

        if scenario.name == "Roots → Coefficients" and yc.shape[1] >= 3:
            a = yc[:, 0]
            safe_a = np.where(np.abs(a) < 1e-8, 1.0, a)
            yc[:, 0] = 1.0
            yc[:, 1] = yc[:, 1] / safe_a
            yc[:, 2] = yc[:, 2] / safe_a

        return yc
    
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
        X = self._apply_scenario_input_constraints(scenario, X)
        y = self._apply_scenario_target_constraints(scenario, y)
        
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

    def prepare_scenario_splits(
        self,
        scenario,
        normalize: bool = True,
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """
        Prepare train/val/test for a scenario with leakage-safe scaling.
        Scalers are fit on training data only.
        """
        X_raw, y_raw = self.prepare_scenario_data(scenario, normalize=False)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_raw, y_raw, train_size=train_size, val_size=val_size, random_state=random_state
        )

        if not normalize:
            return (
                X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32),
                y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)
            )

        scaler_key = f"{scenario.name}_input"
        target_scaler_key = f"{scenario.name}_target"

        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        X_train_n = input_scaler.fit_transform(X_train)
        y_train_n = target_scaler.fit_transform(y_train)
        X_val_n = input_scaler.transform(X_val)
        X_test_n = input_scaler.transform(X_test)
        y_val_n = target_scaler.transform(y_val)
        y_test_n = target_scaler.transform(y_test)

        self.scalers[scaler_key] = input_scaler
        self.scalers[target_scaler_key] = target_scaler

        return (
            X_train_n.astype(np.float32),
            X_val_n.astype(np.float32),
            X_test_n.astype(np.float32),
            y_train_n.astype(np.float32),
            y_val_n.astype(np.float32),
            y_test_n.astype(np.float32),
        )
    
    def transform_input(self, scenario, input_data: np.ndarray) -> np.ndarray:
        """Transform input data using scenario-specific scaler"""
        scaler_key = f"{scenario.name}_input"
        
        if scaler_key not in self.scalers:
            raise ValueError(f"Scaler for scenario '{scenario.name}' not found. Train model first.")
            
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        constrained = self._apply_scenario_input_constraints(scenario, input_data)
        return self.scalers[scaler_key].transform(constrained)
    
    def inverse_transform_output(self, scenario, output_data: np.ndarray) -> np.ndarray:
        """Inverse transform output data using scenario-specific scaler"""
        target_scaler_key = f"{scenario.name}_target"
        
        if target_scaler_key not in self.scalers:
            raise ValueError(f"Target scaler for scenario '{scenario.name}' not found. Train model first.")
            
        output = self.scalers[target_scaler_key].inverse_transform(output_data)
        output = self._apply_scenario_target_constraints(scenario, output)
        return output
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_size: float = 0.7, val_size: float = 0.15, 
                   random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""

        # Handle small datasets
        if len(X) < 10:
            # For very small datasets, use simple splitting
            n_train = max(1, int(len(X) * 0.6))
            n_val = max(1, int(len(X) * 0.2))
            
            X_train, X_rest = X[:n_train], X[n_train:]
            y_train, y_rest = y[:n_train], y[n_train:]
            
            if len(X_rest) > 0:
                X_val, X_test = X_rest[:n_val], X_rest[n_val:]
                y_val, y_test = y_rest[:n_val], y_rest[n_val:]
            else:
                X_val = X_test = X_train
                y_val = y_test = y_train
                
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        # Deterministic split with exact counts for test expectations
        total = len(X)
        train_count = int(round(total * train_size))
        val_count = int(round(total * val_size))
        if train_count + val_count >= total:
            val_count = max(1, total - train_count - 1)
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if val_count > 1:
                val_count -= 1
            else:
                train_count -= 1

        rng = np.random.RandomState(random_state)
        indices = rng.permutation(total)
        train_idx = indices[:train_count]
        val_idx = indices[train_count:train_count + val_count]
        test_idx = indices[train_count + val_count:train_count + val_count + test_count]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

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
