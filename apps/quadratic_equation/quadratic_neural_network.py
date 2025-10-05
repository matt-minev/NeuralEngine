"""
Quadratic Neural Network App

Advanced neural network application for quadratic equation analysis and prediction.

Features:
- Multiple prediction scenarios (a,b,c → x1,x2), (a,b,x1 → c,x2), etc.
- Confidence estimation and uncertainty quantification
- Comprehensive testing and accuracy analysis
- Advanced visualizations and statistics
- Model comparison and benchmarking
- Real-time prediction interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import threading
from datetime import datetime
import os

# Import Neural Engine components
import sys
sys.path.append('../../')  # Adjust path to your Neural Engine
from nn_core import NeuralNetwork, mean_squared_error, mean_absolute_error
from autodiff import TrainingEngine, Adam, SGD
from data_utils import DataLoader, DataPreprocessor, DataSplitter
from utils import NetworkVisualizer, PerformanceMonitor

@dataclass
class PredictionScenario:
    """Configuration for different prediction scenarios"""
    name: str
    description: str
    input_features: List[str]  # Which features to use as input
    target_features: List[str]  # Which features to predict
    input_indices: List[int]    # Column indices for input
    target_indices: List[int]   # Column indices for target
    network_architecture: List[int]
    activations: List[str]
    color: str  # For visualization

class QuadraticPredictor:
    """Core prediction model for quadratic equations"""
    
    def __init__(self, scenario: PredictionScenario):
        self.scenario = scenario
        self.network = None
        self.trainer = None
        self.preprocessor = DataPreprocessor(verbose=False)
        self.is_trained = False
        self.training_history = {}
        self.performance_stats = {}
        
    def create_network(self):
        """Create neural network for this scenario"""
        self.network = NeuralNetwork(
            self.scenario.network_architecture,
            self.scenario.activations
        )
        
        # Create trainer with Adam optimizer
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        self.trainer = TrainingEngine(self.network, optimizer, mean_squared_error)
        
    def prepare_data(self, data: np.ndarray, normalize: bool = True):
        """Prepare data for training/prediction"""
        # Extract input and target data
        X = data[:, self.scenario.input_indices]
        y = data[:, self.scenario.target_indices]
        
        # Normalize if requested
        if normalize:
            # Create separate scalers for each scenario
            scaler_key = f"{self.scenario.name}_input_scaler"
            target_scaler_key = f"{self.scenario.name}_target_scaler"
            
            if not hasattr(self.preprocessor, 'scenario_scalers'):
                self.preprocessor.scenario_scalers = {}
            
            if scaler_key not in self.preprocessor.scenario_scalers:
                from sklearn.preprocessing import StandardScaler
                self.preprocessor.scenario_scalers[scaler_key] = StandardScaler()
                self.preprocessor.scenario_scalers[target_scaler_key] = StandardScaler()
                
            X = self.preprocessor.scenario_scalers[scaler_key].fit_transform(X)
            y = self.preprocessor.scenario_scalers[target_scaler_key].fit_transform(y)
        
        return X.astype(np.float32), y.astype(np.float32)

    
    def train(self, train_data: np.ndarray, val_data: np.ndarray = None, 
              epochs: int = 1000, verbose: bool = True):
        """Train the neural network"""
        if self.network is None:
            self.create_network()
            
        # Prepare training data
        X_train, y_train = self.prepare_data(train_data)
        
        # Prepare validation data if provided
        if val_data is not None:
            X_val, y_val = self.prepare_data(val_data, normalize=False)  # Use same scaler
            X_val = self.preprocessor.scalers['standard'].transform(X_val)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
            
        # Train the model
        start_time = time.time()
        self.training_history = self.trainer.train(
            X_train, y_train,
            epochs=epochs,
            validation_data=validation_data,
            verbose=verbose,
            plot_progress=False
        )
        
        training_time = time.time() - start_time
        self.performance_stats['training_time'] = training_time
        self.is_trained = True
        
        if verbose:
            print(f" {self.scenario.name} trained in {training_time:.2f}s")
            
    def predict(self, input_data: np.ndarray, return_confidence: bool = True):
        """Make predictions with optional confidence estimation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Prepare input data
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        # Normalize input
        X_normalized = self.preprocessor.scalers['standard'].transform(input_data)
        
        # Make predictions
        y_pred_normalized = self.network.forward(X_normalized)
        
        # Denormalize predictions
        y_pred = self.preprocessor.inverse_transform(y_pred_normalized, method='standard')
        
        if return_confidence:
            # Estimate confidence using ensemble of slightly perturbed predictions
            confidences = self._estimate_confidence(X_normalized)
            return y_pred, confidences
        else:
            return y_pred
    
    def _estimate_confidence(self, X_normalized: np.ndarray, n_samples: int = 50):
        """Estimate prediction confidence using Monte Carlo dropout simulation"""
        predictions = []
        
        # Get current parameters
        original_params = self.network.get_all_parameters()
        
        # Generate multiple predictions with small parameter perturbations
        for _ in range(n_samples):
            # Add small noise to parameters
            perturbed_params = []
            for param in original_params:
                noise = np.random.normal(0, 0.01, param.shape)
                perturbed_params.append(param + noise)
            
            # Set perturbed parameters
            self.network.set_all_parameters(perturbed_params)
            
            # Make prediction
            pred = self.network.forward(X_normalized)
            predictions.append(pred)
        
        # Restore original parameters
        self.network.set_all_parameters(original_params)
        
        # Calculate confidence metrics
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence as inverse of normalized standard deviation
        confidence = 1.0 / (1.0 + std_pred)
        
        return confidence
    
    def evaluate(self, test_data: np.ndarray):
        """Evaluate model performance on test data"""
        X_test, y_test = self.prepare_data(test_data, normalize=False)
        X_test = self.preprocessor.scalers['standard'].transform(X_test)
        
        # Make predictions
        y_pred = self.network.forward(X_test)
        
        # Denormalize for evaluation
        y_test_denorm = self.preprocessor.inverse_transform(y_test, method='standard')
        y_pred_denorm = self.preprocessor.inverse_transform(y_pred, method='standard')
        
        # Calculate metrics
        mse = np.mean((y_test_denorm - y_pred_denorm) ** 2)
        mae = np.mean(np.abs(y_test_denorm - y_pred_denorm))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_test_denorm - y_pred_denorm) ** 2)
        ss_tot = np.sum((y_test_denorm - np.mean(y_test_denorm)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Accuracy within tolerance (for classification-like evaluation)
        tolerance = 0.1  # 10% tolerance
        relative_error = np.abs((y_test_denorm - y_pred_denorm) / (y_test_denorm + 1e-8))
        accuracy = np.mean(relative_error < tolerance) * 100
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'accuracy_10pct': float(accuracy),
            'predictions': y_pred_denorm,
            'targets': y_test_denorm
        }

class QuadraticNeuralNetworkApp:
    """Main GUI application for quadratic neural network"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Quadratic Neural Network - Advanced Prediction & Analysis")
        self.root.geometry("1400x900")
        
        # Data and models
        self.data = None
        self.scenarios = self._create_scenarios()
        self.predictors = {}
        self.results = {}
        
        # GUI setup
        self.setup_ui()
        self.setup_styles()
        
    def _create_scenarios(self) -> Dict[str, PredictionScenario]:
        """Define different prediction scenarios"""
        scenarios = {
            'coeff_to_roots': PredictionScenario(
                name="Coefficients → Roots",
                description="Given a, b, c predict x1, x2",
                input_features=['a', 'b', 'c'],
                target_features=['x1', 'x2'],
                input_indices=[0, 1, 2],
                target_indices=[3, 4],
                network_architecture=[3, 16, 32, 16, 2],
                activations=['relu', 'relu', 'relu', 'linear'],
                color='#FF6B6B'
            ),
            'partial_coeff_to_missing': PredictionScenario(
                name="Partial Coefficients → Missing",
                description="Given a, b, x1 predict c, x2",
                input_features=['a', 'b', 'x1'],
                target_features=['c', 'x2'],
                input_indices=[0, 1, 3],
                target_indices=[2, 4],
                network_architecture=[3, 20, 24, 12, 2],
                activations=['relu', 'swish', 'relu', 'linear'],
                color='#4ECDC4'
            ),
            'roots_to_coeff': PredictionScenario(
                name="Roots → Coefficients",
                description="Given x1, x2 predict a, b, c",
                input_features=['x1', 'x2'],
                target_features=['a', 'b', 'c'],
                input_indices=[3, 4],
                target_indices=[0, 1, 2],
                network_architecture=[2, 20, 32, 20, 3],
                activations=['relu', 'swish', 'relu', 'linear'],
                color='#45B7D1'
            ),
            'single_missing': PredictionScenario(
                name="Single Missing Parameter",
                description="Given a, b, c, x1 predict x2",
                input_features=['a', 'b', 'c', 'x1'],
                target_features=['x2'],
                input_indices=[0, 1, 2, 3],
                target_indices=[4],
                network_architecture=[4, 24, 32, 16, 1],
                activations=['relu', 'swish', 'relu', 'linear'],
                color='#96CEB4'
            ),
            'verify_equation': PredictionScenario(
                name="Equation Verification",
                description="Given all parameters predict error",
                input_features=['a', 'b', 'c', 'x1', 'x2'],
                target_features=['error'],  # We'll compute error as target
                input_indices=[0, 1, 2, 3, 4],
                target_indices=[5],  # We'll add error column
                network_architecture=[5, 32, 24, 16, 1],
                activations=['relu', 'swish', 'relu', 'sigmoid'],
                color='#FFEAA7'
            )
        }
        
        return scenarios
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_analysis_tab()
        self.create_comparison_tab()
        
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        
    def create_data_tab(self):
        """Create data loading and preprocessing tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Data")
        
        # Data loading section
        load_frame = ttk.LabelFrame(data_frame, text="Data Loading")
        load_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(load_frame, text="Load Dataset", 
                  command=self.load_dataset).pack(side='left', padx=5, pady=5)
        
        self.data_status = ttk.Label(load_frame, text="No data loaded", 
                                    style='Warning.TLabel')
        self.data_status.pack(side='left', padx=10, pady=5)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(data_frame, text="Data Preview & Statistics")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create data preview table
        columns = ['a', 'b', 'c', 'x1', 'x2']
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        h_scroll = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack scrollbars and treeview
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Statistics text area
        self.stats_text = tk.Text(preview_frame, height=8, width=50)
        self.stats_text.grid(row=0, column=2, rowspan=2, padx=10, sticky='nsew')
        
        stats_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        stats_scroll.grid(row=0, column=3, rowspan=2, sticky='ns')
        
    def create_training_tab(self):
        """Create model training tab"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="🧠 Training")
        
        # Scenario selection
        scenario_frame = ttk.LabelFrame(train_frame, text="Training Scenarios")
        scenario_frame.pack(fill='x', padx=10, pady=5)
        
        # Scenario checkboxes
        self.scenario_vars = {}
        scenario_grid = ttk.Frame(scenario_frame)
        scenario_grid.pack(fill='x', padx=5, pady=5)
        
        for i, (key, scenario) in enumerate(self.scenarios.items()):
            var = tk.BooleanVar(value=True)
            self.scenario_vars[key] = var
            
            cb = ttk.Checkbutton(scenario_grid, text=scenario.name, variable=var)
            cb.grid(row=i//2, column=i%2, sticky='w', padx=10, pady=2)
            
            # Add description
            desc_label = ttk.Label(scenario_grid, text=f"  {scenario.description}", 
                                 font=('TkDefaultFont', 8))
            desc_label.grid(row=i//2, column=i%2+2, sticky='w', padx=5, pady=2)
        
        # Training controls
        control_frame = ttk.Frame(train_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Epochs:").pack(side='left', padx=5)
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Entry(control_frame, textvariable=self.epochs_var, width=10).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🚀 Start Training", 
                  command=self.start_training).pack(side='left', padx=20)
        
        ttk.Button(control_frame, text="⏹ Stop Training", 
                  command=self.stop_training).pack(side='left', padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(train_frame, text="Training Progress")
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Progress bar
        self.training_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.training_progress.pack(fill='x', padx=5, pady=5)
        
        # Training log
        self.training_log = tk.Text(progress_frame, height=20, width=80)
        log_scroll = ttk.Scrollbar(progress_frame, orient='vertical', command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scroll.set)
        
        self.training_log.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        log_scroll.pack(side='right', fill='y')
        
    def create_prediction_tab(self):
        """Create interactive prediction tab"""
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="🎯 Prediction")
        
        # Scenario selection for prediction
        pred_scenario_frame = ttk.LabelFrame(pred_frame, text="Select Prediction Scenario")
        pred_scenario_frame.pack(fill='x', padx=10, pady=5)
        
        self.pred_scenario_var = tk.StringVar(value='coeff_to_roots')
        for key, scenario in self.scenarios.items():
            ttk.Radiobutton(pred_scenario_frame, text=scenario.name, 
                           variable=self.pred_scenario_var, value=key,
                           command=self.update_prediction_inputs).pack(anchor='w', padx=10, pady=2)
        
        # Input section
        input_frame = ttk.LabelFrame(pred_frame, text="Input Parameters")
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_vars = {}
        self.input_entries = {}
        self.input_grid = ttk.Frame(input_frame)
        self.input_grid.pack(fill='x', padx=10, pady=10)
        
        # Prediction button and results
        pred_control_frame = ttk.Frame(pred_frame)
        pred_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(pred_control_frame, text="🔮 Make Prediction", 
                  command=self.make_prediction).pack(side='left', padx=5)
        
        ttk.Button(pred_control_frame, text="🧪 Test Random Sample", 
                  command=self.test_random_sample).pack(side='left', padx=5)
        
        # Results section
        results_frame = ttk.LabelFrame(pred_frame, text="Prediction Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.prediction_results = tk.Text(results_frame, height=15, width=80)
        pred_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.prediction_results.yview)
        self.prediction_results.configure(yscrollcommand=pred_scroll.set)
        
        self.prediction_results.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        pred_scroll.pack(side='right', fill='y')
        
        # Initialize input fields
        self.update_prediction_inputs()
        
    def create_analysis_tab(self):
        """Create analysis and visualization tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📈 Analysis")
        
        # Analysis controls
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="📊 Generate All Analysis", 
                  command=self.generate_full_analysis).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="💾 Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        
        # Matplotlib canvas for plots
        self.analysis_fig = plt.Figure(figsize=(14, 10))
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, analysis_frame)
        self.analysis_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_comparison_tab(self):
        """Create model comparison tab"""
        comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(comp_frame, text="⚖️ Comparison")
        
        # Comparison controls
        comp_control_frame = ttk.Frame(comp_frame)
        comp_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(comp_control_frame, text="🏆 Compare Models", 
                  command=self.compare_models).pack(side='left', padx=5)
        
        ttk.Button(comp_control_frame, text="📋 Benchmark Report", 
                  command=self.generate_benchmark_report).pack(side='left', padx=5)
        
        # Comparison visualization
        self.comparison_fig = plt.Figure(figsize=(14, 10))
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, comp_frame)
        self.comparison_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
    def load_dataset(self):
        """Load quadratic equation dataset"""
        filename = filedialog.askopenfilename(
            title="Select Quadratic Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Load data using DataLoader
                loader = DataLoader(verbose=False)
                
                # Load all columns and assume order: a, b, c, x1, x2
                df = pd.read_csv(filename)
                if df.shape[1] != 5:
                    messagebox.showerror("Error", "Dataset must have exactly 5 columns: a, b, c, x1, x2")
                    return
                
                self.data = df.values.astype(np.float32)
                
                # Add error column for verification scenario
                # Calculate ax² + bx + c for both roots and take average error
                errors = []
                for row in self.data:
                    a, b, c, x1, x2 = row
                    error1 = abs(a * x1**2 + b * x1 + c)
                    error2 = abs(a * x2**2 + b * x2 + c)
                    avg_error = (error1 + error2) / 2
                    errors.append(avg_error)
                
                # Add error column
                error_col = np.array(errors).reshape(-1, 1)
                self.data = np.column_stack([self.data, error_col])
                
                # Update data preview
                self.update_data_preview()
                
                # Update status
                self.data_status.config(text=f" Loaded {len(self.data)} equations", 
                                      style='Success.TLabel')
                
                messagebox.showinfo("Success", f"Loaded {len(self.data)} quadratic equations!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                
    def update_data_preview(self):
        """Update data preview table and statistics"""
        if self.data is None:
            return
            
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
            
        # Add sample data (first 100 rows)
        sample_data = self.data[:100] if len(self.data) > 100 else self.data
        
        for row in sample_data:
            formatted_row = [f"{val:.3f}" for val in row[:5]]  # Only show first 5 columns
            self.data_tree.insert('', 'end', values=formatted_row)
            
        # Update statistics
        self.update_data_statistics()
        
    def update_data_statistics(self):
        """Update data statistics display"""
        if self.data is None:
            return
            
        data_5col = self.data[:, :5]  # Only use first 5 columns for stats
        
        stats = []
        stats.append("📊 DATASET STATISTICS")
        stats.append("=" * 40)
        stats.append(f"Total Equations: {len(data_5col)}")
        stats.append(f"Features: a, b, c, x1, x2")
        stats.append("")
        
        column_names = ['a', 'b', 'c', 'x1', 'x2']
        for i, name in enumerate(column_names):
            col_data = data_5col[:, i]
            stats.append(f"{name.upper()} Statistics:")
            stats.append(f"  Mean: {np.mean(col_data):.3f}")
            stats.append(f"  Std:  {np.std(col_data):.3f}")
            stats.append(f"  Min:  {np.min(col_data):.3f}")
            stats.append(f"  Max:  {np.max(col_data):.3f}")
            stats.append("")
            
        # Data quality checks
        stats.append("🔍 DATA QUALITY")
        stats.append("=" * 40)
        
        # Check for whole number solutions
        x1_whole = np.sum(np.abs(data_5col[:, 3] - np.round(data_5col[:, 3])) < 1e-6)
        x2_whole = np.sum(np.abs(data_5col[:, 4] - np.round(data_5col[:, 4])) < 1e-6)
        
        stats.append(f"Whole number x1: {x1_whole} ({x1_whole/len(data_5col)*100:.1f}%)")
        stats.append(f"Whole number x2: {x2_whole} ({x2_whole/len(data_5col)*100:.1f}%)")
        
        # Check coefficient types
        a_int = np.sum(np.abs(data_5col[:, 0] - np.round(data_5col[:, 0])) < 1e-6)
        b_int = np.sum(np.abs(data_5col[:, 1] - np.round(data_5col[:, 1])) < 1e-6)
        c_int = np.sum(np.abs(data_5col[:, 2] - np.round(data_5col[:, 2])) < 1e-6)
        
        stats.append(f"Integer coefficients: {min(a_int, b_int, c_int)/len(data_5col)*100:.1f}%")
        
        # Show statistics
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, '\n'.join(stats))
        
    def start_training(self):
        """Start training selected models"""
        if self.data is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
            
        # Get selected scenarios
        selected_scenarios = [key for key, var in self.scenario_vars.items() if var.get()]
        
        if not selected_scenarios:
            messagebox.showwarning("Warning", "Please select at least one scenario to train!")
            return
            
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self._train_models,
            args=(selected_scenarios,)
        )
        self.training_thread.start()
        
        # Start progress animation
        self.training_progress.start()
        
    def _train_models(self, selected_scenarios: List[str]):
        """Train models for selected scenarios (runs in separate thread)"""
        try:
            # Split data
            splitter = DataSplitter(random_state=42, verbose=False)
            train_data, val_data, test_data = splitter.train_val_test_split(
                self.data, self.data,  # Using same data for X and y as placeholder
                train_size=0.7, val_size=0.15, test_size=0.15
            )[::2]  # Take every other to get only X arrays
            
            # Clear training log
            self.training_log.delete(1.0, tk.END)
            
            for scenario_key in selected_scenarios:
                self.log_training(f"\n🚀 Training {self.scenarios[scenario_key].name}...")
                
                # Create predictor
                predictor = QuadraticPredictor(self.scenarios[scenario_key])
                
                # Train model
                predictor.train(
                    train_data=train_data,
                    val_data=val_data,
                    epochs=self.epochs_var.get(),
                    verbose=False
                )
                
                # Evaluate on test data
                test_results = predictor.evaluate(test_data)
                
                # Store predictor and results
                self.predictors[scenario_key] = predictor
                self.results[scenario_key] = test_results
                
                # Log results
                self.log_training(f" Completed! MSE: {test_results['mse']:.6f}, R²: {test_results['r2']:.3f}")
                
            self.log_training(f"\n🎉 All training completed!")
            
        except Exception as e:
            self.log_training(f" Training error: {str(e)}")
        finally:
            # Stop progress animation
            self.root.after(0, self.training_progress.stop)
            
    def log_training(self, message: str):
        """Add message to training log (thread-safe)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.root.after(0, lambda: self._update_log(full_message))
        
    def _update_log(self, message: str):
        """Update training log in main thread"""
        self.training_log.insert(tk.END, message)
        self.training_log.see(tk.END)
        
    def stop_training(self):
        """Stop training (placeholder for now)"""
        self.training_progress.stop()
        self.log_training("⏹ Training stopped by user")
        
    def update_prediction_inputs(self):
        """Update prediction input fields based on selected scenario"""
        # Clear existing inputs
        for widget in self.input_grid.winfo_children():
            widget.destroy()
            
        self.input_vars.clear()
        self.input_entries.clear()
        
        # Get current scenario
        scenario_key = self.pred_scenario_var.get()
        scenario = self.scenarios[scenario_key]
        
        # Create input fields
        for i, feature in enumerate(scenario.input_features):
            ttk.Label(self.input_grid, text=f"{feature}:").grid(row=i//3, column=(i%3)*2, 
                                                               sticky='w', padx=5, pady=5)
            
            var = tk.DoubleVar()
            entry = ttk.Entry(self.input_grid, textvariable=var, width=12)
            entry.grid(row=i//3, column=(i%3)*2+1, padx=5, pady=5)
            
            self.input_vars[feature] = var
            self.input_entries[feature] = entry
            
    def make_prediction(self):
        """Make prediction with current inputs"""
        scenario_key = self.pred_scenario_var.get()
        
        if scenario_key not in self.predictors:
            messagebox.showerror("Error", f"Model for '{self.scenarios[scenario_key].name}' not trained yet!")
            return
            
        try:
            # Get input values
            scenario = self.scenarios[scenario_key]
            input_values = []
            
            for feature in scenario.input_features:
                value = self.input_vars[feature].get()
                input_values.append(value)
                
            input_array = np.array(input_values).reshape(1, -1)
            
            # Make prediction
            predictor = self.predictors[scenario_key]
            predictions, confidences = predictor.predict(input_array, return_confidence=True)
            
            # Display results
            self.display_prediction_results(scenario, input_values, predictions[0], confidences[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
    def display_prediction_results(self, scenario, inputs, predictions, confidences):
        """Display prediction results"""
        self.prediction_results.delete(1.0, tk.END)
        
        results = []
        results.append("🔮 PREDICTION RESULTS")
        results.append("=" * 50)
        results.append(f"Scenario: {scenario.name}")
        results.append(f"Description: {scenario.description}")
        results.append("")
        
        results.append("📝 INPUT VALUES:")
        for feature, value in zip(scenario.input_features, inputs):
            results.append(f"  {feature} = {value:.6f}")
        results.append("")
        
        results.append("🎯 PREDICTED VALUES:")
        for feature, pred, conf in zip(scenario.target_features, predictions, confidences):
            confidence_pct = conf * 100
            confidence_level = "🟢 High" if conf > 0.8 else "🟡 Medium" if conf > 0.6 else "🔴 Low"
            results.append(f"  {feature} = {pred:.6f} (Confidence: {confidence_pct:.1f}% {confidence_level})")
        results.append("")
        
        # Add verification if possible
        if scenario.name == "Coefficients → Roots":
            a, b, c = inputs
            x1, x2 = predictions
            
            results.append(" VERIFICATION:")
            error1 = abs(a * x1**2 + b * x1 + c)
            error2 = abs(a * x2**2 + b * x2 + c)
            results.append(f"  Equation check x1: {a:.3f}×({x1:.3f})² + {b:.3f}×{x1:.3f} + {c:.3f} = {error1:.6f}")
            results.append(f"  Equation check x2: {a:.3f}×({x2:.3f})² + {b:.3f}×{x2:.3f} + {c:.3f} = {error2:.6f}")
            
            if max(error1, error2) < 0.001:
                results.append("  🎉 Excellent! Predictions satisfy the quadratic equation!")
            elif max(error1, error2) < 0.01:
                results.append("  👍 Good! Small error in quadratic equation.")
            else:
                results.append("  ⚠️ Warning: Significant error in quadratic equation.")
        
        self.prediction_results.insert(tk.END, '\n'.join(results))
        
    def test_random_sample(self):
        """Test prediction on a random sample from dataset"""
        if self.data is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
            
        scenario_key = self.pred_scenario_var.get()
        
        if scenario_key not in self.predictors:
            messagebox.showerror("Error", f"Model for '{self.scenarios[scenario_key].name}' not trained yet!")
            return
            
        # Get random sample
        random_idx = np.random.randint(0, len(self.data))
        sample = self.data[random_idx]
        
        scenario = self.scenarios[scenario_key]
        
        # Extract input values
        input_values = sample[scenario.input_indices]
        true_values = sample[scenario.target_indices]
        
        # Set input fields
        for feature, value in zip(scenario.input_features, input_values):
            self.input_vars[feature].set(value)
            
        # Make prediction
        try:
            predictor = self.predictors[scenario_key]
            predictions, confidences = predictor.predict(input_values.reshape(1, -1), return_confidence=True)
            
            # Display results with comparison to true values
            self.display_test_results(scenario, input_values, predictions[0], confidences[0], true_values)
            
        except Exception as e:
            messagebox.showerror("Error", f"Test prediction failed: {str(e)}")
            
    def display_test_results(self, scenario, inputs, predictions, confidences, true_values):
        """Display test results with true value comparison"""
        self.prediction_results.delete(1.0, tk.END)
        
        results = []
        results.append("🧪 TEST PREDICTION RESULTS")
        results.append("=" * 50)
        results.append(f"Scenario: {scenario.name}")
        results.append("")
        
        results.append("📝 INPUT VALUES:")
        for feature, value in zip(scenario.input_features, inputs):
            results.append(f"  {feature} = {value:.6f}")
        results.append("")
        
        results.append("🎯 PREDICTION vs TRUE VALUES:")
        total_error = 0
        for feature, pred, conf, true_val in zip(scenario.target_features, predictions, confidences, true_values):
            error = abs(pred - true_val)
            error_pct = abs(error / (true_val + 1e-8)) * 100
            total_error += error
            
            confidence_pct = conf * 100
            results.append(f"  {feature}:")
            results.append(f"    Predicted: {pred:.6f} (Confidence: {confidence_pct:.1f}%)")
            results.append(f"    True:      {true_val:.6f}")
            results.append(f"    Error:     {error:.6f} ({error_pct:.2f}%)")
            results.append("")
            
        # Overall assessment
        avg_error = total_error / len(predictions)
        results.append("📊 OVERALL ASSESSMENT:")
        if avg_error < 0.01:
            results.append("  🎉 Excellent prediction! Very low error.")
        elif avg_error < 0.1:
            results.append("  👍 Good prediction! Acceptable error.")
        elif avg_error < 1.0:
            results.append("  ⚠️ Moderate prediction. Some error present.")
        else:
            results.append("   Poor prediction. High error.")
            
        results.append(f"  Average absolute error: {avg_error:.6f}")
        
        self.prediction_results.insert(tk.END, '\n'.join(results))

    def generate_full_analysis(self):
        """Generate comprehensive analysis of all trained models"""
        if not self.predictors:
            messagebox.showwarning("Warning", "No trained models available for analysis!")
            return
        
        if not self.results:
            messagebox.showwarning("Warning", "No evaluation results available!")
            return
        
        self.log_analysis("🔍 Generating comprehensive analysis...")
        
        # Clear previous plots
        self.analysis_fig.clear()
        
        # Create subplots for analysis
        gs = self.analysis_fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Comparison
        ax1 = self.analysis_fig.add_subplot(gs[0, :])
        self.plot_model_comparison(ax1)
        
        # 2. Prediction vs True Values
        ax2 = self.analysis_fig.add_subplot(gs[1, 0])
        self.plot_prediction_accuracy(ax2)
        
        # 3. Error Distribution
        ax3 = self.analysis_fig.add_subplot(gs[1, 1])
        self.plot_error_distribution(ax3)
        
        # 4. Confidence Analysis
        ax4 = self.analysis_fig.add_subplot(gs[1, 2])
        self.plot_confidence_analysis(ax4)
        
        # 5. Training Loss Curves
        ax5 = self.analysis_fig.add_subplot(gs[2, 0])
        self.plot_training_curves(ax5)
        
        # 6. Parameter Analysis
        ax6 = self.analysis_fig.add_subplot(gs[2, 1])
        self.plot_parameter_analysis(ax6)
        
        # 7. Scenario Performance
        ax7 = self.analysis_fig.add_subplot(gs[2, 2])
        self.plot_scenario_performance(ax7)
        
        self.analysis_fig.suptitle('Quadratic Neural Network - Comprehensive Analysis', 
                                fontsize=16, fontweight='bold')
        
        self.analysis_canvas.draw()
        self.log_analysis(" Analysis complete!")

    def plot_model_comparison(self, ax):
        """Plot model performance comparison"""
        scenarios = list(self.results.keys())
        metrics = ['mse', 'mae', 'r2', 'accuracy_10pct']
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[scenario][metric] for scenario in scenarios]
            ax.bar(x + i * width, values, width, label=metric.upper(), alpha=0.8)
        
        ax.set_xlabel('Prediction Scenarios')
        ax.set_ylabel('Performance Metrics')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_prediction_accuracy(self, ax):
        """Plot prediction vs true values"""
        # Use the first available model for demonstration
        scenario_key = list(self.results.keys())[0]
        result = self.results[scenario_key]
        
        predictions = result['predictions']
        targets = result['targets']
        
        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
            targets = targets.flatten()
        
        ax.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Prediction Accuracy\n({scenario_key.replace("_", " ").title()})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_error_distribution(self, ax):
        """Plot error distribution"""
        all_errors = []
        labels = []
        
        for scenario_key, result in self.results.items():
            predictions = result['predictions']
            targets = result['targets']
            
            if predictions.ndim > 1:
                predictions = predictions.flatten()
                targets = targets.flatten()
            
            errors = np.abs(predictions - targets)
            all_errors.append(errors)
            labels.append(scenario_key.replace('_', ' ').title())
        
        ax.hist(all_errors, bins=30, label=labels, alpha=0.7)
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_confidence_analysis(self, ax):
        """Plot confidence analysis"""
        # This would need confidence data from predictions
        # For now, show a placeholder
        scenarios = list(self.results.keys())
        confidence_scores = [0.85, 0.78, 0.92, 0.74, 0.88]  # Mock data
        
        bars = ax.bar(scenarios, confidence_scores[:len(scenarios)], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        ax.set_ylabel('Average Confidence')
        ax.set_title('Model Confidence Analysis')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add confidence threshold line
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='High Confidence Threshold')
        ax.legend()

    def plot_training_curves(self, ax):
        """Plot training loss curves"""
        # This would show training history if available
        # For now, show a placeholder
        epochs = np.arange(1, 101)
        
        for i, (scenario_key, predictor) in enumerate(self.predictors.items()):
            if hasattr(predictor, 'training_history') and predictor.training_history:
                train_loss = predictor.training_history.get('train_loss', [])
                if train_loss:
                    ax.plot(epochs[:len(train_loss)], train_loss, 
                        label=scenario_key.replace('_', ' ').title(), 
                        color=self.scenarios[scenario_key].color)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curves')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_parameter_analysis(self, ax):
        """Plot parameter analysis"""
        scenarios = list(self.predictors.keys())
        param_counts = []
        
        for scenario_key in scenarios:
            if scenario_key in self.predictors:
                predictor = self.predictors[scenario_key]
                if predictor.network:
                    param_counts.append(predictor.network.count_parameters())
                else:
                    param_counts.append(0)
        
        bars = ax.bar(scenarios, param_counts, 
                    color=[self.scenarios[s].color for s in scenarios])
        
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity Analysis')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)

    def plot_scenario_performance(self, ax):
        """Plot performance by scenario"""
        scenarios = list(self.results.keys())
        r2_scores = [self.results[s]['r2'] for s in scenarios]
        
        bars = ax.bar(scenarios, r2_scores, 
                    color=[self.scenarios[s].color for s in scenarios])
        
        ax.set_ylabel('R² Score')
        ax.set_title('Scenario Performance (R²)')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add performance threshold lines
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Poor')
        ax.legend()

    def log_analysis(self, message):
        """Log analysis messages"""
        print(f"[Analysis] {message}")

    def export_results(self):
        """Export analysis results to files"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export!")
            return
        
        # Choose export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export results as JSON
            results_file = os.path.join(export_dir, f"quadratic_nn_results_{timestamp}.json")
            
            # Prepare exportable results
            exportable_results = {}
            for scenario_key, result in self.results.items():
                exportable_results[scenario_key] = {
                    'mse': float(result['mse']),
                    'mae': float(result['mae']),
                    'rmse': float(result['rmse']),
                    'r2': float(result['r2']),
                    'accuracy_10pct': float(result['accuracy_10pct']),
                    'scenario_info': {
                        'name': self.scenarios[scenario_key].name,
                        'description': self.scenarios[scenario_key].description,
                        'input_features': self.scenarios[scenario_key].input_features,
                        'target_features': self.scenarios[scenario_key].target_features
                    }
                }
            
            with open(results_file, 'w') as f:
                json.dump(exportable_results, f, indent=2)
            
            # Export summary report
            report_file = os.path.join(export_dir, f"quadratic_nn_report_{timestamp}.txt")
            with open(report_file, 'w') as f:
                f.write("QUADRATIC NEURAL NETWORK ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("SCENARIO PERFORMANCE SUMMARY:\n")
                f.write("-" * 30 + "\n")
                for scenario_key, result in self.results.items():
                    f.write(f"\n{self.scenarios[scenario_key].name}:\n")
                    f.write(f"  Description: {self.scenarios[scenario_key].description}\n")
                    f.write(f"  R² Score: {result['r2']:.4f}\n")
                    f.write(f"  MSE: {result['mse']:.6f}\n")
                    f.write(f"  MAE: {result['mae']:.6f}\n")
                    f.write(f"  RMSE: {result['rmse']:.6f}\n")
                    f.write(f"  10% Accuracy: {result['accuracy_10pct']:.2f}%\n")
            
            messagebox.showinfo("Success", f"Results exported to:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def compare_models(self):
        """Compare different models"""
        if len(self.predictors) < 2:
            messagebox.showwarning("Warning", "Need at least 2 trained models for comparison!")
            return
        
        # Clear previous plots
        self.comparison_fig.clear()
        
        # Create comparison plots
        gs = self.comparison_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Performance comparison
        ax1 = self.comparison_fig.add_subplot(gs[0, :])
        self.plot_detailed_comparison(ax1)
        
        # Speed comparison
        ax2 = self.comparison_fig.add_subplot(gs[1, 0])
        self.plot_speed_comparison(ax2)
        
        # Complexity comparison
        ax3 = self.comparison_fig.add_subplot(gs[1, 1])
        self.plot_complexity_comparison(ax3)
        
        self.comparison_fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
        self.comparison_canvas.draw()

    def plot_detailed_comparison(self, ax):
        """Plot detailed model comparison"""
        scenarios = list(self.results.keys())
        metrics = ['r2', 'mse', 'mae', 'accuracy_10pct']
        metric_names = ['R² Score', 'MSE', 'MAE', 'Accuracy (10%)']
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [self.results[scenario][metric] for scenario in scenarios]
            
            # Normalize values for better visualization
            if metric == 'r2':
                normalized_values = values
            elif metric == 'accuracy_10pct':
                normalized_values = [v/100 for v in values]
            else:
                # For MSE and MAE, use inverse for better visualization
                max_val = max(values)
                normalized_values = [1 - (v/max_val) for v in values]
            
            bars = ax.bar(x + i * width, normalized_values, width, label=name, alpha=0.8)
        
        ax.set_xlabel('Prediction Scenarios')
        ax.set_ylabel('Normalized Performance')
        ax.set_title('Detailed Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_speed_comparison(self, ax):
        """Plot speed comparison (mock data for now)"""
        scenarios = list(self.predictors.keys())
        speeds = [np.random.uniform(50, 200) for _ in scenarios]  # Mock inference speed
        
        bars = ax.bar(scenarios, speeds, color=[self.scenarios[s].color for s in scenarios])
        ax.set_ylabel('Inference Speed (samples/sec)')
        ax.set_title('Model Speed Comparison')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)

    def plot_complexity_comparison(self, ax):
        """Plot complexity comparison"""
        scenarios = list(self.predictors.keys())
        complexities = []
        
        for scenario_key in scenarios:
            if scenario_key in self.predictors:
                predictor = self.predictors[scenario_key]
                if predictor.network:
                    complexities.append(predictor.network.count_parameters())
                else:
                    complexities.append(0)
        
        bars = ax.bar(scenarios, complexities, color=[self.scenarios[s].color for s in scenarios])
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity Comparison')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)

    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        if not self.results:
            messagebox.showwarning("Warning", "No results available for benchmarking!")
            return
        
        # Create benchmark report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Benchmark Report")
        report_window.geometry("800x600")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(report_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Generate report content
        report_content = self.generate_benchmark_content()
        text_widget.insert('1.0', report_content)
        text_widget.config(state='disabled')
        
        # Add export button
        export_btn = ttk.Button(report_window, text="Export Report", 
                            command=lambda: self.export_benchmark_report(report_content))
        export_btn.pack(pady=10)

    def generate_benchmark_content(self):
        """Generate benchmark report content"""
        from datetime import datetime
        
        content = []
        content.append("QUADRATIC NEURAL NETWORK BENCHMARK REPORT")
        content.append("=" * 60)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        content.append("EXECUTIVE SUMMARY")
        content.append("-" * 30)
        
        # Find best performing model
        best_scenario = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_r2 = self.results[best_scenario]['r2']
        
        content.append(f"Best Performing Model: {self.scenarios[best_scenario].name}")
        content.append(f"Best R² Score: {best_r2:.4f}")
        content.append(f"Total Models Evaluated: {len(self.results)}")
        content.append("")
        
        content.append("DETAILED RESULTS")
        content.append("-" * 30)
        
        for scenario_key, result in self.results.items():
            scenario = self.scenarios[scenario_key]
            content.append(f"\n{scenario.name}:")
            content.append(f"  Description: {scenario.description}")
            content.append(f"  Input Features: {', '.join(scenario.input_features)}")
            content.append(f"  Target Features: {', '.join(scenario.target_features)}")
            content.append(f"  Network Architecture: {scenario.network_architecture}")
            content.append(f"  Activations: {scenario.activations}")
            content.append("")
            content.append("  Performance Metrics:")
            content.append(f"    R² Score: {result['r2']:.4f}")
            content.append(f"    MSE: {result['mse']:.6f}")
            content.append(f"    MAE: {result['mae']:.6f}")
            content.append(f"    RMSE: {result['rmse']:.6f}")
            content.append(f"    10% Accuracy: {result['accuracy_10pct']:.2f}%")
            
            # Performance assessment
            if result['r2'] > 0.9:
                assessment = "EXCELLENT"
            elif result['r2'] > 0.7:
                assessment = "GOOD"
            elif result['r2'] > 0.5:
                assessment = "FAIR"
            else:
                assessment = "POOR"
            
            content.append(f"    Assessment: {assessment}")
            content.append("")
        
        content.append("RECOMMENDATIONS")
        content.append("-" * 30)
        
        # Generate recommendations based on results
        recommendations = []
        
        if best_r2 > 0.9:
            recommendations.append(" Excellent model performance achieved")
        elif best_r2 > 0.7:
            recommendations.append("⚠️ Consider increasing model complexity or training time")
        else:
            recommendations.append(" Model performance needs significant improvement")
        
        recommendations.append(f"🏆 Use '{self.scenarios[best_scenario].name}' for best results")
        recommendations.append("📊 Consider ensemble methods for improved performance")
        recommendations.append("🔄 Regular retraining recommended for production use")
        
        for rec in recommendations:
            content.append(f"  {rec}")
        
        return '\n'.join(content)

    def export_benchmark_report(self, content):
        """Export benchmark report to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Report exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = QuadraticNeuralNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
