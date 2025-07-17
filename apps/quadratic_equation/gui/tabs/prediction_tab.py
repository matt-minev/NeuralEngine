import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Dict, Any

class PredictionTab:
    """Interactive prediction tab"""
    
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="🎯 Prediction")
        
        self.input_vars = {}
        self.input_entries = {}
        self.current_scenario = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup prediction tab interface"""
        self.create_scenario_selection()
        self.create_input_section()
        self.create_prediction_controls()
        self.create_results_section()
        
    def create_scenario_selection(self):
        """Create scenario selection for prediction"""
        scenario_frame = ttk.LabelFrame(self.frame, text="Prediction Scenario")
        scenario_frame.pack(fill='x', padx=10, pady=5)
        
        self.scenario_var = tk.StringVar(value='coeff_to_roots')
        
        # Create radio buttons for scenarios
        scenarios_grid = ttk.Frame(scenario_frame)
        scenarios_grid.pack(fill='x', padx=10, pady=5)
        
        for i, (key, scenario) in enumerate(self.app.scenarios.items()):
            rb = ttk.Radiobutton(scenarios_grid, text=scenario.name, 
                               variable=self.scenario_var, value=key,
                               command=self.update_input_fields)
            rb.grid(row=i//2, column=(i%2)*2, sticky='w', padx=5, pady=2)
            
            # Description
            desc_label = ttk.Label(scenarios_grid, text=f"  {scenario.description}", 
                                 font=('TkDefaultFont', 8), foreground='gray')
            desc_label.grid(row=i//2, column=(i%2)*2+1, sticky='w', padx=5, pady=2)
            
    def create_input_section(self):
        """Create input parameter section"""
        input_frame = ttk.LabelFrame(self.frame, text="Input Parameters")
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_grid = ttk.Frame(input_frame)
        self.input_grid.pack(fill='x', padx=10, pady=10)
        
        # Initialize with default scenario
        self.update_input_fields()
        
    def create_prediction_controls(self):
        """Create prediction control buttons"""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="🔮 Make Prediction", 
                  command=self.make_prediction).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🎲 Random Test", 
                  command=self.test_random_sample).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🧪 Batch Test", 
                  command=self.batch_test).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🗑 Clear Results", 
                  command=self.clear_results).pack(side='left', padx=5)
        
    def create_results_section(self):
        """Create results display section"""
        results_frame = ttk.LabelFrame(self.frame, text="Prediction Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=15, width=80, font=('Courier', 9))
        results_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        results_scroll.pack(side='right', fill='y')
        
    def update_input_fields(self):
        """Update input fields based on selected scenario"""
        # Clear existing inputs
        for widget in self.input_grid.winfo_children():
            widget.destroy()
            
        self.input_vars.clear()
        self.input_entries.clear()
        
        # Get current scenario
        scenario_key = self.scenario_var.get()
        if scenario_key not in self.app.scenarios:
            return
            
        scenario = self.app.scenarios[scenario_key]
        self.current_scenario = scenario
        
        # Create input fields
        for i, feature in enumerate(scenario.input_features):
            row = i // 3
            col = (i % 3) * 2
            
            # Label
            ttk.Label(self.input_grid, text=f"{feature}:").grid(
                row=row, column=col, sticky='w', padx=5, pady=5
            )
            
            # Entry
            var = tk.DoubleVar()
            entry = ttk.Entry(self.input_grid, textvariable=var, width=12)
            entry.grid(row=row, column=col+1, padx=5, pady=5)
            
            self.input_vars[feature] = var
            self.input_entries[feature] = entry
            
    def make_prediction(self):
        """Make prediction with current inputs"""
        if not self.validate_inputs():
            return
            
        scenario_key = self.scenario_var.get()
        
        if scenario_key not in self.app.predictors:
            messagebox.showerror("Error", f"Model for '{self.current_scenario.name}' not trained yet!")
            return
            
        try:
            # Get input values
            input_values = []
            for feature in self.current_scenario.input_features:
                value = self.input_vars[feature].get()
                input_values.append(value)
                
            input_array = np.array(input_values).reshape(1, -1)
            
            # Make prediction
            predictor = self.app.predictors[scenario_key]
            predictions, confidences = predictor.predict(input_array, return_confidence=True)
            
            # Display results
            self.display_prediction_results(input_values, predictions[0], confidences[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
    def test_random_sample(self):
        """Test prediction on random sample from dataset"""
        if not self.validate_model():
            return
            
        if self.app.data_processor.data is None:
            messagebox.showerror("Error", "No dataset loaded!")
            return
            
        # Get random sample
        random_idx = np.random.randint(0, len(self.app.data_processor.data))
        sample = self.app.data_processor.data[random_idx]
        
        # Extract input values and set in UI
        input_values = sample[self.current_scenario.input_indices]
        true_values = sample[self.current_scenario.target_indices]
        
        # Set input fields
        for feature, value in zip(self.current_scenario.input_features, input_values):
            self.input_vars[feature].set(value)
            
        # Make prediction
        try:
            predictor = self.app.predictors[self.scenario_var.get()]
            predictions, confidences = predictor.predict(input_values.reshape(1, -1), return_confidence=True)
            
            # Display results with comparison
            self.display_test_results(input_values, predictions[0], confidences[0], true_values)
            
        except Exception as e:
            messagebox.showerror("Error", f"Test prediction failed: {str(e)}")
            
    def batch_test(self):
        """Run batch testing on multiple samples"""
        if not self.validate_model():
            return
            
        if self.app.data_processor.data is None:
            messagebox.showerror("Error", "No dataset loaded!")
            return
            
        try:
            # Get test samples
            n_samples = min(10, len(self.app.data_processor.data))
            test_indices = np.random.choice(len(self.app.data_processor.data), n_samples, replace=False)
            
            results = []
            scenario_key = self.scenario_var.get()
            predictor = self.app.predictors[scenario_key]
            
            for i, idx in enumerate(test_indices):
                sample = self.app.data_processor.data[idx]
                input_values = sample[self.current_scenario.input_indices]
                true_values = sample[self.current_scenario.target_indices]
                
                predictions, confidences = predictor.predict(input_values.reshape(1, -1), return_confidence=True)
                
                # Calculate errors
                errors = np.abs(predictions[0] - true_values)
                avg_error = np.mean(errors)
                avg_confidence = np.mean(confidences[0])
                
                results.append({
                    'sample': i + 1,
                    'inputs': input_values,
                    'predictions': predictions[0],
                    'true_values': true_values,
                    'errors': errors,
                    'avg_error': avg_error,
                    'avg_confidence': avg_confidence
                })
                
            self.display_batch_results(results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch test failed: {str(e)}")
            
    def validate_inputs(self):
        """Validate input values"""
        if not self.current_scenario:
            messagebox.showerror("Error", "No scenario selected!")
            return False
            
        try:
            for feature in self.current_scenario.input_features:
                self.input_vars[feature].get()
            return True
        except tk.TclError:
            messagebox.showerror("Error", "Please enter valid numeric values for all inputs!")
            return False
            
    def validate_model(self):
        """Validate that model is trained"""
        scenario_key = self.scenario_var.get()
        if scenario_key not in self.app.predictors:
            messagebox.showerror("Error", f"Model for '{self.current_scenario.name}' not trained yet!")
            return False
        return True
        
    def display_prediction_results(self, inputs, predictions, confidences):
        """Display prediction results with clear solution comparison"""
        self.results_text.delete(1.0, tk.END)
        
        # Calculate actual solutions using quadratic formula
        actual_solutions = self.calculate_actual_solutions(inputs)
        
        results = []
        results.append("🔮 NEURAL NETWORK PREDICTION RESULTS")
        results.append("=" * 60)
        results.append(f"Scenario: {self.current_scenario.name}")
        results.append(f"Description: {self.current_scenario.description}")
        results.append("")
        
        # Input section with larger, clearer display
        results.append("📝 INPUT EQUATION:")
        if self.current_scenario.name == "Coefficients → Roots":
            a, b, c = inputs
            results.append(f"   {a:.3f}x² + {b:.3f}x + {c:.3f} = 0")
        else:
            results.append("📊 INPUT VALUES:")
            for feature, value in zip(self.current_scenario.input_features, inputs):
                results.append(f"   {feature} = {value:.6f}")
        results.append("")
        
        # Main comparison section
        results.append("🎯 SOLUTION COMPARISON:")
        results.append("-" * 40)
        
        if actual_solutions:
            for i, (feature, predicted, confidence) in enumerate(zip(
                self.current_scenario.target_features, predictions, confidences)):
                
                actual = actual_solutions[i] if i < len(actual_solutions) else "N/A"
                confidence_pct = confidence * 100
                
                # Calculate error
                if actual != "N/A":
                    error = abs(predicted - actual)
                    error_pct = abs(error / (actual + 1e-8)) * 100 if actual != 0 else error * 100
                    
                    # Error assessment
                    if error < 0.01:
                        error_status = "🎉 EXCELLENT"
                    elif error < 0.1:
                        error_status = "👍 GOOD"
                    elif error < 1.0:
                        error_status = "⚠️ MODERATE"
                    else:
                        error_status = "❌ POOR"
                        
                    results.append(f"   {feature.upper()}:")
                    results.append(f"     🤖 Predicted:  {predicted:.6f}")
                    results.append(f"     ✅ Actual:     {actual:.6f}")
                    results.append(f"     📊 Error:      {error:.6f} ({error_pct:.2f}%) {error_status}")
                    results.append(f"     🎯 Confidence: {confidence_pct:.1f}%")
                    results.append("")
                else:
                    results.append(f"   {feature.upper()}:")
                    results.append(f"     🤖 Predicted:  {predicted:.6f}")
                    results.append(f"     🎯 Confidence: {confidence_pct:.1f}%")
                    results.append("")
        else:
            # Fallback for scenarios where we can't calculate actual solutions
            results.append("   📊 PREDICTED VALUES:")
            for feature, pred, conf in zip(self.current_scenario.target_features, predictions, confidences):
                confidence_pct = conf * 100
                confidence_level = "🟢 High" if conf > 0.8 else "🟡 Medium" if conf > 0.6 else "🔴 Low"
                results.append(f"     {feature} = {pred:.6f} (Confidence: {confidence_pct:.1f}% {confidence_level})")
            results.append("")
        
        # Overall assessment
        if actual_solutions and len(actual_solutions) == len(predictions):
            total_error = sum(abs(pred - act) for pred, act in zip(predictions, actual_solutions) if act != "N/A")
            avg_error = total_error / len(predictions)
            
            results.append("📈 OVERALL PERFORMANCE:")
            results.append("-" * 25)
            if avg_error < 0.01:
                results.append("   🏆 OUTSTANDING: Neural network is highly accurate!")
            elif avg_error < 0.1:
                results.append("   ✅ EXCELLENT: Very good predictions with low error")
            elif avg_error < 1.0:
                results.append("   ⚠️ FAIR: Moderate accuracy, needs improvement")
            else:
                results.append("   ❌ POOR: High error, model needs retraining")
            
            results.append(f"   Average Error: {avg_error:.6f}")
            
            # Add equation verification for quadratic roots
            if self.current_scenario.name == "Coefficients → Roots":
                self.add_equation_verification(results, inputs, predictions, actual_solutions)
        
        # Display all results
        self.results_text.insert(tk.END, '\n'.join(results))
        
    def display_test_results(self, inputs, predictions, confidences, true_values):
        """Display test results with clear true value comparison"""
        self.results_text.delete(1.0, tk.END)
        
        results = []
        results.append("🧪 RANDOM DATASET TEST RESULTS")
        results.append("=" * 60)
        results.append(f"Scenario: {self.current_scenario.name}")
        results.append("")
        
        # Input section
        results.append("📝 INPUT VALUES FROM DATASET:")
        for feature, value in zip(self.current_scenario.input_features, inputs):
            results.append(f"   {feature} = {value:.6f}")
        results.append("")
        
        # Main comparison
        results.append("🎯 PREDICTION vs DATASET VALUES:")
        results.append("-" * 45)
        
        total_error = 0
        for feature, pred, conf, true_val in zip(
            self.current_scenario.target_features, predictions, confidences, true_values):
            
            error = abs(pred - true_val)
            error_pct = abs(error / (true_val + 1e-8)) * 100
            total_error += error
            
            confidence_pct = conf * 100
            
            # Error assessment
            if error < 0.01:
                error_status = "🎉 EXCELLENT"
            elif error < 0.1:
                error_status = "👍 GOOD"
            elif error < 1.0:
                error_status = "⚠️ MODERATE"
            else:
                error_status = "❌ POOR"
            
            results.append(f"   {feature.upper()}:")
            results.append(f"     🤖 Predicted:    {pred:.6f}")
            results.append(f"     📊 Dataset:      {true_val:.6f}")
            results.append(f"     📈 Error:        {error:.6f} ({error_pct:.2f}%) {error_status}")
            results.append(f"     🎯 Confidence:   {confidence_pct:.1f}%")
            results.append("")
        
        # Overall assessment
        avg_error = total_error / len(predictions)
        results.append("📊 OVERALL TEST ASSESSMENT:")
        results.append("-" * 30)
        
        if avg_error < 0.01:
            results.append("   🏆 OUTSTANDING: Predictions match dataset very closely!")
        elif avg_error < 0.1:
            results.append("   ✅ EXCELLENT: Good accuracy on this test sample")
        elif avg_error < 1.0:
            results.append("   ⚠️ FAIR: Moderate accuracy, room for improvement")
        else:
            results.append("   ❌ POOR: High error, model may need retraining")
        
        results.append(f"   Average Absolute Error: {avg_error:.6f}")
        
        self.results_text.insert(tk.END, '\n'.join(results))

        
    def display_batch_results(self, results):
        """Display batch test results"""
        self.results_text.delete(1.0, tk.END)
        
        output = []
        output.append("🧪 BATCH TEST RESULTS")
        output.append("=" * 50)
        output.append(f"Scenario: {self.current_scenario.name}")
        output.append(f"Samples tested: {len(results)}")
        output.append("")
        
        # Summary statistics
        avg_errors = [r['avg_error'] for r in results]
        avg_confidences = [r['avg_confidence'] for r in results]
        
        output.append("📊 SUMMARY STATISTICS:")
        output.append(f"  Average Error: {np.mean(avg_errors):.6f}")
        output.append(f"  Error Std Dev: {np.std(avg_errors):.6f}")
        output.append(f"  Average Confidence: {np.mean(avg_confidences):.3f}")
        output.append(f"  Best Error: {np.min(avg_errors):.6f}")
        output.append(f"  Worst Error: {np.max(avg_errors):.6f}")
        output.append("")
        
        # Individual results
        output.append("📝 INDIVIDUAL RESULTS:")
        for result in results:
            output.append(f"  Sample {result['sample']}:")
            output.append(f"    Avg Error: {result['avg_error']:.6f}")
            output.append(f"    Avg Confidence: {result['avg_confidence']:.3f}")
            
            # Show predictions vs true values
            for i, (feature, pred, true_val) in enumerate(zip(
                self.current_scenario.target_features, 
                result['predictions'], 
                result['true_values']
            )):
                output.append(f"    {feature}: {pred:.3f} → {true_val:.3f} (error: {result['errors'][i]:.3f})")
            output.append("")
            
        self.results_text.insert(tk.END, '\n'.join(output))
        
    def add_equation_verification(self, results, inputs, predictions, actual_solutions):
        """Add detailed equation verification"""
        a, b, c = inputs
        pred_x1, pred_x2 = predictions
        actual_x1, actual_x2 = actual_solutions
        
        results.append("")
        results.append("🔍 EQUATION VERIFICATION:")
        results.append("-" * 30)
        
        # Test predicted solutions
        pred_error1 = abs(a * pred_x1**2 + b * pred_x1 + c)
        pred_error2 = abs(a * pred_x2**2 + b * pred_x2 + c)
        
        # Test actual solutions (should be ~0)
        actual_error1 = abs(a * actual_x1**2 + b * actual_x1 + c)
        actual_error2 = abs(a * actual_x2**2 + b * actual_x2 + c)
        
        results.append("   Testing Predicted Solutions:")
        results.append(f"     x₁ = {pred_x1:.6f} → Error: {pred_error1:.8f}")
        results.append(f"     x₂ = {pred_x2:.6f} → Error: {pred_error2:.8f}")
        results.append("")
        
        results.append("   Testing Actual Solutions:")
        results.append(f"     x₁ = {actual_x1:.6f} → Error: {actual_error1:.8f}")
        results.append(f"     x₂ = {actual_x2:.6f} → Error: {actual_error2:.8f}")
        results.append("")
        
        # Overall verification status
        max_pred_error = max(pred_error1, pred_error2)
        if max_pred_error < 0.001:
            results.append("   ✅ PERFECT: Predicted solutions satisfy the equation!")
        elif max_pred_error < 0.01:
            results.append("   👍 GOOD: Small error in equation satisfaction")
        elif max_pred_error < 1.0:
            results.append("   ⚠️ MODERATE: Noticeable error in equation satisfaction")
        else:
            results.append("   ❌ POOR: Predicted solutions don't satisfy the equation")

            
    def clear_results(self):
        """Clear results display"""
        self.results_text.delete(1.0, tk.END)
        
    def refresh(self):
        """Refresh tab after data or model changes"""
        # Update input fields for current scenario
        self.update_input_fields()
        
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status (interface for main app)"""
        pass  # Prediction tab doesn't have a separate status display

    def calculate_actual_solutions(self, inputs):
        """Calculate the actual correct solutions for comparison"""
        try:
            if self.current_scenario.name == "Coefficients → Roots":
                # For quadratic equations, calculate using quadratic formula
                a, b, c = inputs
                
                # Handle linear case (a = 0)
                if abs(a) < 1e-10:
                    if abs(b) < 1e-10:
                        return None  # No solution or infinite solutions
                    else:
                        root = -c / b
                        return [root, root]
                
                # Calculate discriminant
                discriminant = b**2 - 4*a*c
                
                if discriminant < 0:
                    return None  # No real solutions
                
                # Calculate the two roots
                sqrt_discriminant = np.sqrt(discriminant)
                x1 = (-b + sqrt_discriminant) / (2*a)
                x2 = (-b - sqrt_discriminant) / (2*a)
                
                return [x1, x2]
            
            else:
                # For other scenarios, we might not be able to calculate actual solutions
                # This would need to be implemented based on the specific scenario
                return None
                
        except Exception as e:
            print(f"Error calculating actual solutions: {e}")
            return None

