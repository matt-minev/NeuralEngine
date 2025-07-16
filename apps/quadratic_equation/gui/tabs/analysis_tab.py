import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import seaborn as sns

class AnalysisTab:
    """Analysis and visualization tab"""
    
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="📈 Analysis")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup analysis tab interface"""
        self.create_controls()
        self.create_visualization()
        
    def create_controls(self):
        """Create analysis control buttons"""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="📊 Generate Analysis", 
                  command=self.generate_analysis).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="📈 Performance Plots", 
                  command=self.create_performance_plots).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🔍 Error Analysis", 
                  command=self.create_error_analysis).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="💾 Export Analysis", 
                  command=self.export_analysis).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🗑 Clear Plots", 
                  command=self.clear_plots).pack(side='left', padx=5)
        
    def create_visualization(self):
        """Create matplotlib visualization area"""
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(14, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create toolbar
        toolbar_frame = ttk.Frame(self.frame)
        toolbar_frame.pack(fill='x', padx=10, pady=5)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def generate_analysis(self):
        """Generate comprehensive analysis"""
        if not self.app.results:
            messagebox.showwarning("Warning", "No trained models available for analysis!")
            return
            
        # Clear previous plots
        self.fig.clear()
        
        # Create comprehensive analysis
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Comparison
        ax1 = self.fig.add_subplot(gs[0, :])
        self.plot_model_comparison(ax1)
        
        # 2. Prediction Accuracy
        ax2 = self.fig.add_subplot(gs[1, 0])
        self.plot_prediction_accuracy(ax2)
        
        # 3. Error Distribution
        ax3 = self.fig.add_subplot(gs[1, 1])
        self.plot_error_distribution(ax3)
        
        # 4. R² Score Comparison
        ax4 = self.fig.add_subplot(gs[1, 2])
        self.plot_r2_comparison(ax4)
        
        # 5. Training Time Analysis
        ax5 = self.fig.add_subplot(gs[2, 0])
        self.plot_training_time(ax5)
        
        # 6. Model Complexity
        ax6 = self.fig.add_subplot(gs[2, 1])
        self.plot_model_complexity(ax6)
        
        # 7. Accuracy vs Complexity
        ax7 = self.fig.add_subplot(gs[2, 2])
        self.plot_accuracy_vs_complexity(ax7)
        
        self.fig.suptitle('Quadratic Neural Network - Comprehensive Analysis', 
                         fontsize=16, fontweight='bold')
        
        self.canvas.draw()
        
    def create_performance_plots(self):
        """Create detailed performance plots"""
        if not self.app.results:
            messagebox.showwarning("Warning", "No results available!")
            return
            
        self.fig.clear()
        
        # Create performance-focused plots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Performance metrics heatmap
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.plot_performance_heatmap(ax1)
        
        # Accuracy trends
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.plot_accuracy_trends(ax2)
        
        # Error analysis
        ax3 = self.fig.add_subplot(gs[1, 0])
        self.plot_detailed_error_analysis(ax3)
        
        # Scenario rankings
        ax4 = self.fig.add_subplot(gs[1, 1])
        self.plot_scenario_rankings(ax4)
        
        self.fig.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        self.canvas.draw()
        
    def create_error_analysis(self):
        """Create detailed error analysis"""
        if not self.app.results:
            messagebox.showwarning("Warning", "No results available!")
            return
            
        self.fig.clear()
        
        # Create error-focused plots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Error distribution by scenario
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.plot_error_by_scenario(ax1)
        
        # MSE vs MAE comparison
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.plot_mse_vs_mae(ax2)
        
        # Error patterns
        ax3 = self.fig.add_subplot(gs[1, 0])
        self.plot_error_patterns(ax3)
        
        # Residual analysis
        ax4 = self.fig.add_subplot(gs[1, 1])
        self.plot_residual_analysis(ax4)
        
        self.fig.suptitle('Error Analysis Dashboard', fontsize=16, fontweight='bold')
        self.canvas.draw()
        
    def plot_model_comparison(self, ax):
        """Plot model performance comparison"""
        scenarios = list(self.app.results.keys())
        metrics = ['r2', 'mse', 'mae', 'accuracy_10pct']
        metric_names = ['R² Score', 'MSE', 'MAE', 'Accuracy (10%)']
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [self.app.results[scenario][metric] for scenario in scenarios]
            
            # Normalize values for better visualization
            if metric == 'r2':
                normalized_values = values
            elif metric == 'accuracy_10pct':
                normalized_values = [v/100 for v in values]
            else:
                # For MSE and MAE, use inverse for better visualization
                max_val = max(values) if values else 1
                normalized_values = [1 - (v/max_val) for v in values]
            
            bars = ax.bar(x + i * width, normalized_values, width, label=name, alpha=0.8)
        
        ax.set_xlabel('Prediction Scenarios')
        ax.set_ylabel('Normalized Performance')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_prediction_accuracy(self, ax):
        """Plot prediction accuracy"""
        scenarios = list(self.app.results.keys())
        accuracies = [self.app.results[s]['accuracy_10pct'] for s in scenarios]
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, accuracies, color=colors, alpha=0.8)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Prediction Accuracy (10% Tolerance)')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom')
                   
    def plot_error_distribution(self, ax):
        """Plot error distribution"""
        scenarios = list(self.app.results.keys())
        mse_values = [self.app.results[s]['mse'] for s in scenarios]
        mae_values = [self.app.results[s]['mae'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mse_values, width, label='MSE', alpha=0.8)
        bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
        
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Error Value')
        ax.set_title('Error Distribution (MSE vs MAE)')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_r2_comparison(self, ax):
        """Plot R² score comparison"""
        scenarios = list(self.app.results.keys())
        r2_scores = [self.app.results[s]['r2'] for s in scenarios]
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, r2_scores, color=colors, alpha=0.8)
        
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Good')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Poor')
        ax.legend()
        
    def plot_training_time(self, ax):
        """Plot training time analysis"""
        scenarios = list(self.app.predictors.keys())
        training_times = []
        
        for scenario in scenarios:
            if scenario in self.app.predictors:
                predictor = self.app.predictors[scenario]
                training_time = predictor.performance_stats.get('training_time', 0)
                training_times.append(training_time)
            else:
                training_times.append(0)
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, training_times, color=colors, alpha=0.8)
        
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Analysis')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
    def plot_model_complexity(self, ax):
        """Plot model complexity analysis"""
        scenarios = list(self.app.scenarios.keys())
        param_counts = []
        
        for scenario in scenarios:
            if scenario in self.app.predictors:
                predictor = self.app.predictors[scenario]
                if predictor.network:
                    param_counts.append(predictor.network.count_parameters())
                else:
                    param_counts.append(0)
            else:
                # Calculate from architecture
                arch = self.app.scenarios[scenario].network_architecture
                params = sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch)-1))
                param_counts.append(params)
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, param_counts, color=colors, alpha=0.8)
        
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity Analysis')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
    def plot_accuracy_vs_complexity(self, ax):
        """Plot accuracy vs complexity scatter"""
        scenarios = list(self.app.results.keys())
        
        accuracies = []
        complexities = []
        
        for scenario in scenarios:
            accuracies.append(self.app.results[scenario]['r2'])
            
            if scenario in self.app.predictors:
                predictor = self.app.predictors[scenario]
                if predictor.network:
                    complexities.append(predictor.network.count_parameters())
                else:
                    complexities.append(0)
            else:
                complexities.append(0)
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        scatter = ax.scatter(complexities, accuracies, c=colors, alpha=0.8, s=100)
        
        # Add labels
        for i, scenario in enumerate(scenarios):
            ax.annotate(scenario.replace('_', ' ').title(), 
                       (complexities[i], accuracies[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Model Complexity (Parameters)')
        ax.set_ylabel('R² Score')
        ax.set_title('Accuracy vs Complexity')
        ax.grid(True, alpha=0.3)
        
    def plot_performance_heatmap(self, ax):
        """Plot performance metrics heatmap"""
        scenarios = list(self.app.results.keys())
        metrics = ['r2', 'mse', 'mae', 'accuracy_10pct']
        
        # Create data matrix
        data = []
        for metric in metrics:
            values = [self.app.results[s][metric] for s in scenarios]
            # Normalize values
            if metric in ['mse', 'mae']:
                values = [1 - (v/max(values)) for v in values]  # Invert for better visualization
            elif metric == 'accuracy_10pct':
                values = [v/100 for v in values]
            data.append(values)
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(scenarios)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.set_yticklabels(['R²', 'MSE (inv)', 'MAE (inv)', 'Accuracy'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add value annotations
        for i in range(len(metrics)):
            for j in range(len(scenarios)):
                text = ax.text(j, i, f'{data[i][j]:.3f}', 
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Performance Metrics Heatmap')
        
    def plot_accuracy_trends(self, ax):
        """Plot accuracy trends"""
        scenarios = list(self.app.results.keys())
        r2_scores = [self.app.results[s]['r2'] for s in scenarios]
        
        ax.plot(scenarios, r2_scores, marker='o', linewidth=2, markersize=8)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Trends')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Fill area under curve
        ax.fill_between(range(len(scenarios)), r2_scores, alpha=0.3)
        
    def plot_detailed_error_analysis(self, ax):
        """Plot detailed error analysis"""
        scenarios = list(self.app.results.keys())
        
        # Create error comparison
        mse_values = [self.app.results[s]['mse'] for s in scenarios]
        mae_values = [self.app.results[s]['mae'] for s in scenarios]
        rmse_values = [self.app.results[s]['rmse'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        bars1 = ax.bar(x - width, mse_values, width, label='MSE', alpha=0.8)
        bars2 = ax.bar(x, mae_values, width, label='MAE', alpha=0.8)
        bars3 = ax.bar(x + width, rmse_values, width, label='RMSE', alpha=0.8)
        
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Error Value')
        ax.set_title('Detailed Error Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_scenario_rankings(self, ax):
        """Plot scenario rankings"""
        scenarios = list(self.app.results.keys())
        
        # Calculate overall scores (weighted combination)
        scores = []
        for scenario in scenarios:
            result = self.app.results[scenario]
            # Weighted score: R² (40%) + Accuracy (30%) + MSE penalty (20%) + MAE penalty (10%)
            score = (result['r2'] * 0.4 + 
                    result['accuracy_10pct']/100 * 0.3 + 
                    (1 - result['mse']) * 0.2 + 
                    (1 - result['mae']) * 0.1)
            scores.append(score)
        
        # Sort by score
        scenario_score_pairs = list(zip(scenarios, scores))
        scenario_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        sorted_scenarios = [pair[0] for pair in scenario_score_pairs]
        sorted_scores = [pair[1] for pair in scenario_score_pairs]
        
        colors = [self.app.scenarios[s].color for s in sorted_scenarios]
        bars = ax.barh(sorted_scenarios, sorted_scores, color=colors, alpha=0.8)
        
        ax.set_xlabel('Overall Score')
        ax.set_title('Scenario Rankings')
        ax.set_yticklabels([s.replace('_', ' ').title() for s in sorted_scenarios])
        ax.grid(True, alpha=0.3)
        
    def plot_error_by_scenario(self, ax):
        """Plot error by scenario"""
        scenarios = list(self.app.results.keys())
        errors = [self.app.results[s]['rmse'] for s in scenarios]
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, errors, color=colors, alpha=0.8)
        
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Square Error by Scenario')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
    def plot_mse_vs_mae(self, ax):
        """Plot MSE vs MAE comparison"""
        scenarios = list(self.app.results.keys())
        mse_values = [self.app.results[s]['mse'] for s in scenarios]
        mae_values = [self.app.results[s]['mae'] for s in scenarios]
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        scatter = ax.scatter(mse_values, mae_values, c=colors, alpha=0.8, s=100)
        
        # Add labels
        for i, scenario in enumerate(scenarios):
            ax.annotate(scenario.replace('_', ' ').title(), 
                       (mse_values[i], mae_values[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('MSE')
        ax.set_ylabel('MAE')
        ax.set_title('MSE vs MAE Comparison')
        ax.grid(True, alpha=0.3)
        
    def plot_error_patterns(self, ax):
        """Plot error patterns"""
        scenarios = list(self.app.results.keys())
        
        # Create error pattern visualization
        error_types = ['MSE', 'MAE', 'RMSE']
        
        for i, scenario in enumerate(scenarios):
            result = self.app.results[scenario]
            errors = [result['mse'], result['mae'], result['rmse']]
            
            ax.plot(error_types, errors, marker='o', label=scenario.replace('_', ' ').title(),
                   color=self.app.scenarios[scenario].color, linewidth=2)
        
        ax.set_ylabel('Error Value')
        ax.set_title('Error Patterns Across Scenarios')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_residual_analysis(self, ax):
        """Plot residual analysis (placeholder)"""
        # This would require actual prediction data
        # For now, show a placeholder
        ax.text(0.5, 0.5, 'Residual Analysis\n(Requires prediction data)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Residual Analysis')
        
    def export_analysis(self):
        """Export analysis results"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Analysis exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export analysis: {str(e)}")
                
    def clear_plots(self):
        """Clear all plots"""
        self.fig.clear()
        self.canvas.draw()
        
    def refresh(self):
        """Refresh tab after data or model changes"""
        # Clear plots when data changes
        self.clear_plots()
        
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status (interface for main app)"""
        pass  # Analysis tab doesn't have a separate status display
