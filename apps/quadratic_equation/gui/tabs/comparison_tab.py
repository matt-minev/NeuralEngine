import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import json

class ComparisonTab:
    """Model comparison and benchmarking tab"""
    
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="⚖️ Comparison")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup comparison tab interface"""
        self.create_controls()
        self.create_visualization()
        self.create_results_section()
        
    def create_controls(self):
        """Create comparison control buttons"""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="🏆 Compare Models", 
                  command=self.compare_models).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="📊 Benchmark Report", 
                  command=self.generate_benchmark_report).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="📈 Performance Matrix", 
                  command=self.create_performance_matrix).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="💾 Export Report", 
                  command=self.export_comparison_report).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🗑 Clear", 
                  command=self.clear_comparison).pack(side='left', padx=5)
        
    def create_visualization(self):
        """Create matplotlib visualization area"""
        viz_frame = ttk.LabelFrame(self.frame, text="Comparison Visualization")
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_results_section(self):
        """Create results text section"""
        results_frame = ttk.LabelFrame(self.frame, text="Comparison Results")
        results_frame.pack(fill='x', padx=10, pady=5)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=10, width=80, font=('Courier', 9))
        results_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        results_scroll.pack(side='right', fill='y')
        
    def compare_models(self):
        """Compare all trained models"""
        if len(self.app.predictors) < 2:
            messagebox.showwarning("Warning", "Need at least 2 trained models for comparison!")
            return
            
        # Clear previous plots
        self.fig.clear()
        
        # Create comparison plots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall Performance Comparison
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.plot_overall_performance(ax1)
        
        # 2. Speed vs Accuracy
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.plot_speed_vs_accuracy(ax2)
        
        # 3. Complexity Analysis
        ax3 = self.fig.add_subplot(gs[0, 2])
        self.plot_complexity_analysis(ax3)
        
        # 4. Error Comparison
        ax4 = self.fig.add_subplot(gs[1, 0])
        self.plot_error_comparison(ax4)
        
        # 5. Confidence Analysis
        ax5 = self.fig.add_subplot(gs[1, 1])
        self.plot_confidence_comparison(ax5)
        
        # 6. Recommendation Score
        ax6 = self.fig.add_subplot(gs[1, 2])
        self.plot_recommendation_score(ax6)
        
        self.fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
        self.canvas.draw()
        
        # Generate text summary
        self.generate_comparison_summary()
        
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        if not self.app.results:
            messagebox.showwarning("Warning", "No results available for benchmarking!")
            return
            
        # Create benchmark report window
        self.create_benchmark_window()
        
    def create_performance_matrix(self):
        """Create performance matrix visualization"""
        if not self.app.results:
            messagebox.showwarning("Warning", "No results available!")
            return
            
        self.fig.clear()
        
        # Create performance matrix
        scenarios = list(self.app.results.keys())
        metrics = ['R²', 'MSE', 'MAE', 'RMSE', 'Accuracy']
        
        # Prepare data matrix
        data_matrix = []
        for scenario in scenarios:
            result = self.app.results[scenario]
            row = [
                result['r2'],
                result['mse'],
                result['mae'],
                result['rmse'],
                result['accuracy_10pct'] / 100
            ]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        ax = self.fig.add_subplot(111)
        im = ax.imshow(data_matrix.T, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(scenarios)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.set_yticklabels(metrics)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score')
        
        # Add value annotations
        for i in range(len(metrics)):
            for j in range(len(scenarios)):
                text = ax.text(j, i, f'{data_matrix[j, i]:.3f}', 
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Performance Matrix - All Models', fontsize=14, fontweight='bold')
        self.canvas.draw()
        
    def plot_overall_performance(self, ax):
        """Plot overall performance comparison"""
        scenarios = list(self.app.results.keys())
        
        # Calculate composite scores
        composite_scores = []
        for scenario in scenarios:
            result = self.app.results[scenario]
            # Weighted composite score
            score = (result['r2'] * 0.4 + 
                    result['accuracy_10pct']/100 * 0.3 + 
                    (1 - min(result['mse'], 1)) * 0.2 + 
                    (1 - min(result['mae'], 1)) * 0.1)
            composite_scores.append(score)
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, composite_scores, color=colors, alpha=0.8)
        
        ax.set_ylabel('Composite Score')
        ax.set_title('Overall Performance Ranking')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, composite_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom')
                   
    def plot_speed_vs_accuracy(self, ax):
        """Plot speed vs accuracy comparison"""
        scenarios = list(self.app.results.keys())
        
        accuracies = [self.app.results[s]['r2'] for s in scenarios]
        # Mock speed data (in real implementation, measure actual inference speed)
        speeds = [np.random.uniform(100, 500) for _ in scenarios]
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        scatter = ax.scatter(speeds, accuracies, c=colors, alpha=0.8, s=100)
        
        # Add labels
        for i, scenario in enumerate(scenarios):
            ax.annotate(scenario.replace('_', ' ').title(), 
                       (speeds[i], accuracies[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Inference Speed (samples/sec)')
        ax.set_ylabel('R² Score')
        ax.set_title('Speed vs Accuracy Trade-off')
        ax.grid(True, alpha=0.3)
        
    def plot_complexity_analysis(self, ax):
        """Plot model complexity analysis"""
        scenarios = list(self.app.scenarios.keys())
        
        # Calculate complexity metrics
        param_counts = []
        layer_counts = []
        
        for scenario in scenarios:
            arch = self.app.scenarios[scenario].network_architecture
            
            # Parameter count estimation
            params = sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch)-1))
            param_counts.append(params)
            
            # Layer count
            layer_counts.append(len(arch))
        
        # Create bubble chart
        colors = [self.app.scenarios[s].color for s in scenarios]
        
        # Use R² scores for bubble size if available
        if scenarios[0] in self.app.results:
            sizes = [self.app.results[s]['r2'] * 500 for s in scenarios]
        else:
            sizes = [100] * len(scenarios)
        
        scatter = ax.scatter(param_counts, layer_counts, s=sizes, c=colors, alpha=0.6)
        
        # Add labels
        for i, scenario in enumerate(scenarios):
            ax.annotate(scenario.replace('_', ' ').title(), 
                       (param_counts[i], layer_counts[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
        
        ax.set_xlabel('Parameter Count')
        ax.set_ylabel('Layer Count')
        ax.set_title('Model Complexity Analysis')
        ax.grid(True, alpha=0.3)
        
    def plot_error_comparison(self, ax):
        """Plot error comparison"""
        scenarios = list(self.app.results.keys())
        
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
        ax.set_title('Error Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_confidence_comparison(self, ax):
        """Plot confidence comparison (mock data)"""
        scenarios = list(self.app.results.keys())
        
        # Mock confidence data (in real implementation, calculate from predictions)
        confidence_scores = [0.85, 0.78, 0.92, 0.74, 0.88][:len(scenarios)]
        
        colors = [self.app.scenarios[s].color for s in scenarios]
        bars = ax.bar(scenarios, confidence_scores, color=colors, alpha=0.8)
        
        ax.set_ylabel('Average Confidence')
        ax.set_title('Model Confidence Comparison')
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add confidence threshold
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='High Confidence')
        ax.legend()
        
    def plot_recommendation_score(self, ax):
        """Plot recommendation scores"""
        scenarios = list(self.app.results.keys())
        
        # Calculate recommendation scores based on multiple factors
        rec_scores = []
        for scenario in scenarios:
            result = self.app.results[scenario]
            
            # Factors: accuracy, stability, complexity, use case
            accuracy_score = result['r2']
            stability_score = 1 - result['mse']  # Lower MSE = higher stability
            complexity_penalty = 0.1  # Assume all models have similar complexity
            
            rec_score = (accuracy_score * 0.5 + 
                        stability_score * 0.3 + 
                        complexity_penalty * 0.2)
            rec_scores.append(rec_score)
        
        # Sort by recommendation score
        scenario_score_pairs = list(zip(scenarios, rec_scores))
        scenario_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        sorted_scenarios = [pair[0] for pair in scenario_score_pairs]
        sorted_scores = [pair[1] for pair in scenario_score_pairs]
        
        colors = [self.app.scenarios[s].color for s in sorted_scenarios]
        bars = ax.barh(sorted_scenarios, sorted_scores, color=colors, alpha=0.8)
        
        ax.set_xlabel('Recommendation Score')
        ax.set_title('Model Recommendations')
        ax.set_yticklabels([s.replace('_', ' ').title() for s in sorted_scenarios])
        ax.grid(True, alpha=0.3)
        
    def generate_comparison_summary(self):
        """Generate text summary of comparison"""
        if not self.app.results:
            return
            
        summary = []
        summary.append("🏆 MODEL COMPARISON SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Models Compared: {len(self.app.results)}")
        summary.append("")
        
        # Find best performers
        best_r2 = max(self.app.results.items(), key=lambda x: x[1]['r2'])
        best_accuracy = max(self.app.results.items(), key=lambda x: x[1]['accuracy_10pct'])
        lowest_error = min(self.app.results.items(), key=lambda x: x[1]['mse'])
        
        summary.append("🥇 TOP PERFORMERS:")
        summary.append(f"  Best R²: {self.app.scenarios[best_r2[0]].name} ({best_r2[1]['r2']:.4f})")
        summary.append(f"  Best Accuracy: {self.app.scenarios[best_accuracy[0]].name} ({best_accuracy[1]['accuracy_10pct']:.2f}%)")
        summary.append(f"  Lowest Error: {self.app.scenarios[lowest_error[0]].name} (MSE: {lowest_error[1]['mse']:.6f})")
        summary.append("")
        
        # Performance breakdown
        summary.append("📊 PERFORMANCE BREAKDOWN:")
        for scenario_key, result in self.app.results.items():
            scenario_name = self.app.scenarios[scenario_key].name
            summary.append(f"  {scenario_name}:")
            summary.append(f"    R²: {result['r2']:.4f} | MSE: {result['mse']:.6f} | Accuracy: {result['accuracy_10pct']:.2f}%")
            
        summary.append("")
        
        # Recommendations
        summary.append("💡 RECOMMENDATIONS:")
        if best_r2[1]['r2'] > 0.9:
            summary.append("  ✅ Excellent model performance achieved!")
        elif best_r2[1]['r2'] > 0.7:
            summary.append("  ⚠️ Good performance, consider optimization for production use.")
        else:
            summary.append("  ❌ Model performance needs improvement.")
            
        summary.append(f"  🎯 Recommended model: {self.app.scenarios[best_r2[0]].name}")
        summary.append("  📈 Consider ensemble methods for improved robustness.")
        
        # Display summary
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, '\n'.join(summary))
        
    def create_benchmark_window(self):
        """Create benchmark report window"""
        benchmark_window = tk.Toplevel(self.app.root)
        benchmark_window.title("Benchmark Report")
        benchmark_window.geometry("900x700")
        
        # Create text widget
        text_frame = ttk.Frame(benchmark_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Generate report content
        report_content = self.generate_detailed_benchmark_report()
        text_widget.insert('1.0', report_content)
        text_widget.config(state='disabled')
        
        # Export button
        export_btn = ttk.Button(benchmark_window, text="💾 Export Report", 
                               command=lambda: self.export_benchmark_report(report_content))
        export_btn.pack(pady=10)
        
    def generate_detailed_benchmark_report(self):
        """Generate detailed benchmark report"""
        content = []
        content.append("QUADRATIC NEURAL NETWORK BENCHMARK REPORT")
        content.append("=" * 60)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Location: Varna, Bulgaria")
        content.append("")
        
        # Executive Summary
        content.append("EXECUTIVE SUMMARY")
        content.append("-" * 30)
        
        if self.app.results:
            best_scenario = max(self.app.results.keys(), key=lambda k: self.app.results[k]['r2'])
            best_r2 = self.app.results[best_scenario]['r2']
            
            content.append(f"Best Performing Model: {self.app.scenarios[best_scenario].name}")
            content.append(f"Best R² Score: {best_r2:.4f}")
            content.append(f"Total Models Evaluated: {len(self.app.results)}")
            content.append("")
            
            # Performance assessment
            if best_r2 > 0.9:
                assessment = "EXCELLENT - Production ready"
            elif best_r2 > 0.7:
                assessment = "GOOD - Suitable for most applications"
            elif best_r2 > 0.5:
                assessment = "FAIR - Needs improvement"
            else:
                assessment = "POOR - Requires significant work"
                
            content.append(f"Overall Assessment: {assessment}")
            content.append("")
        
        # Detailed Results
        content.append("DETAILED RESULTS")
        content.append("-" * 30)
        
        for scenario_key, result in self.app.results.items():
            scenario = self.app.scenarios[scenario_key]
            content.append(f"\n{scenario.name}")
            content.append(f"  Description: {scenario.description}")
            content.append(f"  Input Features: {', '.join(scenario.input_features)}")
            content.append(f"  Target Features: {', '.join(scenario.target_features)}")
            content.append(f"  Network Architecture: {scenario.network_architecture}")
            content.append("")
            content.append("  Performance Metrics:")
            content.append(f"    R² Score: {result['r2']:.4f}")
            content.append(f"    Mean Squared Error: {result['mse']:.6f}")
            content.append(f"    Mean Absolute Error: {result['mae']:.6f}")
            content.append(f"    Root Mean Squared Error: {result['rmse']:.6f}")
            content.append(f"    10% Accuracy: {result['accuracy_10pct']:.2f}%")
            content.append("")
            
            # Individual assessment
            if result['r2'] > 0.9:
                individual_assessment = "EXCELLENT"
            elif result['r2'] > 0.7:
                individual_assessment = "GOOD"
            elif result['r2'] > 0.5:
                individual_assessment = "FAIR"
            else:
                individual_assessment = "POOR"
                
            content.append(f"    Assessment: {individual_assessment}")
            content.append("")
        
        # Recommendations
        content.append("RECOMMENDATIONS")
        content.append("-" * 30)
        
        recommendations = [
            f"🥇 Primary recommendation: Use '{self.app.scenarios[best_scenario].name}' for best results",
            "📊 Consider ensemble methods combining multiple models",
            "🔄 Implement regular model retraining pipeline",
            "⚡ Optimize inference speed for production deployment",
            "🧪 Conduct A/B testing before full deployment",
            "📈 Monitor model performance drift over time"
        ]
        
        for rec in recommendations:
            content.append(f"  {rec}")
        
        content.append("")
        content.append("TECHNICAL SPECIFICATIONS")
        content.append("-" * 30)
        content.append("  Framework: Custom Neural Engine")
        content.append("  Training Algorithm: Adam Optimizer")
        content.append("  Loss Function: Mean Squared Error")
        content.append("  Activation Functions: ReLU, Swish, Linear")
        content.append("  Data Preprocessing: Standard Scaling")
        content.append("  Validation: Train/Val/Test Split (70/15/15)")
        
        return '\n'.join(content)
        
    def export_comparison_report(self):
        """Export comparison report"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    # Export as JSON
                    export_data = {
                        'comparison_date': datetime.now().isoformat(),
                        'models_compared': len(self.app.results),
                        'results': {}
                    }
                    
                    for scenario_key, result in self.app.results.items():
                        export_data['results'][scenario_key] = {
                            'scenario_info': {
                                'name': self.app.scenarios[scenario_key].name,
                                'description': self.app.scenarios[scenario_key].description,
                                'input_features': self.app.scenarios[scenario_key].input_features,
                                'target_features': self.app.scenarios[scenario_key].target_features
                            },
                            'performance': {
                                'r2': float(result['r2']),
                                'mse': float(result['mse']),
                                'mae': float(result['mae']),
                                'rmse': float(result['rmse']),
                                'accuracy_10pct': float(result['accuracy_10pct'])
                            }
                        }
                    
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                else:
                    # Export as text
                    report_content = self.generate_detailed_benchmark_report()
                    with open(filename, 'w') as f:
                        f.write(report_content)
                
                messagebox.showinfo("Success", f"Report exported to: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
                
    def export_benchmark_report(self, content):
        """Export benchmark report to file"""
        from tkinter import filedialog
        
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
                
    def clear_comparison(self):
        """Clear comparison results"""
        self.fig.clear()
        self.canvas.draw()
        self.results_text.delete(1.0, tk.END)
        
    def refresh(self):
        """Refresh tab after data or model changes"""
        self.clear_comparison()
        
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status (interface for main app)"""
        pass
