"""
Visualization tab component for dataset analysis and plotting.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import seaborn as sns
from typing import Optional, Dict, Any
import pandas as pd


class VisualizationTab:
    """GUI tab for dataset visualization and analysis."""
    
    def __init__(self, notebook: ttk.Notebook, main_app):
        self.main_app = main_app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="📊 Visualization")
        
        # Matplotlib components
        self.fig: Optional[plt.Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.ax1: Optional[plt.Axes] = None
        self.ax2: Optional[plt.Axes] = None
        
        # Plot configuration
        self.plot_config = {
            'style': 'seaborn-v0_8',
            'figure_size': (14, 10),
            'dpi': 100,
            'font_size': 10
        }
        
        self.setup_ui()
        self.setup_plotting_style()
        
    def setup_ui(self):
        """Setup the visualization tab interface."""
        self.create_control_panel()
        self.create_visualization_area()
        
    def create_control_panel(self):
        """Create visualization control panel."""
        control_frame = ttk.LabelFrame(self.frame, text="Visualization Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Create control sections
        self.create_plot_selection(control_frame)
        self.create_plot_options(control_frame)
        self.create_action_buttons(control_frame)
        
    def create_plot_selection(self, parent):
        """Create plot type selection controls."""
        selection_frame = ttk.LabelFrame(parent, text="Plot Types")
        selection_frame.pack(fill='x', padx=5, pady=5)
        
        # Plot selection variables
        self.show_distributions = tk.BooleanVar(value=True)
        self.show_correlations = tk.BooleanVar(value=True)
        self.show_statistics = tk.BooleanVar(value=False)
        self.show_quality_metrics = tk.BooleanVar(value=False)
        
        # Plot type checkboxes
        plot_grid = ttk.Frame(selection_frame)
        plot_grid.pack(fill='x', padx=10, pady=5)
        
        ttk.Checkbutton(plot_grid, text="📈 Coefficient & Solution Distributions", 
                       variable=self.show_distributions).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        ttk.Checkbutton(plot_grid, text="🔗 Correlation Analysis", 
                       variable=self.show_correlations).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Checkbutton(plot_grid, text="📊 Statistical Summary", 
                       variable=self.show_statistics).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        ttk.Checkbutton(plot_grid, text="✅ Data Quality Metrics", 
                       variable=self.show_quality_metrics).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
    def create_plot_options(self, parent):
        """Create plot customization options."""
        options_frame = ttk.LabelFrame(parent, text="Plot Options")
        options_frame.pack(fill='x', padx=5, pady=5)
        
        options_grid = ttk.Frame(options_frame)
        options_grid.pack(fill='x', padx=10, pady=5)
        
        # Bin count for histograms
        ttk.Label(options_grid, text="Histogram bins:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.bins_var = tk.IntVar(value=30)
        bins_spinbox = ttk.Spinbox(options_grid, from_=10, to=100, textvariable=self.bins_var, width=10)
        bins_spinbox.grid(row=0, column=1, padx=5, pady=2)
        
        # Color scheme selection
        ttk.Label(options_grid, text="Color scheme:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.color_scheme_var = tk.StringVar(value="Set1")
        color_combo = ttk.Combobox(options_grid, textvariable=self.color_scheme_var, 
                                  values=["Set1", "Set2", "viridis", "plasma", "coolwarm"], width=12)
        color_combo.grid(row=0, column=3, padx=5, pady=2)
        
        # Alpha transparency
        ttk.Label(options_grid, text="Transparency:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.alpha_var = tk.DoubleVar(value=0.7)
        alpha_scale = ttk.Scale(options_grid, from_=0.1, to=1.0, variable=self.alpha_var, 
                               orient='horizontal', length=100)
        alpha_scale.grid(row=1, column=1, padx=5, pady=2)
        
        # Show grid
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_grid, text="Show grid", 
                       variable=self.show_grid_var).grid(row=1, column=2, sticky='w', padx=5, pady=2)
        
        # Show legends
        self.show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_grid, text="Show legends", 
                       variable=self.show_legend_var).grid(row=1, column=3, sticky='w', padx=5, pady=2)
        
    def create_action_buttons(self, parent):
        """Create action buttons for plot operations."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        # Primary actions
        ttk.Button(button_frame, text="🔄 Update Plots", 
                  command=self.update_plots).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="📊 Quick Analysis", 
                  command=self.quick_analysis).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🔍 Detailed Analysis", 
                  command=self.detailed_analysis).pack(side='left', padx=5)
        
        # Export and utility actions
        ttk.Button(button_frame, text="💾 Export Plot", 
                  command=self.export_plot).pack(side='left', padx=20)
        
        ttk.Button(button_frame, text="📋 Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🗑 Clear Plots", 
                  command=self.clear_plots).pack(side='left', padx=5)
        
    def create_visualization_area(self):
        """Create the main visualization area with matplotlib."""
        viz_frame = ttk.LabelFrame(self.frame, text="Dataset Analysis")
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Setup matplotlib figure
        try:
            plt.style.use(self.plot_config['style'])
        except:
            pass  # Use default style if seaborn not available
            
        self.fig = plt.Figure(figsize=self.plot_config['figure_size'], 
                             dpi=self.plot_config['dpi'])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add navigation toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill='x', padx=5, pady=5)
        
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            self.toolbar.update()
        except:
            # Fallback if toolbar not available
            ttk.Label(toolbar_frame, text="Matplotlib navigation not available").pack()
            
    def setup_plotting_style(self):
        """Configure matplotlib plotting style."""
        plt.rcParams.update({
            'font.size': self.plot_config['font_size'],
            'axes.titlesize': self.plot_config['font_size'] + 2,
            'axes.labelsize': self.plot_config['font_size'],
            'xtick.labelsize': self.plot_config['font_size'] - 1,
            'ytick.labelsize': self.plot_config['font_size'] - 1,
            'legend.fontsize': self.plot_config['font_size'] - 1,
            'figure.titlesize': self.plot_config['font_size'] + 4
        })
        
    def update_plots(self):
        """Update all plots based on current data and settings."""
        if not self.has_data():
            messagebox.showwarning("Warning", "No data available for plotting!")
            return
            
        # Clear previous plots
        self.fig.clear()
        
        # Get current dataset
        df = self.main_app.get_dataset_dataframe()
        
        # Create subplot layout based on selected plots
        selected_plots = self.get_selected_plots()
        
        if not selected_plots:
            messagebox.showwarning("Warning", "Please select at least one plot type!")
            return
            
        # Create appropriate subplot layout
        n_plots = len(selected_plots)
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 1, 2
        elif n_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
            
        # Generate selected plots
        plot_idx = 1
        
        if 'distributions' in selected_plots:
            ax = self.fig.add_subplot(rows, cols, plot_idx)
            self.plot_distributions(ax, df)
            plot_idx += 1
            
        if 'correlations' in selected_plots:
            ax = self.fig.add_subplot(rows, cols, plot_idx)
            self.plot_correlations(ax, df)
            plot_idx += 1
            
        if 'statistics' in selected_plots:
            ax = self.fig.add_subplot(rows, cols, plot_idx)
            self.plot_statistics(ax, df)
            plot_idx += 1
            
        if 'quality' in selected_plots:
            ax = self.fig.add_subplot(rows, cols, plot_idx)
            self.plot_quality_metrics(ax, df)
            plot_idx += 1
            
        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()
        
    def get_selected_plots(self) -> list:
        """Get list of selected plot types."""
        selected = []
        
        if self.show_distributions.get():
            selected.append('distributions')
        if self.show_correlations.get():
            selected.append('correlations')
        if self.show_statistics.get():
            selected.append('statistics')
        if self.show_quality_metrics.get():
            selected.append('quality')
            
        return selected
        
    def plot_distributions(self, ax, df: pd.DataFrame):
        """Plot coefficient and solution distributions."""
        try:
            # Plot histograms for all variables
            variables = ['a', 'b', 'c', 'x1', 'x2']
            colors = plt.cm.get_cmap(self.color_scheme_var.get())(np.linspace(0, 1, len(variables)))
            
            for i, var in enumerate(variables):
                ax.hist(df[var], bins=self.bins_var.get(), alpha=self.alpha_var.get(), 
                       label=var, color=colors[i])
                
            ax.set_title('Distribution of Coefficients and Solutions')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            if self.show_legend_var.get():
                ax.legend()
            if self.show_grid_var.get():
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting distributions:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            
    def plot_correlations(self, ax, df: pd.DataFrame):
        """Plot correlation matrix heatmap."""
        try:
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            im = ax.imshow(corr_matrix.values, cmap=self.color_scheme_var.get(), 
                          aspect='auto', vmin=-1, vmax=1)
            
            # Set labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns)
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title('Correlation Matrix')
            
            # Add colorbar
            cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation Coefficient')
            
            if self.show_grid_var.get():
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting correlations:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            
    def plot_statistics(self, ax, df: pd.DataFrame):
        """Plot statistical summary box plots."""
        try:
            # Create box plots for all variables
            variables = ['a', 'b', 'c', 'x1', 'x2']
            data_to_plot = [df[var] for var in variables]
            
            bp = ax.boxplot(data_to_plot, labels=variables, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.get_cmap(self.color_scheme_var.get())(np.linspace(0, 1, len(variables)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(self.alpha_var.get())
                
            ax.set_title('Statistical Summary (Box Plots)')
            ax.set_xlabel('Variables')
            ax.set_ylabel('Value')
            
            if self.show_grid_var.get():
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting statistics:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            
    def plot_quality_metrics(self, ax, df: pd.DataFrame):
        """Plot data quality metrics."""
        try:
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(df)
            
            # Create bar plot
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            colors = plt.cm.get_cmap(self.color_scheme_var.get())(np.linspace(0, 1, len(metric_names)))
            bars = ax.bar(metric_names, metric_values, color=colors, alpha=self.alpha_var.get())
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}%' if value < 10 else f'{value:.0f}%',
                       ha='center', va='bottom')
                       
            ax.set_title('Data Quality Metrics')
            ax.set_ylabel('Percentage (%)')
            ax.tick_params(axis='x', rotation=45)
            
            if self.show_grid_var.get():
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting quality metrics:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            
    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics."""
        metrics = {}
        
        try:
            total_equations = len(df)
            
            # Whole number solutions
            x1_whole = np.sum(np.abs(df['x1'] - np.round(df['x1'])) < 1e-6)
            x2_whole = np.sum(np.abs(df['x2'] - np.round(df['x2'])) < 1e-6)
            
            metrics['X1 Whole'] = (x1_whole / total_equations) * 100
            metrics['X2 Whole'] = (x2_whole / total_equations) * 100
            metrics['Any Whole'] = (np.sum((np.abs(df['x1'] - np.round(df['x1'])) < 1e-6) | 
                                          (np.abs(df['x2'] - np.round(df['x2'])) < 1e-6)) / total_equations) * 100
            
            # Integer coefficients
            a_int = np.sum(np.abs(df['a'] - np.round(df['a'])) < 1e-6)
            b_int = np.sum(np.abs(df['b'] - np.round(df['b'])) < 1e-6)
            c_int = np.sum(np.abs(df['c'] - np.round(df['c'])) < 1e-6)
            
            metrics['Int Coeff'] = (min(a_int, b_int, c_int) / total_equations) * 100
            
            # Solution verification
            verified_count = 0
            for _, row in df.iterrows():
                a, b, c, x1, x2 = row['a'], row['b'], row['c'], row['x1'], row['x2']
                error1 = abs(a * x1**2 + b * x1 + c)
                error2 = abs(a * x2**2 + b * x2 + c)
                if error1 < 1e-6 and error2 < 1e-6:
                    verified_count += 1
                    
            metrics['Verified'] = (verified_count / total_equations) * 100
            
        except Exception:
            metrics = {'Error': 0}
            
        return metrics
        
    def quick_analysis(self):
        """Perform quick analysis with default settings."""
        # Set all plot types to true for comprehensive view
        self.show_distributions.set(True)
        self.show_correlations.set(True)
        self.show_statistics.set(False)
        self.show_quality_metrics.set(True)
        
        # Update plots
        self.update_plots()
        
    def detailed_analysis(self):
        """Perform detailed analysis with all plots."""
        # Enable all plot types
        self.show_distributions.set(True)
        self.show_correlations.set(True)
        self.show_statistics.set(True)
        self.show_quality_metrics.set(True)
        
        # Update plots
        self.update_plots()
        
        # Show detailed statistics
        self.show_detailed_statistics()
        
    def show_detailed_statistics(self):
        """Show detailed statistics in a popup window."""
        if not self.has_data():
            return
            
        stats_window = tk.Toplevel(self.main_app.root)
        stats_window.title("Detailed Statistics")
        stats_window.geometry("700x500")
        stats_window.transient(self.main_app.root)
        
        # Create text widget
        text_frame = ttk.Frame(stats_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap='word', font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Generate detailed statistics
        stats_content = self.generate_detailed_statistics()
        text_widget.insert('1.0', stats_content)
        text_widget.config(state='disabled')
        
    def generate_detailed_statistics(self) -> str:
        """Generate detailed statistics text."""
        if not self.has_data():
            return "No data available."
            
        df = self.main_app.get_dataset_dataframe()
        
        stats = []
        stats.append("📊 DETAILED DATASET STATISTICS")
        stats.append("=" * 50)
        stats.append(f"Dataset size: {len(df)} equations")
        stats.append("")
        
        # Basic statistics for each variable
        for var in ['a', 'b', 'c', 'x1', 'x2']:
            data = df[var]
            stats.append(f"{var.upper()} STATISTICS:")
            stats.append(f"  Mean: {data.mean():.6f}")
            stats.append(f"  Std:  {data.std():.6f}")
            stats.append(f"  Min:  {data.min():.6f}")
            stats.append(f"  Max:  {data.max():.6f}")
            stats.append(f"  Q1:   {data.quantile(0.25):.6f}")
            stats.append(f"  Q2:   {data.quantile(0.5):.6f}")
            stats.append(f"  Q3:   {data.quantile(0.75):.6f}")
            stats.append("")
            
        # Quality metrics
        quality_metrics = self.calculate_quality_metrics(df)
        stats.append("🔍 DATA QUALITY METRICS:")
        for metric, value in quality_metrics.items():
            stats.append(f"  {metric}: {value:.2f}%")
        stats.append("")
        
        # Correlation insights
        corr_matrix = df.corr()
        stats.append("🔗 CORRELATION INSIGHTS:")
        
        # Find highest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                correlations.append((abs(corr_val), corr_val, var1, var2))
                
        correlations.sort(reverse=True)
        
        for _, corr_val, var1, var2 in correlations[:5]:
            stats.append(f"  {var1} vs {var2}: {corr_val:.3f}")
            
        return '\n'.join(stats)
        
    def export_plot(self):
        """Export current plot to file."""
        if self.fig is None:
            messagebox.showwarning("Warning", "No plot to export!")
            return
            
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("EPS files", "*.eps"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Plot exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
                
    def generate_report(self):
        """Generate comprehensive analysis report."""
        if not self.has_data():
            messagebox.showwarning("Warning", "No data available for report!")
            return
            
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                report_content = self.generate_analysis_report()
                with open(filename, 'w') as f:
                    f.write(report_content)
                messagebox.showinfo("Success", f"Report generated: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
                
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report text."""
        from datetime import datetime
        
        report = []
        report.append("QUADRATIC EQUATION DATASET ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Location: Varna, Bulgaria")
        report.append("")
        
        # Dataset overview
        summary = self.main_app.get_dataset_summary()
        report.append("DATASET OVERVIEW")
        report.append("-" * 30)
        report.append(f"Total equations: {summary.get('total_equations', 0)}")
        
        if 'generation_stats' in summary:
            gen_stats = summary['generation_stats']
            report.append(f"Whole solutions: {gen_stats.get('whole_solutions', 0)}")
            report.append(f"Real solutions: {gen_stats.get('real_solutions', 0)}")
            report.append(f"Rejected: {gen_stats.get('rejected', 0)}")
        report.append("")
        
        # Detailed statistics
        detailed_stats = self.generate_detailed_statistics()
        report.append(detailed_stats)
        
        # Recommendations
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        df = self.main_app.get_dataset_dataframe()
        quality_metrics = self.calculate_quality_metrics(df)
        
        if quality_metrics.get('Verified', 0) > 95:
            report.append("✅ Excellent data quality - all solutions verified")
        elif quality_metrics.get('Verified', 0) > 90:
            report.append("✅ Good data quality - most solutions verified")
        else:
            report.append("⚠️ Some verification issues detected")
            
        if quality_metrics.get('Any Whole', 0) > 50:
            report.append("✅ Good distribution of whole number solutions")
        else:
            report.append("ℹ️ Consider increasing whole number solution ratio")
            
        return '\n'.join(report)
        
    def clear_plots(self):
        """Clear all plots."""
        if self.fig:
            self.fig.clear()
            self.canvas.draw()
            
    def has_data(self) -> bool:
        """Check if dataset has data available."""
        return (hasattr(self.main_app, 'generator') and 
                self.main_app.generator.data and 
                len(self.main_app.generator.data) > 0)
