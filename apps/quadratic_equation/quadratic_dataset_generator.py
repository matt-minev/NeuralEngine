"""
Quadratic Equation Dataset Generator

This tool generates datasets of quadratic equations in the format:
[a, b, c, x1, x2] where ax² + bx + c = 0 has solutions x1 and x2

Features:
- Interactive GUI with real-time generation visualization
- Configurable dataset size and parameters
- Priority options (whole number solutions, real solutions, etc.)
- Manual data editor with validation
- Data paste functionality
- CSV export with custom naming
- Integration with Neural Engine data format
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import json
from typing import List, Tuple, Optional, Dict, Any
import os
from dataclasses import dataclass, asdict

@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    dataset_size: int = 1000
    dataset_name: str = "quadratic_dataset"
    infinite_mode: bool = False  # NEW: Infinite generation mode
    
    # Coefficient ranges
    a_min: float = -10.0
    a_max: float = 10.0
    b_min: float = -10.0
    b_max: float = 10.0
    c_min: float = -10.0
    c_max: float = 10.0
    
    # Generation preferences
    prioritize_whole_solutions: bool = True
    whole_solution_ratio: float = 0.6
    only_whole_solutions: bool = False
    allow_complex_solutions: bool = False
    force_real_solutions: bool = True
    systematic_generation: bool = False  # NEW: Systematic textbook-style generation
    
    # Solution constraints
    solution_min: float = -100.0
    solution_max: float = 100.0
    
    # Advanced options
    avoid_zero_a: bool = True
    integer_coefficients_only: bool = False
    symmetric_solutions: bool = False
    
class QuadraticSolver:
    """Utilities for solving quadratic equations"""
    
    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Solve ax² + bx + c = 0
        Returns (x1, x2) or (None, None) if no real solutions
        """
        if a == 0:
            # Linear equation: bx + c = 0
            if b == 0:
                return None, None
            return -c/b, -c/b
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None, None  # No real solutions
        
        sqrt_discriminant = np.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        
        return x1, x2
    
    @staticmethod
    def is_whole_number(x: float, tolerance: float = 1e-10) -> bool:
        """Check if a number is effectively a whole number"""
        return abs(x - round(x)) < tolerance
    
    @staticmethod
    def verify_solution(a: float, b: float, c: float, x: float, tolerance: float = 1e-10) -> bool:
        """Verify if x is a solution to ax² + bx + c = 0"""
        result = a * x**2 + b * x + c
        return abs(result) < tolerance

class DatasetGenerator:
    """Core dataset generation logic"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.data = []
        self.generation_stats = {
            'total_generated': 0,
            'whole_solutions': 0,
            'real_solutions': 0,
            'complex_solutions': 0,
            'rejected': 0
        }
        # NEW: Systematic generation state
        self.systematic_state = {
            'a_current': int(self.config.a_min) if self.config.integer_coefficients_only else self.config.a_min,
            'b_current': int(self.config.b_min) if self.config.integer_coefficients_only else self.config.b_min,
            'c_current': int(self.config.c_min) if self.config.integer_coefficients_only else self.config.c_min,
            'completed_cycle': False
        }
    
    def generate_coefficients(self) -> Tuple[float, float, float]:
        """Generate random coefficients based on configuration"""
        if self.config.systematic_generation:
            return self.generate_systematic_coefficients()
        
        if self.config.integer_coefficients_only:
            # Ensure proper integer generation
            a = float(np.random.randint(int(self.config.a_min), int(self.config.a_max) + 1))
            b = float(np.random.randint(int(self.config.b_min), int(self.config.b_max) + 1))
            c = float(np.random.randint(int(self.config.c_min), int(self.config.c_max) + 1))
        else:
            a = np.random.uniform(self.config.a_min, self.config.a_max)
            b = np.random.uniform(self.config.b_min, self.config.b_max)
            c = np.random.uniform(self.config.c_min, self.config.c_max)
        
        if self.config.avoid_zero_a and a == 0:
            a = 1.0 if np.random.random() < 0.5 else -1.0
        
        return a, b, c
    
    def generate_systematic_coefficients(self) -> Tuple[float, float, float]:
        """Generate coefficients systematically for textbook coverage"""
        # Get current values
        a = self.systematic_state['a_current']
        b = self.systematic_state['b_current'] 
        c = self.systematic_state['c_current']
        
        # Advance to next combination
        if self.config.integer_coefficients_only:
            # Integer systematic generation
            c += 1
            if c > int(self.config.c_max):
                c = int(self.config.c_min)
                b += 1
                if b > int(self.config.b_max):
                    b = int(self.config.b_min)
                    a += 1
                    if a > int(self.config.a_max):
                        a = int(self.config.a_min)
                        self.systematic_state['completed_cycle'] = True
        else:
            # For non-integer, use smaller steps
            step = 0.5
            c += step
            if c > self.config.c_max:
                c = self.config.c_min
                b += step
                if b > self.config.b_max:
                    b = self.config.b_min
                    a += step
                    if a > self.config.a_max:
                        a = self.config.a_min
                        self.systematic_state['completed_cycle'] = True
        
        # Update state
        self.systematic_state['a_current'] = a
        self.systematic_state['b_current'] = b
        self.systematic_state['c_current'] = c
        
        # Handle zero 'a' if needed
        if self.config.avoid_zero_a and a == 0:
            a = 1.0
        
        return float(a), float(b), float(c)

    def generate_whole_solution_equation(self) -> Tuple[float, float, float, float, float]:
        """Generate equation with at least one whole number solution"""
        # Pick whole number solutions first
        x1 = np.random.randint(int(self.config.solution_min), int(self.config.solution_max) + 1)
        
        if self.config.symmetric_solutions:
            x2 = -x1
        else:
            x2 = np.random.randint(int(self.config.solution_min), int(self.config.solution_max) + 1)
        
        # Generate coefficients from solutions: a(x-x1)(x-x2) = ax² - a(x1+x2)x + ax1x2
        # FIX: Respect integer_coefficients_only setting
        if self.config.integer_coefficients_only:
            a = float(np.random.randint(int(self.config.a_min), int(self.config.a_max) + 1))
        else:
            a = np.random.uniform(self.config.a_min, self.config.a_max)
            
        if self.config.avoid_zero_a and a == 0:
            a = 1.0
        
        b = -a * (x1 + x2)
        c = a * x1 * x2
        
        # If integer coefficients required, ensure b and c are also integers
        if self.config.integer_coefficients_only:
            b = float(round(b))
            c = float(round(c))
        
        return a, b, c, x1, x2

    def generate_single_equation(self) -> Optional[Tuple[float, float, float, float, float]]:
        """Generate a single quadratic equation with solutions"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Decide generation method
            if (self.config.prioritize_whole_solutions and 
                np.random.random() < self.config.whole_solution_ratio):
                try:
                    result = self.generate_whole_solution_equation()
                    if result:
                        a, b, c, x1, x2 = result
                        
                        # Check if we need to validate the result
                        if self.config.only_whole_solutions:
                            if not (QuadraticSolver.is_whole_number(x1) or QuadraticSolver.is_whole_number(x2)):
                                self.generation_stats['rejected'] += 1
                                continue
                        
                        return result
                except:
                    continue
            
            # Regular generation
            a, b, c = self.generate_coefficients()
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            if x1 is None or x2 is None:
                if self.config.allow_complex_solutions:
                    # Handle complex solutions (placeholder)
                    continue
                else:
                    self.generation_stats['rejected'] += 1
                    continue
            
            # Check solution bounds
            if (x1 < self.config.solution_min or x1 > self.config.solution_max or
                x2 < self.config.solution_min or x2 > self.config.solution_max):
                self.generation_stats['rejected'] += 1
                continue
            
            # Check for whole number solutions if required
            if self.config.only_whole_solutions:
                if not (QuadraticSolver.is_whole_number(x1) or QuadraticSolver.is_whole_number(x2)):
                    self.generation_stats['rejected'] += 1
                    continue
            
            return a, b, c, x1, x2
        
        return None
    
    def generate_dataset(self, progress_callback=None, stop_event=None):
        """Generate complete dataset with progress tracking"""
        self.data = []
        self.generation_stats = {
            'total_generated': 0,
            'whole_solutions': 0,
            'real_solutions': 0,
            'complex_solutions': 0,
            'rejected': 0
        }
        
        # NEW: Handle infinite mode
        if self.config.infinite_mode:
            count = 0
            while not (stop_event and stop_event.is_set()):
                equation = self.generate_single_equation()
                if equation:
                    a, b, c, x1, x2 = equation
                    self.data.append([a, b, c, x1, x2])
                    
                    # Update statistics
                    self.generation_stats['total_generated'] += 1
                    if (QuadraticSolver.is_whole_number(x1) or 
                        QuadraticSolver.is_whole_number(x2)):
                        self.generation_stats['whole_solutions'] += 1
                    self.generation_stats['real_solutions'] += 1
                    
                    count += 1
                    if progress_callback and count % 10 == 0:
                        progress_callback(count, float('inf'), self.generation_stats)
                
                # Check if systematic generation completed a cycle
                if (self.config.systematic_generation and 
                    self.systematic_state['completed_cycle']):
                    break
        else:
            # Original finite generation
            for i in range(self.config.dataset_size):
                if stop_event and stop_event.is_set():
                    break
                
                equation = self.generate_single_equation()
                if equation:
                    a, b, c, x1, x2 = equation
                    self.data.append([a, b, c, x1, x2])
                    
                    # Update statistics
                    self.generation_stats['total_generated'] += 1
                    if (QuadraticSolver.is_whole_number(x1) or 
                        QuadraticSolver.is_whole_number(x2)):
                        self.generation_stats['whole_solutions'] += 1
                    self.generation_stats['real_solutions'] += 1
                    
                    if progress_callback:
                        progress_callback(i + 1, self.config.dataset_size, self.generation_stats)
        
        return self.data
    
    def save_dataset(self, filepath: str):
        """Save dataset to CSV file"""
        df = pd.DataFrame(self.data, columns=['a', 'b', 'c', 'x1', 'x2'])
        df.to_csv(filepath, index=False)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get dataset as pandas DataFrame"""
        return pd.DataFrame(self.data, columns=['a', 'b', 'c', 'x1', 'x2'])

class QuadraticDatasetGUI:
    """Main GUI application for dataset generation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Quadratic Equation Dataset Generator")
        self.root.geometry("1200x800")
        
        self.config = GenerationConfig()
        self.generator = DatasetGenerator(self.config)
        self.generation_thread = None
        self.stop_event = threading.Event()
        
        self.setup_ui()
        self.update_config_display()
    
    def setup_ui(self):
        """Setup the main UI"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_generation_tab()
        self.create_editor_tab()
        self.create_visualization_tab()
        
    def create_generation_tab(self):
        """Create the dataset generation tab"""
        gen_frame = ttk.Frame(self.notebook)
        self.notebook.add(gen_frame, text="Generate Dataset")
        
        # Left panel - Configuration
        left_frame = ttk.Frame(gen_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Dataset configuration
        config_frame = ttk.LabelFrame(left_frame, text="Dataset Configuration")
        config_frame.pack(fill='x', pady=5)
        
        # Dataset name and size
        ttk.Label(config_frame, text="Dataset Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.name_var = tk.StringVar(value=self.config.dataset_name)
        ttk.Entry(config_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Dataset Size:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.size_var = tk.IntVar(value=self.config.dataset_size)
        ttk.Entry(config_frame, textvariable=self.size_var, width=30).grid(row=1, column=1, padx=5, pady=2)
        
        # Coefficient ranges
        ranges_frame = ttk.LabelFrame(left_frame, text="Coefficient Ranges")
        ranges_frame.pack(fill='x', pady=5)
        
        # A coefficient
        ttk.Label(ranges_frame, text="a range:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.a_min_var = tk.DoubleVar(value=self.config.a_min)
        self.a_max_var = tk.DoubleVar(value=self.config.a_max)
        ttk.Entry(ranges_frame, textvariable=self.a_min_var, width=8).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(ranges_frame, text="to").grid(row=0, column=2, padx=2)
        ttk.Entry(ranges_frame, textvariable=self.a_max_var, width=8).grid(row=0, column=3, padx=2, pady=2)
        
        # B coefficient
        ttk.Label(ranges_frame, text="b range:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.b_min_var = tk.DoubleVar(value=self.config.b_min)
        self.b_max_var = tk.DoubleVar(value=self.config.b_max)
        ttk.Entry(ranges_frame, textvariable=self.b_min_var, width=8).grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(ranges_frame, text="to").grid(row=1, column=2, padx=2)
        ttk.Entry(ranges_frame, textvariable=self.b_max_var, width=8).grid(row=1, column=3, padx=2, pady=2)
        
        # C coefficient
        ttk.Label(ranges_frame, text="c range:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.c_min_var = tk.DoubleVar(value=self.config.c_min)
        self.c_max_var = tk.DoubleVar(value=self.config.c_max)
        ttk.Entry(ranges_frame, textvariable=self.c_min_var, width=8).grid(row=2, column=1, padx=2, pady=2)
        ttk.Label(ranges_frame, text="to").grid(row=2, column=2, padx=2)
        ttk.Entry(ranges_frame, textvariable=self.c_max_var, width=8).grid(row=2, column=3, padx=2, pady=2)
        
        # Generation options
        options_frame = ttk.LabelFrame(left_frame, text="Generation Options")
        options_frame.pack(fill='x', pady=5)
        
        self.whole_solutions_var = tk.BooleanVar(value=self.config.prioritize_whole_solutions)
        ttk.Checkbutton(options_frame, text="Prioritize whole number solutions", 
                       variable=self.whole_solutions_var).pack(anchor='w', padx=5, pady=2)
        
        ttk.Label(options_frame, text="Whole solution ratio:").pack(anchor='w', padx=5, pady=2)
        self.whole_ratio_var = tk.DoubleVar(value=self.config.whole_solution_ratio)
        ttk.Scale(options_frame, from_=0.0, to=1.0, variable=self.whole_ratio_var, 
                 orient='horizontal').pack(fill='x', padx=5, pady=2)
        
        self.real_solutions_var = tk.BooleanVar(value=self.config.force_real_solutions)
        ttk.Checkbutton(options_frame, text="Force real solutions only", 
                       variable=self.real_solutions_var).pack(anchor='w', padx=5, pady=2)
        
        self.integer_coeff_var = tk.BooleanVar(value=self.config.integer_coefficients_only)
        ttk.Checkbutton(options_frame, text="Integer coefficients only", 
                       variable=self.integer_coeff_var).pack(anchor='w', padx=5, pady=2)
        
        self.only_whole_var = tk.BooleanVar(value=self.config.only_whole_solutions)
        ttk.Checkbutton(options_frame, text="Only accept equations with whole number solutions", 
                    variable=self.only_whole_var).pack(anchor='w', padx=5, pady=2)

        self.infinite_mode_var = tk.BooleanVar(value=self.config.infinite_mode)
        ttk.Checkbutton(options_frame, text="Infinite mode (generate until stopped)", 
                    variable=self.infinite_mode_var).pack(anchor='w', padx=5, pady=2)

        self.systematic_var = tk.BooleanVar(value=self.config.systematic_generation)
        ttk.Checkbutton(options_frame, text="Systematic generation (textbook coverage)", 
                    variable=self.systematic_var).pack(anchor='w', padx=5, pady=2)

        
        # Generation controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill='x', pady=10)
        
        ttk.Button(controls_frame, text="Generate Dataset", 
                  command=self.start_generation).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Stop Generation", 
                  command=self.stop_generation).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Save Dataset", 
                  command=self.save_dataset).pack(side='left', padx=5)
        
        # Right panel - Progress and preview
        right_frame = ttk.Frame(gen_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(right_frame, text="Generation Progress")
        progress_frame.pack(fill='x', pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        self.status_text = tk.Text(progress_frame, height=6, width=50)
        self.status_text.pack(fill='x', padx=5, pady=5)
        
        # Data preview
        preview_frame = ttk.LabelFrame(right_frame, text="Data Preview")
        preview_frame.pack(fill='both', expand=True, pady=5)
        
        # Create treeview for data preview
        columns = ('a', 'b', 'c', 'x1', 'x2')
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)
        
        self.preview_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def create_editor_tab(self):
        """Create the manual data editor tab"""
        editor_frame = ttk.Frame(self.notebook)
        self.notebook.add(editor_frame, text="Manual Editor")
        
        # Instructions
        instructions = ttk.Label(editor_frame, 
                                text="Enter quadratic equation coefficients. Solutions will be calculated automatically.\n"
                                     "Red highlighting indicates incorrect manual solutions.")
        instructions.pack(pady=10)
        
        # Editor section
        editor_section = ttk.LabelFrame(editor_frame, text="Add/Edit Equation")
        editor_section.pack(fill='x', padx=10, pady=5)
        
        # Input fields
        input_frame = ttk.Frame(editor_section)
        input_frame.pack(pady=5)
        
        ttk.Label(input_frame, text="a:").grid(row=0, column=0, padx=5, pady=2)
        self.edit_a_var = tk.DoubleVar()
        self.edit_a_entry = ttk.Entry(input_frame, textvariable=self.edit_a_var, width=10)
        self.edit_a_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="b:").grid(row=0, column=2, padx=5, pady=2)
        self.edit_b_var = tk.DoubleVar()
        self.edit_b_entry = ttk.Entry(input_frame, textvariable=self.edit_b_var, width=10)
        self.edit_b_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(input_frame, text="c:").grid(row=0, column=4, padx=5, pady=2)
        self.edit_c_var = tk.DoubleVar()
        self.edit_c_entry = ttk.Entry(input_frame, textvariable=self.edit_c_var, width=10)
        self.edit_c_entry.grid(row=0, column=5, padx=5, pady=2)
        
        # Solution fields (optional manual input)
        ttk.Label(input_frame, text="x1 (optional):").grid(row=1, column=0, padx=5, pady=2)
        self.edit_x1_var = tk.StringVar()
        self.edit_x1_entry = ttk.Entry(input_frame, textvariable=self.edit_x1_var, width=10)
        self.edit_x1_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(input_frame, text="x2 (optional):").grid(row=1, column=2, padx=5, pady=2)
        self.edit_x2_var = tk.StringVar()
        self.edit_x2_entry = ttk.Entry(input_frame, textvariable=self.edit_x2_var, width=10)
        self.edit_x2_entry.grid(row=1, column=3, padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(editor_section)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Calculate Solutions", 
                  command=self.calculate_solutions).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Add to Dataset", 
                  command=self.add_to_dataset).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Paste Data", 
                  command=self.paste_data).pack(side='left', padx=5)
        
        # Results display
        self.results_text = tk.Text(editor_section, height=3, width=80)
        self.results_text.pack(fill='x', padx=5, pady=5)
        
        # Manual dataset viewer
        viewer_frame = ttk.LabelFrame(editor_frame, text="Current Dataset")
        viewer_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for manual dataset
        columns = ('a', 'b', 'c', 'x1', 'x2')
        self.manual_tree = ttk.Treeview(viewer_frame, columns=columns, show='headings')
        
        for col in columns:
            self.manual_tree.heading(col, text=col)
            self.manual_tree.column(col, width=80)
        
        manual_scrollbar = ttk.Scrollbar(viewer_frame, orient='vertical', command=self.manual_tree.yview)
        self.manual_tree.configure(yscrollcommand=manual_scrollbar.set)
        
        self.manual_tree.pack(side='left', fill='both', expand=True)
        manual_scrollbar.pack(side='right', fill='y')
        
        # Bind validation
        self.edit_a_var.trace('w', self.validate_input)
        self.edit_b_var.trace('w', self.validate_input)
        self.edit_c_var.trace('w', self.validate_input)
        self.edit_x1_var.trace('w', self.validate_input)
        self.edit_x2_var.trace('w', self.validate_input)
    
    def create_visualization_tab(self):
        """Create the visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(viz_frame)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Button(control_frame, text="Update Plots", 
                  command=self.update_plots).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear Plots", 
                  command=self.clear_plots).pack(side='left', padx=5)
    
    def update_config_display(self):
        """Update configuration display"""
        pass
    
    def validate_input(self, *args):
        """Validate manual input and highlight errors"""
        try:
            a = self.edit_a_var.get()
            b = self.edit_b_var.get()
            c = self.edit_c_var.get()
            
            # Calculate correct solutions
            x1_correct, x2_correct = QuadraticSolver.solve_quadratic(a, b, c)
            
            # Check manual solutions if provided
            x1_manual = self.edit_x1_var.get().strip()
            x2_manual = self.edit_x2_var.get().strip()
            
            # Reset colors
            self.edit_x1_entry.configure(style='TEntry')
            self.edit_x2_entry.configure(style='TEntry')
            
            if x1_manual and x1_correct is not None:
                try:
                    x1_val = float(x1_manual)
                    if not QuadraticSolver.verify_solution(a, b, c, x1_val):
                        self.edit_x1_entry.configure(style='Error.TEntry')
                except ValueError:
                    self.edit_x1_entry.configure(style='Error.TEntry')
            
            if x2_manual and x2_correct is not None:
                try:
                    x2_val = float(x2_manual)
                    if not QuadraticSolver.verify_solution(a, b, c, x2_val):
                        self.edit_x2_entry.configure(style='Error.TEntry')
                except ValueError:
                    self.edit_x2_entry.configure(style='Error.TEntry')
                    
        except:
            pass
    
    def calculate_solutions(self):
        """Calculate and display solutions"""
        try:
            a = self.edit_a_var.get()
            b = self.edit_b_var.get()
            c = self.edit_c_var.get()
            
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            if x1 is not None and x2 is not None:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Solutions for {a}x² + {b}x + {c} = 0:\n")
                self.results_text.insert(tk.END, f"x1 = {x1:.6f}\n")
                self.results_text.insert(tk.END, f"x2 = {x2:.6f}")
                
                # Auto-fill solution fields
                self.edit_x1_var.set(f"{x1:.6f}")
                self.edit_x2_var.set(f"{x2:.6f}")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No real solutions exist for this equation.")
                
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
    
    def add_to_dataset(self):
        """Add current equation to dataset"""
        try:
            a = self.edit_a_var.get()
            b = self.edit_b_var.get()
            c = self.edit_c_var.get()
            
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            if x1 is not None and x2 is not None:
                # Add to generator data
                self.generator.data.append([a, b, c, x1, x2])
                
                # Update manual tree view
                self.manual_tree.insert('', 'end', values=(f"{a:.3f}", f"{b:.3f}", f"{c:.3f}", 
                                                          f"{x1:.3f}", f"{x2:.3f}"))
                
                # Clear input fields
                self.edit_a_var.set(0)
                self.edit_b_var.set(0)
                self.edit_c_var.set(0)
                self.edit_x1_var.set("")
                self.edit_x2_var.set("")
                
                messagebox.showinfo("Success", "Equation added to dataset!")
            else:
                messagebox.showerror("Error", "Cannot add equation with no real solutions.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add equation: {str(e)}")
    
    def paste_data(self):
        """Handle pasting data from clipboard"""
        try:
            # Get clipboard content
            clipboard_data = self.root.clipboard_get()
            
            # Parse clipboard data (expecting CSV format)
            lines = clipboard_data.strip().split('\n')
            added_count = 0
            
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
                        
                        # Calculate solutions
                        x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
                        
                        if x1 is not None and x2 is not None:
                            self.generator.data.append([a, b, c, x1, x2])
                            self.manual_tree.insert('', 'end', values=(f"{a:.3f}", f"{b:.3f}", f"{c:.3f}", 
                                                                      f"{x1:.3f}", f"{x2:.3f}"))
                            added_count += 1
                    except ValueError:
                        continue
            
            messagebox.showinfo("Success", f"Added {added_count} equations from clipboard!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to paste data: {str(e)}")
    
    def update_config(self):
        """Update configuration from UI"""
        self.config.dataset_name = self.name_var.get()
        self.config.dataset_size = self.size_var.get()
        self.config.infinite_mode = self.infinite_mode_var.get()  # NEW
        self.config.systematic_generation = self.systematic_var.get()  # NEW
        self.config.a_min = self.a_min_var.get()
        self.config.a_max = self.a_max_var.get()
        self.config.b_min = self.b_min_var.get()
        self.config.b_max = self.b_max_var.get()
        self.config.c_min = self.c_min_var.get()
        self.config.c_max = self.c_max_var.get()
        self.config.prioritize_whole_solutions = self.whole_solutions_var.get()
        self.config.whole_solution_ratio = self.whole_ratio_var.get()
        self.config.only_whole_solutions = self.only_whole_var.get()
        self.config.force_real_solutions = self.real_solutions_var.get()
        self.config.integer_coefficients_only = self.integer_coeff_var.get()
    
    def progress_callback(self, current, total, stats):
        """Handle progress updates"""
        if self.config.infinite_mode:
            # For infinite mode, show different progress info
            self.progress_var.set(0)  # No progress bar for infinite
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, f"Infinite generation: {current} equations\n")
            self.status_text.insert(tk.END, f"Generated: {stats['total_generated']}\n")
            self.status_text.insert(tk.END, f"Whole solutions: {stats['whole_solutions']}\n")
            self.status_text.insert(tk.END, f"Real solutions: {stats['real_solutions']}\n")
            self.status_text.insert(tk.END, f"Rejected: {stats['rejected']}")
        else:
            # Original progress display
            progress = (current / total) * 100
            self.progress_var.set(progress)
            
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, f"Generating equation {current}/{total} ({progress:.1f}%)\n")
            self.status_text.insert(tk.END, f"Generated: {stats['total_generated']}\n")
            self.status_text.insert(tk.END, f"Whole solutions: {stats['whole_solutions']}\n")
            self.status_text.insert(tk.END, f"Real solutions: {stats['real_solutions']}\n")
            self.status_text.insert(tk.END, f"Rejected: {stats['rejected']}")
        
        # Update preview every 10 equations
        if current % 10 == 0:
            self.update_preview()
        
        self.root.update_idletasks()
    
    def update_preview(self):
        """Update the data preview to show only final dataset"""
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        # Show only the final dataset (not live generation)
        if hasattr(self, 'generation_thread') and self.generation_thread and self.generation_thread.is_alive():
            # During generation, show last 50 items
            data_to_show = self.generator.data[-50:] if len(self.generator.data) > 50 else self.generator.data
        else:
            # After generation, show all data (or last 200 for performance)
            data_to_show = self.generator.data[-200:] if len(self.generator.data) > 200 else self.generator.data
        
        for row in data_to_show:
            self.preview_tree.insert('', 'end', values=[f"{val:.3f}" for val in row])

    def start_generation(self):
        """Start dataset generation in a separate thread"""
        if self.generation_thread and self.generation_thread.is_alive():
            messagebox.showwarning("Warning", "Generation already in progress!")
            return
        
        self.update_config()
        self.generator = DatasetGenerator(self.config)
        self.stop_event.clear()
        
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self.generator.generate_dataset,
            args=(self.progress_callback, self.stop_event)
        )
        self.generation_thread.start()
        
        # Monitor thread completion
        self.root.after(100, self.check_generation_complete)
    
    def check_generation_complete(self):
        """Check if generation is complete"""
        if self.generation_thread and self.generation_thread.is_alive():
            self.root.after(100, self.check_generation_complete)
        else:
            self.update_preview()
            self.update_plots()
            messagebox.showinfo("Complete", "Dataset generation completed!")
    
    def stop_generation(self):
        """Stop dataset generation"""
        self.stop_event.set()
        messagebox.showinfo("Stopped", "Generation stopped by user.")
    
    def save_dataset(self):
        """Save the generated dataset"""
        if not self.generator.data:
            messagebox.showwarning("Warning", "No data to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.config.dataset_name}.csv"  # CHANGED from initialvalue
        )
        
        if filename:
            try:
                self.generator.save_dataset(filename)
                messagebox.showinfo("Success", f"Dataset saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")

    
    def update_plots(self):
        """Update visualization plots"""
        if not self.generator.data:
            return
        
        df = self.generator.get_dataframe()
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Coefficient distributions
        self.ax1.hist([df['a'], df['b'], df['c']], bins=30, alpha=0.7, label=['a', 'b', 'c'])
        self.ax1.set_title('Coefficient Distributions')
        self.ax1.set_xlabel('Value')
        self.ax1.set_ylabel('Frequency')
        self.ax1.legend()
        
        # Plot 2: Solution distributions
        self.ax2.hist([df['x1'], df['x2']], bins=30, alpha=0.7, label=['x1', 'x2'])
        self.ax2.set_title('Solution Distributions')
        self.ax2.set_xlabel('Value')
        self.ax2.set_ylabel('Frequency')
        self.ax2.legend()
        
        self.canvas.draw()
    
    def clear_plots(self):
        """Clear visualization plots"""
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Configure style for error highlighting
    style = ttk.Style()
    style.configure('Error.TEntry', fieldbackground='#ffcccc')
    
    app = QuadraticDatasetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()