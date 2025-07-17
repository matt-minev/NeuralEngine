"""
Generation tab component for the quadratic equation dataset generator.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any


class GenerationTab:
    """GUI tab for dataset generation controls and progress monitoring."""
    
    def __init__(self, notebook: ttk.Notebook, main_app):
        self.main_app = main_app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Generate Dataset")
        
        # UI variables
        self.name_var: Optional[tk.StringVar] = None
        self.size_var: Optional[tk.IntVar] = None
        self.a_min_var: Optional[tk.DoubleVar] = None
        self.a_max_var: Optional[tk.DoubleVar] = None
        self.b_min_var: Optional[tk.DoubleVar] = None
        self.b_max_var: Optional[tk.DoubleVar] = None
        self.c_min_var: Optional[tk.DoubleVar] = None
        self.c_max_var: Optional[tk.DoubleVar] = None
        self.whole_solutions_var: Optional[tk.BooleanVar] = None
        self.whole_ratio_var: Optional[tk.DoubleVar] = None
        self.real_solutions_var: Optional[tk.BooleanVar] = None
        self.integer_coeff_var: Optional[tk.BooleanVar] = None
        self.only_whole_var: Optional[tk.BooleanVar] = None
        self.infinite_mode_var: Optional[tk.BooleanVar] = None
        self.systematic_var: Optional[tk.BooleanVar] = None
        
        # UI components
        self.progress_var: Optional[tk.DoubleVar] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.status_text: Optional[tk.Text] = None
        self.preview_tree: Optional[ttk.Treeview] = None
        self.generate_button: Optional[ttk.Button] = None
        self.stop_button: Optional[ttk.Button] = None
        
        self.setup_ui()
        self.load_config_to_ui()
        
    def setup_ui(self):
        """Setup the generation tab interface."""
        self.create_left_panel()
        self.create_right_panel()
        
    def create_left_panel(self):
        """Create the left panel with configuration controls."""
        left_frame = ttk.Frame(self.frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        self.create_dataset_config_section(left_frame)
        self.create_coefficient_ranges_section(left_frame)
        self.create_generation_options_section(left_frame)
        self.create_generation_controls_section(left_frame)
        
    def create_dataset_config_section(self, parent):
        """Create dataset configuration section."""
        config_frame = ttk.LabelFrame(parent, text="Dataset Configuration")
        config_frame.pack(fill='x', pady=5)
        
        # Dataset name
        ttk.Label(config_frame, text="Dataset Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.name_var = tk.StringVar(value=self.main_app.config.dataset_name)
        ttk.Entry(config_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        # Dataset size
        ttk.Label(config_frame, text="Dataset Size:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.size_var = tk.IntVar(value=self.main_app.config.dataset_size)
        ttk.Entry(config_frame, textvariable=self.size_var, width=30).grid(row=1, column=1, padx=5, pady=2)
        
    def create_coefficient_ranges_section(self, parent):
        """Create coefficient ranges configuration section."""
        ranges_frame = ttk.LabelFrame(parent, text="Coefficient Ranges")
        ranges_frame.pack(fill='x', pady=5)
        
        # A coefficient range
        ttk.Label(ranges_frame, text="a range:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.a_min_var = tk.DoubleVar(value=self.main_app.config.a_min)
        self.a_max_var = tk.DoubleVar(value=self.main_app.config.a_max)
        ttk.Entry(ranges_frame, textvariable=self.a_min_var, width=8).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(ranges_frame, text="to").grid(row=0, column=2, padx=2)
        ttk.Entry(ranges_frame, textvariable=self.a_max_var, width=8).grid(row=0, column=3, padx=2, pady=2)
        
        # B coefficient range
        ttk.Label(ranges_frame, text="b range:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.b_min_var = tk.DoubleVar(value=self.main_app.config.b_min)
        self.b_max_var = tk.DoubleVar(value=self.main_app.config.b_max)
        ttk.Entry(ranges_frame, textvariable=self.b_min_var, width=8).grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(ranges_frame, text="to").grid(row=1, column=2, padx=2)
        ttk.Entry(ranges_frame, textvariable=self.b_max_var, width=8).grid(row=1, column=3, padx=2, pady=2)
        
        # C coefficient range
        ttk.Label(ranges_frame, text="c range:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.c_min_var = tk.DoubleVar(value=self.main_app.config.c_min)
        self.c_max_var = tk.DoubleVar(value=self.main_app.config.c_max)
        ttk.Entry(ranges_frame, textvariable=self.c_min_var, width=8).grid(row=2, column=1, padx=2, pady=2)
        ttk.Label(ranges_frame, text="to").grid(row=2, column=2, padx=2)
        ttk.Entry(ranges_frame, textvariable=self.c_max_var, width=8).grid(row=2, column=3, padx=2, pady=2)
        
    def create_generation_options_section(self, parent):
        """Create generation options configuration section."""
        options_frame = ttk.LabelFrame(parent, text="Generation Options")
        options_frame.pack(fill='x', pady=5)
        
        # Prioritize whole solutions
        self.whole_solutions_var = tk.BooleanVar(value=self.main_app.config.prioritize_whole_solutions)
        ttk.Checkbutton(options_frame, text="Prioritize whole number solutions", 
                    variable=self.whole_solutions_var).pack(anchor='w', padx=5, pady=2)
        
        # Whole solution ratio
        ttk.Label(options_frame, text="Whole solution ratio:").pack(anchor='w', padx=5, pady=2)
        self.whole_ratio_var = tk.DoubleVar(value=self.main_app.config.whole_solution_ratio)
        ttk.Scale(options_frame, from_=0.0, to=1.0, variable=self.whole_ratio_var, 
                orient='horizontal').pack(fill='x', padx=5, pady=2)
        
        # Force real solutions
        self.real_solutions_var = tk.BooleanVar(value=self.main_app.config.force_real_solutions)
        ttk.Checkbutton(options_frame, text="Force real solutions only", 
                    variable=self.real_solutions_var).pack(anchor='w', padx=5, pady=2)
        
        # Integer coefficients only
        self.integer_coeff_var = tk.BooleanVar(value=self.main_app.config.integer_coefficients_only)
        ttk.Checkbutton(options_frame, text="Integer coefficients only", 
                    variable=self.integer_coeff_var).pack(anchor='w', padx=5, pady=2)
        
        # Only whole solutions
        self.only_whole_var = tk.BooleanVar(value=self.main_app.config.only_whole_solutions)
        ttk.Checkbutton(options_frame, text="Only accept equations with whole number solutions", 
                    variable=self.only_whole_var).pack(anchor='w', padx=5, pady=2)
        
        # Infinite mode
        self.infinite_mode_var = tk.BooleanVar(value=self.main_app.config.infinite_mode)
        ttk.Checkbutton(options_frame, text="Infinite mode (generate until stopped)", 
                    variable=self.infinite_mode_var).pack(anchor='w', padx=5, pady=2)
        
        # Systematic generation
        self.systematic_var = tk.BooleanVar(value=self.main_app.config.systematic_generation)
        ttk.Checkbutton(options_frame, text="Systematic generation (textbook coverage)", 
                    variable=self.systematic_var).pack(anchor='w', padx=5, pady=2)
        
        # NEW: Textbook mode section
        textbook_frame = ttk.LabelFrame(options_frame, text="📚 Textbook Mode (8th Grade Friendly)")
        textbook_frame.pack(fill='x', padx=5, pady=5)
        
        self.textbook_mode_var = tk.BooleanVar(value=self.main_app.config.textbook_mode)
        ttk.Checkbutton(textbook_frame, text="Enable textbook-style equations with small coefficients", 
                    variable=self.textbook_mode_var).pack(anchor='w', padx=5, pady=2)
        
        # Textbook coefficient size
        coeff_frame = ttk.Frame(textbook_frame)
        coeff_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(coeff_frame, text="Max coefficient size:").pack(side='left', padx=5)
        self.textbook_max_coeff_var = tk.IntVar(value=self.main_app.config.textbook_max_coeff)
        ttk.Spinbox(coeff_frame, from_=3, to=20, textvariable=self.textbook_max_coeff_var, 
                width=5).pack(side='left', padx=5)
        
        # Perfect discriminant option
        self.textbook_perfect_disc_var = tk.BooleanVar(value=self.main_app.config.textbook_prefer_perfect_discriminant)
        ttk.Checkbutton(textbook_frame, text="Prefer perfect square discriminants (cleaner solutions)", 
                    variable=self.textbook_perfect_disc_var).pack(anchor='w', padx=5, pady=2)
        
        # Simple solutions option
        self.textbook_simple_var = tk.BooleanVar(value=self.main_app.config.textbook_simple_solutions)
        ttk.Checkbutton(textbook_frame, text="Generate simple integer solutions when possible", 
                    variable=self.textbook_simple_var).pack(anchor='w', padx=5, pady=2)
        
        # Integer solutions ratio for textbook mode
        ttk.Label(textbook_frame, text="Integer solution ratio for textbook mode:").pack(anchor='w', padx=5, pady=2)
        self.textbook_int_ratio_var = tk.DoubleVar(value=self.main_app.config.textbook_integer_solutions_ratio)
        ttk.Scale(textbook_frame, from_=0.0, to=1.0, variable=self.textbook_int_ratio_var, 
                orient='horizontal').pack(fill='x', padx=5, pady=2)

        
    def create_generation_controls_section(self, parent):
        """Create generation control buttons."""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill='x', pady=10)
        
        # Generate button
        self.generate_button = ttk.Button(controls_frame, text="🚀 Generate Dataset", 
                                        command=self.main_app.start_generation)
        self.generate_button.pack(side='left', padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(controls_frame, text="⏹ Stop Generation", 
                                    command=self.main_app.stop_generation)
        self.stop_button.pack(side='left', padx=5)
        
        # Save button
        ttk.Button(controls_frame, text="💾 Save Dataset", 
                  command=self.main_app.save_dataset).pack(side='left', padx=5)
        
        # Clear button
        ttk.Button(controls_frame, text="🗑 Clear Dataset", 
                  command=self.main_app.clear_dataset).pack(side='left', padx=5)
        
    def create_right_panel(self):
        """Create the right panel with progress and preview."""
        right_frame = ttk.Frame(self.frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        self.create_progress_section(right_frame)
        self.create_preview_section(right_frame)
        
    def create_progress_section(self, parent):
        """Create progress monitoring section."""
        progress_frame = ttk.LabelFrame(parent, text="Generation Progress")
        progress_frame.pack(fill='x', pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        # Status text
        self.status_text = tk.Text(progress_frame, height=6, width=50, font=('Courier', 9))
        status_scroll = ttk.Scrollbar(progress_frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        # Pack status text and scrollbar
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        self.status_text.pack(side='left', fill='both', expand=True)
        status_scroll.pack(side='right', fill='y')
        
    def create_preview_section(self, parent):
        """Create data preview section."""
        preview_frame = ttk.LabelFrame(parent, text="Data Preview")
        preview_frame.pack(fill='both', expand=True, pady=5)
        
        # Create treeview for data preview
        columns = ('a', 'b', 'c', 'x1', 'x2')
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=80)
        
        # Scrollbar for preview
        preview_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)
        
        # Pack preview tree and scrollbar
        self.preview_tree.pack(side='left', fill='both', expand=True)
        preview_scroll.pack(side='right', fill='y')
        
    def update_config_from_ui(self):
        """Update the main configuration from UI values."""
        if not all([self.name_var, self.size_var, self.a_min_var, self.a_max_var,
                self.b_min_var, self.b_max_var, self.c_min_var, self.c_max_var,
                self.whole_solutions_var, self.whole_ratio_var, self.real_solutions_var,
                self.integer_coeff_var, self.only_whole_var, self.infinite_mode_var,
                self.systematic_var]):
            return
            
        self.main_app.config.dataset_name = self.name_var.get()
        self.main_app.config.dataset_size = self.size_var.get()
        self.main_app.config.a_min = self.a_min_var.get()
        self.main_app.config.a_max = self.a_max_var.get()
        self.main_app.config.b_min = self.b_min_var.get()
        self.main_app.config.b_max = self.b_max_var.get()
        self.main_app.config.c_min = self.c_min_var.get()
        self.main_app.config.c_max = self.c_max_var.get()
        self.main_app.config.prioritize_whole_solutions = self.whole_solutions_var.get()
        self.main_app.config.whole_solution_ratio = self.whole_ratio_var.get()
        self.main_app.config.only_whole_solutions = self.only_whole_var.get()
        self.main_app.config.force_real_solutions = self.real_solutions_var.get()
        self.main_app.config.integer_coefficients_only = self.integer_coeff_var.get()
        self.main_app.config.infinite_mode = self.infinite_mode_var.get()
        self.main_app.config.systematic_generation = self.systematic_var.get()
        
        # NEW: Textbook mode configuration
        self.main_app.config.textbook_mode = self.textbook_mode_var.get()
        self.main_app.config.textbook_max_coeff = self.textbook_max_coeff_var.get()
        self.main_app.config.textbook_prefer_perfect_discriminant = self.textbook_perfect_disc_var.get()
        self.main_app.config.textbook_simple_solutions = self.textbook_simple_var.get()
        self.main_app.config.textbook_integer_solutions_ratio = self.textbook_int_ratio_var.get()

        
    def load_config_to_ui(self):
        """Load configuration values into UI components."""
        if not all([self.name_var, self.size_var, self.a_min_var, self.a_max_var,
                self.b_min_var, self.b_max_var, self.c_min_var, self.c_max_var,
                self.whole_solutions_var, self.whole_ratio_var, self.real_solutions_var,
                self.integer_coeff_var, self.only_whole_var, self.infinite_mode_var,
                self.systematic_var]):
            return
            
        self.name_var.set(self.main_app.config.dataset_name)
        self.size_var.set(self.main_app.config.dataset_size)
        self.a_min_var.set(self.main_app.config.a_min)
        self.a_max_var.set(self.main_app.config.a_max)
        self.b_min_var.set(self.main_app.config.b_min)
        self.b_max_var.set(self.main_app.config.b_max)
        self.c_min_var.set(self.main_app.config.c_min)
        self.c_max_var.set(self.main_app.config.c_max)
        self.whole_solutions_var.set(self.main_app.config.prioritize_whole_solutions)
        self.whole_ratio_var.set(self.main_app.config.whole_solution_ratio)
        self.real_solutions_var.set(self.main_app.config.force_real_solutions)
        self.integer_coeff_var.set(self.main_app.config.integer_coefficients_only)
        self.only_whole_var.set(self.main_app.config.only_whole_solutions)
        self.infinite_mode_var.set(self.main_app.config.infinite_mode)
        self.systematic_var.set(self.main_app.config.systematic_generation)
        
        # NEW: Textbook mode configuration loading
        if hasattr(self.main_app.config, 'textbook_mode'):
            self.textbook_mode_var.set(self.main_app.config.textbook_mode)
            self.textbook_max_coeff_var.set(self.main_app.config.textbook_max_coeff)
            self.textbook_perfect_disc_var.set(self.main_app.config.textbook_prefer_perfect_discriminant)
            self.textbook_simple_var.set(self.main_app.config.textbook_simple_solutions)
            self.textbook_int_ratio_var.set(self.main_app.config.textbook_integer_solutions_ratio)
        
    def update_progress(self, current: int, total: float, stats: Dict[str, Any]):
        """Update progress display."""
        if not self.progress_var or not self.status_text:
            return
            
        # Determine mode for display
        is_infinite = self.main_app.config.infinite_mode
        is_textbook = self.main_app.config.textbook_mode
        
        # Clear and update status text
        self.status_text.delete(1.0, tk.END)
        
        if is_infinite:
            # Infinite mode display
            mode_text = "♾️ Infinite"
            if is_textbook:
                mode_text += " Textbook"
            mode_text += " generation"
            
            self.progress_var.set(0)  # No progress bar for infinite
            self.status_text.insert(tk.END, f"{mode_text}: {current} equations\n")
        else:
            # Finite mode display
            progress = (current / total) * 100 if total > 0 else 0
            self.progress_var.set(progress)
            
            mode_text = "📊 Generating"
            if is_textbook:
                mode_text = "📚 Textbook generating"
            
            self.status_text.insert(tk.END, f"{mode_text} equation {current}/{int(total)} ({progress:.1f}%)\n")
        
        # Common statistics
        self.status_text.insert(tk.END, f"Generated: {stats['total_generated']}\n")
        self.status_text.insert(tk.END, f"Whole solutions: {stats['whole_solutions']}\n")
        self.status_text.insert(tk.END, f"Real solutions: {stats['real_solutions']}\n")
        self.status_text.insert(tk.END, f"Rejected: {stats['rejected']}")
        
        # Add textbook mode specific info
        if is_textbook:
            whole_pct = (stats['whole_solutions'] / stats['total_generated'] * 100) if stats['total_generated'] > 0 else 0
            self.status_text.insert(tk.END, f"\n📚 Textbook friendly: {whole_pct:.1f}% whole solutions")
        
        # Auto-scroll to bottom
        self.status_text.see(tk.END)

        
    def update_preview(self):
        """Update the data preview display."""
        if not self.preview_tree:
            return
            
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        # Show current dataset
        if hasattr(self.main_app, 'generation_thread') and self.main_app.generation_thread and self.main_app.generation_thread.is_alive():
            # During generation, show last 50 items
            data_to_show = self.main_app.generator.data[-50:] if len(self.main_app.generator.data) > 50 else self.main_app.generator.data
        else:
            # After generation, show all data (or last 200 for performance)
            data_to_show = self.main_app.generator.data[-200:] if len(self.main_app.generator.data) > 200 else self.main_app.generator.data
        
        for row in data_to_show:
            self.preview_tree.insert('', 'end', values=[f"{val:.3f}" for val in row])
            
    def set_generation_state(self, is_generating: bool):
        """Update UI state based on generation status."""
        if not self.generate_button or not self.stop_button:
            return
            
        if is_generating:
            self.generate_button.config(state='disabled', text='🔄 Generating...')
            self.stop_button.config(state='normal')
            # Start progress bar animation for infinite mode
            if self.main_app.config.infinite_mode:
                self.progress_bar.config(mode='indeterminate')
                self.progress_bar.start()
            else:
                self.progress_bar.config(mode='determinate')
        else:
            # Reset to normal state
            self.generate_button.config(state='normal', text='🚀 Generate Dataset')
            self.stop_button.config(state='disabled')
            # Stop progress bar animation
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')
            # Reset progress to 0
            if hasattr(self, 'progress_var'):
                self.progress_var.set(0)

            
    def validate_inputs(self) -> bool:
        """Validate input values before generation."""
        try:
            # Validate numeric ranges
            if self.a_min_var.get() >= self.a_max_var.get():
                raise ValueError("Invalid 'a' range: min must be less than max")
            if self.b_min_var.get() >= self.b_max_var.get():
                raise ValueError("Invalid 'b' range: min must be less than max") 
            if self.c_min_var.get() >= self.c_max_var.get():
                raise ValueError("Invalid 'c' range: min must be less than max")
            
            # Validate dataset size
            if self.size_var.get() <= 0 and not self.infinite_mode_var.get():
                raise ValueError("Dataset size must be positive")
                
            # Validate whole solution ratio
            if not (0.0 <= self.whole_ratio_var.get() <= 1.0):
                raise ValueError("Whole solution ratio must be between 0 and 1")
                
            return True
            
        except ValueError as e:
            tk.messagebox.showerror("Validation Error", str(e))
            return False
            
    def get_generation_summary(self) -> str:
        """Get a summary of current generation settings."""
        summary = []
        summary.append("📋 GENERATION SETTINGS SUMMARY")
        summary.append("=" * 40)
        summary.append(f"Dataset Name: {self.name_var.get()}")
        
        if self.infinite_mode_var.get():
            summary.append("Mode: Infinite (until stopped)")
        else:
            summary.append(f"Dataset Size: {self.size_var.get()}")
            
        summary.append(f"Coefficient Ranges:")
        summary.append(f"  a: [{self.a_min_var.get()}, {self.a_max_var.get()}]")
        summary.append(f"  b: [{self.b_min_var.get()}, {self.b_max_var.get()}]")
        summary.append(f"  c: [{self.c_min_var.get()}, {self.c_max_var.get()}]")
        
        summary.append("Options:")
        summary.append(f"  Integer coefficients: {'✓' if self.integer_coeff_var.get() else '✗'}")
        summary.append(f"  Whole solutions only: {'✓' if self.only_whole_var.get() else '✗'}")
        summary.append(f"  Prioritize whole solutions: {'✓' if self.whole_solutions_var.get() else '✗'}")
        summary.append(f"  Systematic generation: {'✓' if self.systematic_var.get() else '✗'}")
        
        return '\n'.join(summary)
