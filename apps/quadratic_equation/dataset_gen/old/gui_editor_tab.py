"""
Editor tab component for manual quadratic equation editing and validation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Tuple
import numpy as np

from config import QuadraticSolver


class EditorTab:
    """GUI tab for manual equation editing and data input."""
    
    def __init__(self, notebook: ttk.Notebook, main_app):
        self.main_app = main_app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Manual Editor")
        
        # UI variables for input fields
        self.edit_a_var: Optional[tk.DoubleVar] = None
        self.edit_b_var: Optional[tk.DoubleVar] = None
        self.edit_c_var: Optional[tk.DoubleVar] = None
        self.edit_x1_var: Optional[tk.StringVar] = None
        self.edit_x2_var: Optional[tk.StringVar] = None
        
        # UI components
        self.edit_a_entry: Optional[ttk.Entry] = None
        self.edit_b_entry: Optional[ttk.Entry] = None
        self.edit_c_entry: Optional[ttk.Entry] = None
        self.edit_x1_entry: Optional[ttk.Entry] = None
        self.edit_x2_entry: Optional[ttk.Entry] = None
        self.results_text: Optional[tk.Text] = None
        self.manual_tree: Optional[ttk.Treeview] = None
        
        # Validation state
        self.validation_errors = set()
        
        self.setup_ui()
        self.setup_validation()
        
    def setup_ui(self):
        """Setup the editor tab interface."""
        self.create_instructions()
        self.create_editor_section()
        self.create_dataset_viewer()
        
    def create_instructions(self):
        """Create instruction section."""
        instructions_frame = ttk.Frame(self.frame)
        instructions_frame.pack(fill='x', padx=10, pady=5)
        
        instructions_text = (
            "📝 Manual Equation Editor\n"
            "• Enter coefficients a, b, c to automatically calculate solutions\n"
            "• Optionally enter x1, x2 values for verification (red = incorrect)\n"
            "• Use 'Paste Data' to import multiple equations from CSV format\n"
            "• All valid equations are added to the current dataset"
        )
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                                     font=('TkDefaultFont', 9), 
                                     foreground='#666666')
        instructions_label.pack(anchor='w')
        
    def create_editor_section(self):
        """Create the equation editor section."""
        editor_section = ttk.LabelFrame(self.frame, text="Add/Edit Equation")
        editor_section.pack(fill='x', padx=10, pady=5)
        
        # Input fields container
        input_container = ttk.Frame(editor_section)
        input_container.pack(fill='x', padx=10, pady=10)
        
        self.create_coefficient_inputs(input_container)
        self.create_solution_inputs(input_container)
        self.create_editor_buttons(editor_section)
        self.create_results_display(editor_section)
        
    def create_coefficient_inputs(self, parent):
        """Create coefficient input fields."""
        coeff_frame = ttk.LabelFrame(parent, text="Coefficients (ax² + bx + c = 0)")
        coeff_frame.pack(fill='x', pady=(0, 5))
        
        # Create input grid
        input_grid = ttk.Frame(coeff_frame)
        input_grid.pack(fill='x', padx=10, pady=10)
        
        # A coefficient
        ttk.Label(input_grid, text="a:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.edit_a_var = tk.DoubleVar()
        self.edit_a_entry = ttk.Entry(input_grid, textvariable=self.edit_a_var, width=12)
        self.edit_a_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # B coefficient
        ttk.Label(input_grid, text="b:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.edit_b_var = tk.DoubleVar()
        self.edit_b_entry = ttk.Entry(input_grid, textvariable=self.edit_b_var, width=12)
        self.edit_b_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # C coefficient
        ttk.Label(input_grid, text="c:").grid(row=0, column=4, padx=5, pady=5, sticky='e')
        self.edit_c_var = tk.DoubleVar()
        self.edit_c_entry = ttk.Entry(input_grid, textvariable=self.edit_c_var, width=12)
        self.edit_c_entry.grid(row=0, column=5, padx=5, pady=5)
        
        # Auto-calculate button
        ttk.Button(input_grid, text="🧮 Calculate", 
                  command=self.calculate_solutions).grid(row=0, column=6, padx=10, pady=5)
        
    def create_solution_inputs(self, parent):
        """Create solution input fields."""
        solution_frame = ttk.LabelFrame(parent, text="Solutions (Optional - for verification)")
        solution_frame.pack(fill='x', pady=5)
        
        # Create solution grid
        solution_grid = ttk.Frame(solution_frame)
        solution_grid.pack(fill='x', padx=10, pady=10)
        
        # X1 solution
        ttk.Label(solution_grid, text="x₁:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.edit_x1_var = tk.StringVar()
        self.edit_x1_entry = ttk.Entry(solution_grid, textvariable=self.edit_x1_var, width=15)
        self.edit_x1_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # X2 solution
        ttk.Label(solution_grid, text="x₂:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.edit_x2_var = tk.StringVar()
        self.edit_x2_entry = ttk.Entry(solution_grid, textvariable=self.edit_x2_var, width=15)
        self.edit_x2_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Verification status
        self.verification_label = ttk.Label(solution_grid, text="", font=('TkDefaultFont', 8))
        self.verification_label.grid(row=0, column=4, padx=10, pady=5)
        
    def create_editor_buttons(self, parent):
        """Create editor action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        # Primary actions
        ttk.Button(button_frame, text="➕ Add to Dataset", 
                  command=self.add_to_dataset).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🗑 Clear Fields", 
                  command=self.clear_fields).pack(side='left', padx=5)
        
        # Data import actions
        ttk.Button(button_frame, text="📋 Paste Data", 
                  command=self.paste_data).pack(side='left', padx=20)
        
        ttk.Button(button_frame, text="📁 Import CSV", 
                  command=self.import_csv).pack(side='left', padx=5)
        
        # Random generation
        ttk.Button(button_frame, text="🎲 Random Equation", 
                  command=self.generate_random_equation).pack(side='left', padx=20)
        
    def create_results_display(self, parent):
        """Create results display area."""
        results_frame = ttk.LabelFrame(parent, text="Calculation Results")
        results_frame.pack(fill='x', padx=10, pady=5)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=4, width=80, font=('Courier', 9))
        results_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        # Pack results area
        results_container = ttk.Frame(results_frame)
        results_container.pack(fill='x', padx=5, pady=5)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scroll.pack(side='right', fill='y')
        
    def create_dataset_viewer(self):
        """Create dataset viewer section."""
        viewer_frame = ttk.LabelFrame(self.frame, text="Current Dataset")
        viewer_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(viewer_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(control_frame, text="Dataset actions:").pack(side='left', padx=5)
        ttk.Button(control_frame, text="🔄 Refresh", 
                  command=self.update_dataset_viewer).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🗑 Remove Selected", 
                  command=self.remove_selected_equation).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📊 Show Statistics", 
                  command=self.show_dataset_statistics).pack(side='left', padx=5)
        
        # Dataset tree view
        tree_frame = ttk.Frame(viewer_frame)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        columns = ('Index', 'a', 'b', 'c', 'x1', 'x2', 'Verification')
        self.manual_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=12)
        
        # Configure columns
        self.manual_tree.heading('Index', text='#')
        self.manual_tree.column('Index', width=50)
        
        for col in ['a', 'b', 'c', 'x1', 'x2']:
            self.manual_tree.heading(col, text=col)
            self.manual_tree.column(col, width=80)
            
        self.manual_tree.heading('Verification', text='✓')
        self.manual_tree.column('Verification', width=60)
        
        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.manual_tree.yview)
        self.manual_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Pack tree and scrollbar
        self.manual_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        
        # Bind events
        self.manual_tree.bind('<Double-1>', self.on_equation_double_click)
        
    def setup_validation(self):
        """Setup real-time validation for input fields."""
        self.edit_a_var.trace('w', self.validate_input)
        self.edit_b_var.trace('w', self.validate_input)
        self.edit_c_var.trace('w', self.validate_input)
        self.edit_x1_var.trace('w', self.validate_input)
        self.edit_x2_var.trace('w', self.validate_input)
        
    def validate_input(self, *args):
        """Validate manual input and highlight errors."""
        try:
            # Clear previous validation errors
            self.validation_errors.clear()
            
            # Get coefficient values
            a = self.edit_a_var.get()
            b = self.edit_b_var.get()
            c = self.edit_c_var.get()
            
            # Calculate correct solutions
            x1_correct, x2_correct = QuadraticSolver.solve_quadratic(a, b, c)
            
            # Reset entry styles
            self.edit_x1_entry.configure(style='TEntry')
            self.edit_x2_entry.configure(style='TEntry')
            
            # Validate manual solutions if provided
            x1_manual = self.edit_x1_var.get().strip()
            x2_manual = self.edit_x2_var.get().strip()
            
            verification_status = []
            
            if x1_manual and x1_correct is not None:
                try:
                    x1_val = float(x1_manual)
                    if QuadraticSolver.verify_solution(a, b, c, x1_val):
                        verification_status.append("x₁✓")
                    else:
                        self.edit_x1_entry.configure(style='Error.TEntry')
                        self.validation_errors.add('x1')
                        verification_status.append("x₁✗")
                except ValueError:
                    self.edit_x1_entry.configure(style='Error.TEntry')
                    self.validation_errors.add('x1')
                    verification_status.append("x₁✗")
            
            if x2_manual and x2_correct is not None:
                try:
                    x2_val = float(x2_manual)
                    if QuadraticSolver.verify_solution(a, b, c, x2_val):
                        verification_status.append("x₂✓")
                    else:
                        self.edit_x2_entry.configure(style='Error.TEntry')
                        self.validation_errors.add('x2')
                        verification_status.append("x₂✗")
                except ValueError:
                    self.edit_x2_entry.configure(style='Error.TEntry')
                    self.validation_errors.add('x2')
                    verification_status.append("x₂✗")
            
            # Update verification label
            if verification_status:
                self.verification_label.config(text=" ".join(verification_status))
            else:
                self.verification_label.config(text="")
                
        except Exception:
            # Handle calculation errors silently
            pass
            
    def calculate_solutions(self):
        """Calculate and display solutions for current coefficients."""
        try:
            a = self.edit_a_var.get()
            b = self.edit_b_var.get()
            c = self.edit_c_var.get()
            
            # Solve quadratic equation
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            # Clear and update results display
            self.results_text.delete(1.0, tk.END)
            
            if x1 is not None and x2 is not None:
                # Display equation and solutions
                self.results_text.insert(tk.END, f"📐 Equation: {a}x² + {b}x + {c} = 0\n")
                self.results_text.insert(tk.END, f"🎯 Solutions:\n")
                self.results_text.insert(tk.END, f"   x₁ = {x1:.6f}\n")
                self.results_text.insert(tk.END, f"   x₂ = {x2:.6f}\n")
                
                # Auto-fill solution fields
                self.edit_x1_var.set(f"{x1:.6f}")
                self.edit_x2_var.set(f"{x2:.6f}")
                
                # Additional information
                discriminant = b**2 - 4*a*c
                self.results_text.insert(tk.END, f"ℹ️  Discriminant: {discriminant:.6f}")
                
                if QuadraticSolver.is_whole_number(x1) or QuadraticSolver.is_whole_number(x2):
                    self.results_text.insert(tk.END, " (Contains whole number solutions)")
                    
            else:
                self.results_text.insert(tk.END, f"📐 Equation: {a}x² + {b}x + {c} = 0\n")
                self.results_text.insert(tk.END, "❌ No real solutions exist for this equation.\n")
                
                discriminant = b**2 - 4*a*c
                self.results_text.insert(tk.END, f"ℹ️  Discriminant: {discriminant:.6f} (negative)")
                
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"❌ Error: {str(e)}")
            
    def add_to_dataset(self):
        """Add current equation to the dataset."""
        try:
            # Validate coefficients
            a = self.edit_a_var.get()
            b = self.edit_b_var.get()
            c = self.edit_c_var.get()
            
            # Calculate solutions
            x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
            
            if x1 is not None and x2 is not None:
                # Check for validation errors
                if self.validation_errors:
                    result = messagebox.askyesno(
                        "Validation Errors", 
                        "There are validation errors in manual solutions. "
                        "Add equation with calculated solutions instead?"
                    )
                    if not result:
                        return
                
                # Add to main dataset
                success = self.main_app.add_equation_to_dataset(a, b, c, x1, x2)
                
                if success:
                    self.clear_fields()
                    messagebox.showinfo("Success", "Equation added to dataset!")
                else:
                    messagebox.showerror("Error", "Failed to add equation to dataset.")
            else:
                messagebox.showerror("Error", "Cannot add equation with no real solutions.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add equation: {str(e)}")
            
    def clear_fields(self):
        """Clear all input fields."""
        self.edit_a_var.set(0)
        self.edit_b_var.set(0)
        self.edit_c_var.set(0)
        self.edit_x1_var.set("")
        self.edit_x2_var.set("")
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.verification_label.config(text="")
        
    def paste_data(self):
        """Handle pasting data from clipboard."""
        try:
            # Get clipboard content
            clipboard_data = self.main_app.root.clipboard_get()
            
            # Create paste dialog
            self.show_paste_dialog(clipboard_data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to access clipboard: {str(e)}")
            
    def show_paste_dialog(self, clipboard_data: str):
        """Show dialog for pasting and previewing clipboard data."""
        dialog = tk.Toplevel(self.main_app.root)
        dialog.title("Paste Data Preview")
        dialog.geometry("600x400")
        dialog.transient(self.main_app.root)
        dialog.grab_set()
        
        # Instructions
        ttk.Label(dialog, text="Preview clipboard data (CSV format: a,b,c or a,b,c,x1,x2):").pack(pady=10)
        
        # Data preview
        preview_frame = ttk.Frame(dialog)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        preview_text = tk.Text(preview_frame, height=15, width=70, font=('Courier', 9))
        preview_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=preview_text.yview)
        preview_text.configure(yscrollcommand=preview_scroll.set)
        
        preview_text.pack(side='left', fill='both', expand=True)
        preview_scroll.pack(side='right', fill='y')
        
        # Insert clipboard data
        preview_text.insert('1.0', clipboard_data)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        def import_data():
            data = preview_text.get('1.0', tk.END)
            dialog.destroy()
            self.process_paste_data(data)
            
        ttk.Button(button_frame, text="✅ Import Data", command=import_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side='left', padx=5)
        
    def process_paste_data(self, data: str):
        """Process pasted data and add valid equations."""
        lines = data.strip().split('\n')
        added_count = 0
        error_count = 0
        errors = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
                
            try:
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) >= 3:
                    a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
                    
                    # Calculate solutions
                    x1, x2 = QuadraticSolver.solve_quadratic(a, b, c)
                    
                    if x1 is not None and x2 is not None:
                        success = self.main_app.add_equation_to_dataset(a, b, c, x1, x2)
                        if success:
                            added_count += 1
                        else:
                            error_count += 1
                            errors.append(f"Line {line_num}: Failed to add equation")
                    else:
                        error_count += 1
                        errors.append(f"Line {line_num}: No real solutions")
                else:
                    error_count += 1
                    errors.append(f"Line {line_num}: Invalid format (need at least 3 values)")
                    
            except ValueError as e:
                error_count += 1
                errors.append(f"Line {line_num}: {str(e)}")
                
        # Show results
        result_message = f"Import completed!\n\n✅ Added: {added_count} equations"
        if error_count > 0:
            result_message += f"\n❌ Errors: {error_count} lines"
            if len(errors) <= 5:
                result_message += "\n\nErrors:\n" + "\n".join(errors)
            else:
                result_message += f"\n\nFirst 5 errors:\n" + "\n".join(errors[:5])
                result_message += f"\n... and {len(errors) - 5} more"
                
        messagebox.showinfo("Import Results", result_message)
        
    def import_csv(self):
        """Import equations from CSV file."""
        from tkinter import filedialog
        
        filename = filedialog.askopenfilename(
            title="Import CSV File",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = f.read()
                self.process_paste_data(data)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {str(e)}")
                
    def generate_random_equation(self):
        """Generate a random quadratic equation."""
        # Use current generator configuration for ranges
        config = self.main_app.config
        
        if config.integer_coefficients_only:
            a = np.random.randint(int(config.a_min), int(config.a_max) + 1)
            b = np.random.randint(int(config.b_min), int(config.b_max) + 1)
            c = np.random.randint(int(config.c_min), int(config.c_max) + 1)
        else:
            a = np.random.uniform(config.a_min, config.a_max)
            b = np.random.uniform(config.b_min, config.b_max)
            c = np.random.uniform(config.c_min, config.c_max)
            
        if config.avoid_zero_a and a == 0:
            a = 1.0 if np.random.random() < 0.5 else -1.0
            
        # Set values and calculate
        self.edit_a_var.set(a)
        self.edit_b_var.set(b)
        self.edit_c_var.set(c)
        
        self.calculate_solutions()
        
    def update_dataset_viewer(self):
        """Update the dataset viewer with current data."""
        # Clear existing items
        for item in self.manual_tree.get_children():
            self.manual_tree.delete(item)
            
        # Get current dataset
        if hasattr(self.main_app, 'generator') and self.main_app.generator.data:
            data = self.main_app.generator.data
            
            for i, row in enumerate(data):
                a, b, c, x1, x2 = row
                
                # Verify solutions
                verification = "✓"
                if not (QuadraticSolver.verify_solution(a, b, c, x1) and 
                       QuadraticSolver.verify_solution(a, b, c, x2)):
                    verification = "✗"
                    
                self.manual_tree.insert('', 'end', values=(
                    i + 1,
                    f"{a:.3f}",
                    f"{b:.3f}", 
                    f"{c:.3f}",
                    f"{x1:.3f}",
                    f"{x2:.3f}",
                    verification
                ))
                
    def remove_selected_equation(self):
        """Remove selected equation from dataset."""
        selection = self.manual_tree.selection()
        
        if not selection:
            messagebox.showwarning("Warning", "Please select an equation to remove.")
            return
            
        # Get selected index
        item = selection[0]
        index_str = self.manual_tree.item(item, 'values')[0]
        index = int(index_str) - 1
        
        # Confirm removal
        result = messagebox.askyesno("Confirm", f"Remove equation #{index + 1}?")
        
        if result and hasattr(self.main_app, 'generator'):
            try:
                del self.main_app.generator.data[index]
                self.main_app.update_all_tabs()
                messagebox.showinfo("Success", "Equation removed!")
            except IndexError:
                messagebox.showerror("Error", "Invalid equation index.")
                
    def show_dataset_statistics(self):
        """Show statistics for current dataset."""
        if not hasattr(self.main_app, 'generator') or not self.main_app.generator.data:
            messagebox.showinfo("Statistics", "No data available.")
            return
            
        summary = self.main_app.get_dataset_summary()
        
        stats_text = "📊 DATASET STATISTICS\n"
        stats_text += "=" * 30 + "\n"
        stats_text += f"Total equations: {summary.get('total_equations', 0)}\n\n"
        
        if 'coefficient_stats' in summary:
            stats_text += "Coefficient ranges:\n"
            for coef, stats in summary['coefficient_stats'].items():
                stats_text += f"  {coef}: {stats['min']:.3f} to {stats['max']:.3f} (avg: {stats['mean']:.3f})\n"
                
        if 'generation_stats' in summary:
            gen_stats = summary['generation_stats']
            stats_text += f"\nGeneration statistics:\n"
            stats_text += f"  Whole solutions: {gen_stats.get('whole_solutions', 0)}\n"
            stats_text += f"  Real solutions: {gen_stats.get('real_solutions', 0)}\n"
            stats_text += f"  Rejected: {gen_stats.get('rejected', 0)}\n"
            
        messagebox.showinfo("Dataset Statistics", stats_text)
        
    def on_equation_double_click(self, event):
        """Handle double-click on equation in tree view."""
        selection = self.manual_tree.selection()
        
        if selection:
            item = selection[0]
            values = self.manual_tree.item(item, 'values')
            
            if len(values) >= 6:
                # Load equation into editor
                try:
                    index = int(values[0]) - 1
                    a, b, c, x1, x2 = float(values[1]), float(values[2]), float(values[3]), float(values[4]), float(values[5])
                    
                    self.edit_a_var.set(a)
                    self.edit_b_var.set(b)
                    self.edit_c_var.set(c)
                    self.edit_x1_var.set(f"{x1:.6f}")
                    self.edit_x2_var.set(f"{x2:.6f}")
                    
                    self.calculate_solutions()
                    
                except (ValueError, IndexError):
                    messagebox.showerror("Error", "Failed to load equation data.")
