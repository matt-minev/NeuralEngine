"""
Simplified GUI for textbook quadratic equation generation.
Focused on simplicity and educational-quality equations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from typing import Optional
import pandas as pd

from simple_config import TextbookConfig, QuadraticSolver
from simple_generator import TextbookQuadraticGenerator


class TextbookGeneratorGUI:
    """Simplified GUI focused on textbook equation generation."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Textbook Quadratic Equation Generator")
        self.root.geometry("900x700")
        
        # Core components
        self.config = TextbookConfig()
        self.generator = TextbookQuadraticGenerator(self.config)
        
        # Generation state
        self.generation_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_generating = False
        
        # UI components
        self.equations_generated = 0
        self.start_time = 0
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Setup the main user interface."""
        # Title and description
        self.create_header()
        
        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Left panel: Controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        self.create_mode_selection(left_frame)
        self.create_generation_controls(left_frame)
        self.create_current_equation_display(left_frame)
        
        # Right panel: Live preview and stats
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        self.create_live_preview(right_frame)
        self.create_statistics_display(right_frame)
        
    def create_header(self):
        """Create application header with description."""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=15, pady=10)
        
        title_label = ttk.Label(header_frame, 
                               text="📚 Textbook Quadratic Equation Generator",
                               font=('TkDefaultFont', 16, 'bold'))
        title_label.pack()
        
        desc_label = ttk.Label(header_frame,
                              text="Generate realistic quadratic equations for educational use\n"
                                   "Perfect for 8th-12th grade mathematics and neural network training",
                              font=('TkDefaultFont', 10),
                              foreground='#666666')
        desc_label.pack(pady=(5, 0))
        
    def create_mode_selection(self, parent):
        """Create mode selection controls."""
        mode_frame = ttk.LabelFrame(parent, text="📖 Generation Mode")
        mode_frame.pack(fill='x', pady=(0, 10))
        
        # Mode selection
        self.mode_var = tk.StringVar(value='textbook')
        
        textbook_rb = ttk.Radiobutton(mode_frame, 
                                     text="🎯 Textbook Mode (Recommended)",
                                     variable=self.mode_var, 
                                     value='textbook',
                                     command=self.on_mode_change)
        textbook_rb.pack(anchor='w', padx=10, pady=5)
        
        advanced_rb = ttk.Radiobutton(mode_frame,
                                     text="🔬 Advanced Mode (Complex equations)",
                                     variable=self.mode_var,
                                     value='advanced',
                                     command=self.on_mode_change)
        advanced_rb.pack(anchor='w', padx=10, pady=5)
        
        # Mode-specific settings
        self.settings_frame = ttk.Frame(mode_frame)
        self.settings_frame.pack(fill='x', padx=10, pady=5)
        
        self.create_textbook_settings()
        
    def create_textbook_settings(self):
        """Create textbook mode settings."""
        # Clear existing settings
        for widget in self.settings_frame.winfo_children():
            widget.destroy()
            
        if self.mode_var.get() == 'textbook':
            # Max coefficient size
            coeff_frame = ttk.Frame(self.settings_frame)
            coeff_frame.pack(fill='x', pady=2)
            
            ttk.Label(coeff_frame, text="Max coefficient:").pack(side='left')
            self.max_coeff_var = tk.IntVar(value=self.config.max_coeff)
            coeff_spinbox = ttk.Spinbox(coeff_frame, from_=3, to=10, 
                                       textvariable=self.max_coeff_var, 
                                       width=5)
            coeff_spinbox.pack(side='right')
            
            # Perfect discriminants
            self.perfect_disc_var = tk.BooleanVar(value=self.config.force_perfect_discriminant)
            ttk.Checkbutton(self.settings_frame, 
                           text="Perfect square discriminants (cleaner solutions)",
                           variable=self.perfect_disc_var).pack(anchor='w', pady=2)
            
            # Whole solutions preference
            self.whole_sols_var = tk.BooleanVar(value=self.config.prefer_whole_solutions)
            ttk.Checkbutton(self.settings_frame,
                           text="Prefer whole number solutions",
                           variable=self.whole_sols_var).pack(anchor='w', pady=2)
            
        else:  # Advanced mode
            # Advanced coefficient range
            adv_frame = ttk.Frame(self.settings_frame)
            adv_frame.pack(fill='x', pady=2)
            
            ttk.Label(adv_frame, text="Max coefficient:").pack(side='left')
            self.adv_coeff_var = tk.IntVar(value=self.config.advanced_max_coeff)
            adv_spinbox = ttk.Spinbox(adv_frame, from_=10, to=50,
                                     textvariable=self.adv_coeff_var,
                                     width=5)
            adv_spinbox.pack(side='right')
            
            # Allow irrational solutions
            self.irrational_var = tk.BooleanVar(value=self.config.allow_irrational_solutions)
            ttk.Checkbutton(self.settings_frame,
                           text="Allow irrational solutions",
                           variable=self.irrational_var).pack(anchor='w', pady=2)
            
    def create_generation_controls(self, parent):
        """Create generation control buttons."""
        controls_frame = ttk.LabelFrame(parent, text="🎮 Generation Controls")
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Main control buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_button = ttk.Button(button_frame, 
                                      text="🚀 Start Generating",
                                      command=self.start_generation,
                                      style='Start.TButton')
        self.start_button.pack(fill='x', pady=2)
        
        self.stop_button = ttk.Button(button_frame,
                                     text="⏹ Stop Generation",
                                     command=self.stop_generation,
                                     state='disabled',
                                     style='Stop.TButton')
        self.stop_button.pack(fill='x', pady=2)
        
        # Action buttons
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        ttk.Button(action_frame, text="💾 Save Dataset",
                  command=self.save_dataset).pack(fill='x', pady=1)
        
        ttk.Button(action_frame, text="🗑 Clear All",
                  command=self.clear_data).pack(fill='x', pady=1)
        
        ttk.Button(action_frame, text="📊 Export Stats",
                  command=self.export_statistics).pack(fill='x', pady=1)
        
    def create_current_equation_display(self, parent):
        """Create current equation display."""
        current_frame = ttk.LabelFrame(parent, text="🔍 Current Equation")
        current_frame.pack(fill='x', pady=(0, 10))
        
        # Equation display
        self.equation_label = ttk.Label(current_frame,
                                       text="Click 'Start Generating' to begin",
                                       font=('Courier', 12),
                                       foreground='#666666')
        self.equation_label.pack(padx=10, pady=10)
        
        # Solutions display
        self.solutions_label = ttk.Label(current_frame,
                                        text="",
                                        font=('Courier', 10),
                                        foreground='#333333')
        self.solutions_label.pack(padx=10, pady=(0, 10))
        
        # Equation info
        self.info_label = ttk.Label(current_frame,
                                   text="",
                                   font=('TkDefaultFont', 8),
                                   foreground='#666666')
        self.info_label.pack(padx=10, pady=(0, 10))
        
    def create_live_preview(self, parent):
        """Create live preview of generated equations."""
        preview_frame = ttk.LabelFrame(parent, text="📝 Live Preview (Latest 20 Equations)")
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Create treeview for equations
        columns = ('Equation', 'Solutions', 'Type')
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, 
                                        show='headings', height=12)
        
        # Configure columns
        self.preview_tree.heading('Equation', text='Equation')
        self.preview_tree.heading('Solutions', text='Solutions')
        self.preview_tree.heading('Type', text='Type')
        
        self.preview_tree.column('Equation', width=200)
        self.preview_tree.column('Solutions', width=150)
        self.preview_tree.column('Type', width=100)
        
        # Scrollbar
        preview_scroll = ttk.Scrollbar(preview_frame, orient='vertical', 
                                      command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)
        
        # Pack tree and scrollbar
        tree_container = ttk.Frame(preview_frame)
        tree_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.preview_tree.pack(side='left', fill='both', expand=True)
        preview_scroll.pack(side='right', fill='y')
        
    def create_statistics_display(self, parent):
        """Create real-time statistics display."""
        stats_frame = ttk.LabelFrame(parent, text="📈 Generation Statistics")
        stats_frame.pack(fill='x')
        
        # Statistics grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill='x', padx=10, pady=10)
        
        # Total generated
        ttk.Label(stats_grid, text="Total Generated:").grid(row=0, column=0, sticky='w', padx=5)
        self.total_label = ttk.Label(stats_grid, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.total_label.grid(row=0, column=1, sticky='w', padx=5)
        
        # Generation rate
        ttk.Label(stats_grid, text="Rate:").grid(row=0, column=2, sticky='w', padx=5)
        self.rate_label = ttk.Label(stats_grid, text="0/sec", font=('TkDefaultFont', 10, 'bold'))
        self.rate_label.grid(row=0, column=3, sticky='w', padx=5)
        
        # Perfect discriminants
        ttk.Label(stats_grid, text="Perfect Discriminants:").grid(row=1, column=0, sticky='w', padx=5)
        self.perfect_label = ttk.Label(stats_grid, text="0 (0%)")
        self.perfect_label.grid(row=1, column=1, sticky='w', padx=5)
        
        # Whole solutions
        ttk.Label(stats_grid, text="Whole Solutions:").grid(row=1, column=2, sticky='w', padx=5)
        self.whole_label = ttk.Label(stats_grid, text="0 (0%)")
        self.whole_label.grid(row=1, column=3, sticky='w', padx=5)
        
        # Textbook friendly
        ttk.Label(stats_grid, text="Textbook Friendly:").grid(row=2, column=0, sticky='w', padx=5)
        self.textbook_label = ttk.Label(stats_grid, text="0 (0%)")
        self.textbook_label.grid(row=2, column=1, sticky='w', padx=5)
        
        # Runtime
        ttk.Label(stats_grid, text="Runtime:").grid(row=2, column=2, sticky='w', padx=5)
        self.runtime_label = ttk.Label(stats_grid, text="00:00:00")
        self.runtime_label.grid(row=2, column=3, sticky='w', padx=5)
        
    def setup_styles(self):
        """Setup custom button styles."""
        style = ttk.Style()
        
        # Start button style
        style.configure('Start.TButton', foreground='green')
        style.configure('Stop.TButton', foreground='red')
        
    def on_mode_change(self):
        """Handle mode change."""
        self.update_config_from_ui()
        self.create_textbook_settings()
        
    def update_config_from_ui(self):
        """Update configuration from UI settings."""
        # Update mode
        self.config.textbook_mode = (self.mode_var.get() == 'textbook')
        
        if self.config.textbook_mode:
            # Textbook mode settings
            if hasattr(self, 'max_coeff_var'):
                self.config.max_coeff = self.max_coeff_var.get()
            if hasattr(self, 'perfect_disc_var'):
                self.config.force_perfect_discriminant = self.perfect_disc_var.get()
            if hasattr(self, 'whole_sols_var'):
                self.config.prefer_whole_solutions = self.whole_sols_var.get()
        else:
            # Advanced mode settings
            if hasattr(self, 'adv_coeff_var'):
                self.config.advanced_max_coeff = self.adv_coeff_var.get()
            if hasattr(self, 'irrational_var'):
                self.config.allow_irrational_solutions = self.irrational_var.get()
        
        # Update generator configuration
        self.generator.config = self.config
        
    def start_generation(self):
        """Start infinite equation generation."""
        if self.is_generating:
            messagebox.showwarning("Warning", "Generation already in progress!")
            return
            
        # Update configuration
        self.update_config_from_ui()
        
        # Reset generator
        self.generator = TextbookQuadraticGenerator(self.config)
        self.stop_event.clear()
        
        # Update UI state
        self.is_generating = True
        self.start_button.config(state='disabled', text='🔄 Generating...')
        self.stop_button.config(state='normal')
        
        # Record start time
        self.start_time = time.time()
        self.equations_generated = 0
        
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self._generation_worker,
            daemon=True
        )
        self.generation_thread.start()
        
        # Start UI update timer
        self.update_ui_timer()
        
    def stop_generation(self):
        """Stop equation generation."""
        self.stop_event.set()
        self.is_generating = False
        
        # Update UI state
        self.start_button.config(state='normal', text='🚀 Start Generating')
        self.stop_button.config(state='disabled')
        
        messagebox.showinfo("Stopped", f"Generation stopped.\nTotal equations: {self.equations_generated}")
        
    def _generation_worker(self):
        """Worker thread for equation generation."""
        try:
            # Generate equations infinitely until stopped
            self.generator.generate_infinite(
                progress_callback=self.progress_callback,
                stop_event=self.stop_event
            )
        except Exception as e:
            # Handle any generation errors
            self.root.after(0, lambda: messagebox.showerror("Error", f"Generation error: {str(e)}"))
        finally:
            # Ensure UI is reset
            self.root.after(0, self._generation_finished)
            
    def _generation_finished(self):
        """Handle generation completion."""
        self.is_generating = False
        self.start_button.config(state='normal', text='🚀 Start Generating')
        self.stop_button.config(state='disabled')
        
    def progress_callback(self, count: int, stats: dict):
        """Handle progress updates from generator."""
        self.equations_generated = count
        
        # Update UI in main thread
        self.root.after(0, lambda: self.update_current_equation())
        
    def update_current_equation(self):
        """Update current equation display."""
        latest = self.generator.get_latest_equation()
        
        if latest:
            # Update equation display
            equation_str = latest['equation_string']
            self.equation_label.config(text=equation_str, foreground='#000000')
            
            # Update solutions display
            x1, x2 = latest['solutions']['x1'], latest['solutions']['x2']
            solutions_str = f"x₁ = {x1:.3f}, x₂ = {x2:.3f}"
            self.solutions_label.config(text=solutions_str)
            
            # Update info
            info_parts = []
            if latest['is_perfect_square']:
                info_parts.append("Perfect discriminant")
            if latest['whole_solutions']:
                info_parts.append("Whole solutions")
                
            info_str = " • ".join(info_parts) if info_parts else "Standard equation"
            self.info_label.config(text=info_str)
            
            # Update preview list
            self.update_preview_list(latest)
            
    def update_preview_list(self, equation_data):
        """Update the preview list with new equation."""
        # Add to preview tree (keep only last 20)
        equation_str = equation_data['equation_string']
        x1, x2 = equation_data['solutions']['x1'], equation_data['solutions']['x2']
        solutions_str = f"{x1:.2f}, {x2:.2f}"
        
        # Determine type
        eq_type = "Standard"
        if equation_data['whole_solutions']:
            eq_type = "Whole"
        elif equation_data['is_perfect_square']:
            eq_type = "Perfect"
            
        # Insert at top
        self.preview_tree.insert('', 0, values=(equation_str, solutions_str, eq_type))
        
        # Keep only last 20 items
        items = self.preview_tree.get_children()
        if len(items) > 20:
            for item in items[20:]:
                self.preview_tree.delete(item)
                
    def update_ui_timer(self):
        """Update UI statistics periodically."""
        if self.is_generating:
            self.update_statistics_display()
            # Schedule next update
            self.root.after(1000, self.update_ui_timer)  # Update every second
            
    def update_statistics_display(self):
        """Update the statistics display."""
        stats = self.generator.get_statistics()
        
        # Total generated
        total = stats['total_generated']
        self.total_label.config(text=str(total))
        
        # Generation rate
        if self.start_time > 0:
            elapsed = time.time() - self.start_time
            rate = total / elapsed if elapsed > 0 else 0
            self.rate_label.config(text=f"{rate:.1f}/sec")
            
            # Runtime
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.runtime_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Percentages
        if total > 0:
            if 'percentages' in stats:
                percentages = stats['percentages']
                
                # Perfect discriminants
                perfect_count = stats['perfect_discriminants']
                perfect_pct = percentages['perfect_discriminants']
                self.perfect_label.config(text=f"{perfect_count} ({perfect_pct:.1f}%)")
                
                # Whole solutions
                whole_count = stats['whole_solutions']
                whole_pct = percentages['whole_solutions']
                self.whole_label.config(text=f"{whole_count} ({whole_pct:.1f}%)")
                
                # Textbook friendly
                textbook_count = stats['textbook_friendly']
                textbook_pct = percentages['textbook_friendly']
                self.textbook_label.config(text=f"{textbook_count} ({textbook_pct:.1f}%)")
                
    def save_dataset(self):
        """Save generated dataset."""
        if not self.generator.data:
            messagebox.showwarning("Warning", "No data to save!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.config.dataset_name}.csv"
        )
        
        if filename:
            try:
                self.generator.save_dataset(filename)
                messagebox.showinfo("Success", 
                    f"Saved {len(self.generator.data)} equations to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
                
    def clear_data(self):
        """Clear all generated data."""
        if not self.generator.data:
            messagebox.showinfo("Info", "No data to clear.")
            return
            
        result = messagebox.askyesno("Confirm", 
            f"Clear all {len(self.generator.data)} generated equations?")
        
        if result:
            self.generator.reset_data()
            
            # Clear preview
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
                
            # Reset displays
            self.equation_label.config(text="Click 'Start Generating' to begin",
                                      foreground='#666666')
            self.solutions_label.config(text="")
            self.info_label.config(text="")
            
            # Reset statistics
            self.total_label.config(text="0")
            self.rate_label.config(text="0/sec")
            self.perfect_label.config(text="0 (0%)")
            self.whole_label.config(text="0 (0%)")
            self.textbook_label.config(text="0 (0%)")
            self.runtime_label.config(text="00:00:00")
            
            messagebox.showinfo("Success", "All data cleared!")
            
    def export_statistics(self):
        """Export generation statistics."""
        if not self.generator.data:
            messagebox.showwarning("Warning", "No data to export!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"{self.config.dataset_name}_stats.txt"
        )
        
        if filename:
            try:
                stats = self.generator.get_statistics()
                elapsed = time.time() - self.start_time if self.start_time > 0 else 0
                
                with open(filename, 'w') as f:
                    f.write("TEXTBOOK QUADRATIC GENERATOR STATISTICS\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Mode: {'Textbook' if self.config.textbook_mode else 'Advanced'}\n")
                    f.write(f"Total Equations: {stats['total_generated']}\n")
                    f.write(f"Generation Time: {elapsed:.1f} seconds\n")
                    f.write(f"Generation Rate: {stats['total_generated']/elapsed:.1f} equations/sec\n\n")
                    
                    if 'percentages' in stats:
                        f.write("Quality Metrics:\n")
                        f.write(f"Perfect Discriminants: {stats['perfect_discriminants']} ({stats['percentages']['perfect_discriminants']:.1f}%)\n")
                        f.write(f"Whole Solutions: {stats['whole_solutions']} ({stats['percentages']['whole_solutions']:.1f}%)\n")
                        f.write(f"Textbook Friendly: {stats['textbook_friendly']} ({stats['percentages']['textbook_friendly']:.1f}%)\n")
                
                messagebox.showinfo("Success", f"Statistics exported to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")


def main():
    """Main entry point for the simplified generator."""
    root = tk.Tk()
    app = TextbookGeneratorGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.is_generating:
            result = messagebox.askyesno("Confirm", 
                "Generation is in progress. Stop and exit?")
            if result:
                app.stop_generation()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Run application
    root.mainloop()


if __name__ == "__main__":
    main()
