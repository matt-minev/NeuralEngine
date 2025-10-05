"""
Main GUI window controller for the quadratic equation dataset generator.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Optional, Callable, Any

from config import GenerationConfig
from generator import DatasetGenerator
from gui_generation_tab import GenerationTab
from gui_editor_tab import EditorTab
from gui_visualization_tab import VisualizationTab


class QuadraticDatasetGUI:
    """Main GUI application controller for dataset generation."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Quadratic Equation Dataset Generator")
        self.root.geometry("1200x800")
        
        # Core components
        self.config = GenerationConfig()
        self.generator = DatasetGenerator(self.config)
        
        # Threading components
        self.generation_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # GUI components will be initialized in setup_ui
        self.notebook: Optional[ttk.Notebook] = None
        self.generation_tab: Optional[GenerationTab] = None
        self.editor_tab: Optional[EditorTab] = None
        self.visualization_tab: Optional[VisualizationTab] = None
        
        # Initialize the GUI
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Setup the main user interface."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create and add tabs
        self.generation_tab = GenerationTab(self.notebook, self)
        self.editor_tab = EditorTab(self.notebook, self)
        self.visualization_tab = VisualizationTab(self.notebook, self)
        
    def setup_styles(self):
        """Configure custom styles for the application."""
        style = ttk.Style()
        
        # Configure error highlighting style
        style.configure('Error.TEntry', fieldbackground='#ffcccc')
        
        # Configure success highlighting style
        style.configure('Success.TEntry', fieldbackground='#ccffcc')
        
        # Configure warning highlighting style  
        style.configure('Warning.TEntry', fieldbackground='#ffffcc')
        
    def update_config(self):
        """Update the configuration from all GUI components."""
        if self.generation_tab:
            self.generation_tab.update_config_from_ui()
            
    def start_generation(self):
        """Start dataset generation in a separate thread."""
        if self.generation_thread and self.generation_thread.is_alive():
            messagebox.showwarning("Warning", "Generation already in progress!")
            return
            
        # Update configuration from UI
        self.update_config()
        
        # Create new generator with updated config
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
        
        # Update UI state
        if self.generation_tab:
            self.generation_tab.set_generation_state(True)
            
    def stop_generation(self):
        """Stop dataset generation."""
        self.stop_event.set()
        messagebox.showinfo("Stopped", "Generation stopped by user.")
        
        # Update UI state
        if self.generation_tab:
            self.generation_tab.set_generation_state(False)
            
    def check_generation_complete(self):
        """Check if generation is complete and update UI accordingly."""
        if self.generation_thread and self.generation_thread.is_alive():
            self.root.after(100, self.check_generation_complete)
        else:
            # Generation completed
            self.update_all_tabs()
            
            # Update UI state
            if self.generation_tab:
                self.generation_tab.set_generation_state(False)
                
            messagebox.showinfo("Complete", "Dataset generation completed!")
            
    def progress_callback(self, current: int, total: float, stats: dict):
        """Handle progress updates from the generator."""
        if self.generation_tab:
            self.generation_tab.update_progress(current, total, stats)
            
    def update_all_tabs(self):
        """Update all tabs with current data."""
        if self.generation_tab:
            self.generation_tab.update_preview()
            
        if self.editor_tab:
            self.editor_tab.update_dataset_viewer()
            
        if self.visualization_tab:
            self.visualization_tab.update_plots()
            
    def save_dataset(self):
        """Save the generated dataset."""
        if not self.generator.data:
            messagebox.showwarning("Warning", "No data to save!")
            return
            
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.config.dataset_name}.csv"
        )
        
        if filename:
            try:
                self.generator.save_dataset(filename)
                messagebox.showinfo("Success", f"Dataset saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
                
    def add_equation_to_dataset(self, a: float, b: float, c: float, x1: float, x2: float) -> bool:
        """Add a manually entered equation to the dataset."""
        success = self.generator.add_equation(a, b, c, x1, x2)
        
        if success:
            # Update all tabs
            self.update_all_tabs()
            
        return success
        
    def get_dataset_summary(self) -> dict:
        """Get summary statistics of the current dataset."""
        return self.generator.get_data_summary()
        
    def get_dataset_dataframe(self):
        """Get the dataset as a pandas DataFrame."""
        return self.generator.get_dataframe()
        
    def clear_dataset(self):
        """Clear all data from the dataset."""
        result = messagebox.askyesno("Confirm", "Are you sure you want to clear all data?")
        
        if result:
            self.generator.clear_data()
            self.update_all_tabs()
            messagebox.showinfo("Success", "Dataset cleared!")
            
    def export_config(self):
        """Export current configuration to a file."""
        from tkinter import filedialog
        import json
        from dataclasses import asdict
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.config.dataset_name}_config.json"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(asdict(self.config), f, indent=2)
                messagebox.showinfo("Success", f"Configuration exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export configuration: {str(e)}")
                
    def import_config(self):
        """Import configuration from a file."""
        from tkinter import filedialog
        import json
        
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = json.load(f)
                    
                # Update configuration
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
                # Update UI with new configuration
                if self.generation_tab:
                    self.generation_tab.load_config_to_ui()
                    
                messagebox.showinfo("Success", f"Configuration imported from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import configuration: {str(e)}")
                
    def show_about(self):
        """Show about dialog."""
        about_text = """
Quadratic Equation Dataset Generator

A powerful tool for generating and managing quadratic equation datasets.

Features:
• Multiple generation modes (random, systematic, whole-number focused)
• Interactive manual equation editor
• Real-time visualization and statistics
• Export capabilities for machine learning applications
• Comprehensive configuration options

Created for neural network training and mathematical research.

Version: 2.0
        """
        
        messagebox.showinfo("About", about_text)
        
    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Dataset", command=self.save_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Export Config", command=self.export_config)
        file_menu.add_command(label="Import Config", command=self.import_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Generation menu
        generation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Generation", menu=generation_menu)
        generation_menu.add_command(label="Start Generation", command=self.start_generation)
        generation_menu.add_command(label="Stop Generation", command=self.stop_generation)
        generation_menu.add_separator()
        generation_menu.add_command(label="Clear Dataset", command=self.clear_dataset)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def run(self):
        """Run the main application loop."""
        self.create_menu()
        self.root.mainloop()
        
    def on_closing(self):
        """Handle application closing."""
        if self.generation_thread and self.generation_thread.is_alive():
            result = messagebox.askyesno("Confirm", "Generation is in progress. Do you want to stop it and exit?")
            if result:
                self.stop_generation()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = QuadraticDatasetGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Run the application
    app.run()


if __name__ == "__main__":
    main()
