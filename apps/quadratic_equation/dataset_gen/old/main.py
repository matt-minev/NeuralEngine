#!/usr/bin/env python3
"""
Main entry point for the Quadratic Equation Dataset Generator.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add the parent directory to Python path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import with absolute imports
from config import GenerationConfig
from generator import DatasetGenerator
from gui_generation_tab import GenerationTab
from gui_editor_tab import EditorTab
from gui_visualization_tab import VisualizationTab

class QuadraticDatasetGUI:
    """Main GUI application controller for dataset generation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Quadratic Equation Dataset Generator")
        self.root.geometry("1200x800")
        
        # Core components
        self.config = GenerationConfig()
        self.generator = DatasetGenerator(self.config)
        
        # Threading components
        self.generation_thread = None
        self.stop_event = None
        
        # GUI components
        self.notebook = None
        self.generation_tab = None
        self.editor_tab = None
        self.visualization_tab = None
        
        # Initialize the GUI
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Setup the main user interface."""
        import threading
        self.stop_event = threading.Event()
        
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
        style.configure('Success.TEntry', fieldbackground='#ccffcc')
        style.configure('Warning.TEntry', fieldbackground='#ffffcc')
        
    def update_config(self):
        """Update the configuration from all GUI components."""
        if self.generation_tab:
            self.generation_tab.update_config_from_ui()
            
    def start_generation(self):
        """Start dataset generation in a separate thread."""
        import threading
        from tkinter import messagebox
        
        # Check if generation is actually running (more robust check)
        if (self.generation_thread and 
            self.generation_thread.is_alive() and 
            not (self.stop_event and self.stop_event.is_set())):
            messagebox.showwarning("Warning", "Generation already in progress!")
            return
        
        # Clean up any previous thread
        if self.generation_thread:
            self.generation_thread = None
        
        # Update configuration from UI
        self.update_config()
        
        # Create new generator with updated config
        self.generator = DatasetGenerator(self.config)
        self.stop_event.clear()
        
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self.generator.generate_dataset,
            args=(self.progress_callback, self.stop_event),
            daemon=True  # NEW: Make thread daemon for cleaner shutdown
        )
        self.generation_thread.start()
        
        # Monitor thread completion
        self.root.after(100, self.check_generation_complete)
        
        # Update UI state
        if self.generation_tab:
            self.generation_tab.set_generation_state(True)

            
    def stop_generation(self):
        """Stop dataset generation."""
        from tkinter import messagebox
        
        # Set stop event
        if self.stop_event:
            self.stop_event.set()
        
        # Wait for thread to complete (with timeout)
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        # Force reset generation state
        self.generation_thread = None
        
        # Update UI state immediately
        if self.generation_tab:
            self.generation_tab.set_generation_state(False)
        
        messagebox.showinfo("Stopped", "Generation stopped by user.")
            
    def check_generation_complete(self):
        """Check if generation is complete and update UI accordingly."""
        from tkinter import messagebox
        
        # More robust thread checking
        if (self.generation_thread and 
            self.generation_thread.is_alive() and 
            not (self.stop_event and self.stop_event.is_set())):
            # Still running, check again later
            self.root.after(100, self.check_generation_complete)
        else:
            # Generation completed or stopped
            self.generation_thread = None  # Clear thread reference
            
            # Update all tabs
            self.update_all_tabs()
            
            # Update UI state
            if self.generation_tab:
                self.generation_tab.set_generation_state(False)
            
            # Show completion message only if not manually stopped
            if not (self.stop_event and self.stop_event.is_set()):
                messagebox.showinfo("Complete", "Dataset generation completed!")

            
    def progress_callback(self, current, total, stats):
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
        from tkinter import filedialog, messagebox
        
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
                messagebox.showinfo("Success", f"Dataset saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
                
    def add_equation_to_dataset(self, a, b, c, x1, x2):
        """Add a manually entered equation to the dataset."""
        success = self.generator.add_equation(a, b, c, x1, x2)
        
        if success:
            # Update all tabs
            self.update_all_tabs()
            
        return success
        
    def get_dataset_summary(self):
        """Get summary statistics of the current dataset."""
        return self.generator.get_data_summary()
        
    def get_dataset_dataframe(self):
        """Get the dataset as a pandas DataFrame."""
        return self.generator.get_dataframe()
        
    def clear_dataset(self):
        """Clear all data from the dataset."""
        from tkinter import messagebox
        
        result = messagebox.askyesno("Confirm", "Are you sure you want to clear all data?")
        
        if result:
            self.generator.clear_data()
            self.update_all_tabs()
            messagebox.showinfo("Success", "Dataset cleared!")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = QuadraticDatasetGUI(root)
    
    # Handle window closing
    def on_closing():
        if hasattr(app, 'generation_thread') and app.generation_thread and app.generation_thread.is_alive():
            from tkinter import messagebox
            result = messagebox.askyesno("Confirm", "Generation is in progress. Do you want to stop it and exit?")
            if result:
                app.stop_generation()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()
