import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

class DataTab:
    """Data loading and preview tab"""
    
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="📊 Data")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup data tab interface"""
        # Data loading section
        self.create_loading_section()
        
        # Data preview section
        self.create_preview_section()
        
        # Statistics section
        self.create_statistics_section()
        
    def create_loading_section(self):
        """Create data loading controls"""
        load_frame = ttk.LabelFrame(self.frame, text="Data Loading")
        load_frame.pack(fill='x', padx=10, pady=5)
        
        # Load button
        ttk.Button(load_frame, text="📁 Load Dataset", 
                  command=self.load_dataset).pack(side='left', padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(load_frame, text="No data loaded", 
                                     style='Warning.TLabel')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Data info
        self.info_label = ttk.Label(load_frame, text="")
        self.info_label.pack(side='left', padx=10, pady=5)
        
    def create_preview_section(self):
        """Create data preview table"""
        preview_frame = ttk.LabelFrame(self.frame, text="Data Preview")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create main container with grid layout
        container = ttk.Frame(preview_frame)
        container.pack(fill='both', expand=True, padx=5, pady=5)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=2)
        container.grid_columnconfigure(1, weight=1)
        
        # Data table
        table_frame = ttk.Frame(container)
        table_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        columns = ['a', 'b', 'c', 'x1', 'x2']
        self.data_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Scrollbars for table
        v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.data_tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
    def create_statistics_section(self):
        """Create statistics display"""
        # Statistics text area (positioned in the preview section)
        stats_frame = ttk.Frame(self.frame.winfo_children()[-1].winfo_children()[0])  # Get container from preview
        stats_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        stats_label = ttk.Label(stats_frame, text="Dataset Statistics", font=('TkDefaultFont', 10, 'bold'))
        stats_label.pack(anchor='w', pady=(0, 5))
        
        self.stats_text = tk.Text(stats_frame, height=20, width=40, font=('Courier', 9))
        stats_scroll = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scroll.pack(side='right', fill='y')
        
    def load_dataset(self):
        """Load quadratic equation dataset"""
        filename = filedialog.askopenfilename(
            title="Select Quadratic Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            if self.app.data_processor.load_data(filename):
                self.update_preview()
                self.update_statistics()
                self.status_label.config(
                    text=f"✅ Loaded {len(self.app.data_processor.data)} equations", 
                    style='Success.TLabel'
                )
                self.info_label.config(text=f"File: {filename.split('/')[-1]}")
                
                # Refresh other tabs
                self.app.refresh_all_tabs()
                
                messagebox.showinfo("Success", 
                    f"Successfully loaded {len(self.app.data_processor.data)} quadratic equations!")
            else:
                self.status_label.config(text="❌ Failed to load data", style='Error.TLabel')
                
    def update_preview(self):
        """Update data preview table"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
            
        # Add sample data
        sample_data = self.app.data_processor.get_sample_data(100)
        
        for row in sample_data:
            formatted_row = [f"{val:.3f}" for val in row]
            self.data_tree.insert('', 'end', values=formatted_row)
            
    def update_statistics(self):
        """Update statistics display"""
        stats = self.app.data_processor.get_stats()
        
        if not stats:
            return
            
        stats_text = []
        stats_text.append("📊 DATASET OVERVIEW")
        stats_text.append("=" * 40)
        stats_text.append(f"Total Equations: {stats['total_equations']}")
        stats_text.append(f"Features: a, b, c, x1, x2")
        stats_text.append("")
        
        # Column statistics
        for name, col_stats in stats['columns'].items():
            stats_text.append(f"{name.upper()} Statistics:")
            stats_text.append(f"  Mean: {col_stats['mean']:.3f}")
            stats_text.append(f"  Std:  {col_stats['std']:.3f}")
            stats_text.append(f"  Min:  {col_stats['min']:.3f}")
            stats_text.append(f"  Max:  {col_stats['max']:.3f}")
            stats_text.append("")
            
        # Quality metrics
        if 'quality' in stats:
            stats_text.append("🔍 DATA QUALITY")
            stats_text.append("=" * 40)
            stats_text.append(f"Whole x1 solutions: {stats['quality']['x1_whole_pct']:.1f}%")
            stats_text.append(f"Whole x2 solutions: {stats['quality']['x2_whole_pct']:.1f}%")
            
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, '\n'.join(stats_text))
        
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status message"""
        styles = {
            'info': 'Info.TLabel',
            'success': 'Success.TLabel',
            'warning': 'Warning.TLabel',
            'error': 'Error.TLabel'
        }
        
        style = styles.get(status_type, 'Info.TLabel')
        self.status_label.config(text=message, style=style)
