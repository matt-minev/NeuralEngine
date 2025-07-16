import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any

from config.scenarios import get_default_scenarios
from core.data_processor import QuadraticDataProcessor
from core.predictor import QuadraticPredictor
from gui.tabs.data_tab import DataTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.prediction_tab import PredictionTab
from gui.tabs.analysis_tab import AnalysisTab
from gui.tabs.comparison_tab import ComparisonTab

class QuadraticNeuralNetworkApp:
    """Main application window controller"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Quadratic Neural Network - Advanced Analysis")
        self.root.geometry("1400x900")
        
        # Shared application state
        self.scenarios = get_default_scenarios()
        self.data_processor = QuadraticDataProcessor(verbose=True)
        self.predictors = {}
        self.results = {}
        
        # Initialize GUI
        self.setup_styles()
        self.setup_ui()
        
    def setup_styles(self):
        """Configure custom styles"""
        style = ttk.Style()
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Info.TLabel', foreground='blue')
        
    def setup_ui(self):
        """Setup main user interface"""
        # Create main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize tabs
        self.data_tab = DataTab(self.notebook, self)
        self.training_tab = TrainingTab(self.notebook, self)
        self.prediction_tab = PredictionTab(self.notebook, self)
        self.analysis_tab = AnalysisTab(self.notebook, self)
        self.comparison_tab = ComparisonTab(self.notebook, self)
        
    def update_status(self, message: str, status_type: str = 'info'):
        """Update application status across all tabs"""
        for tab in [self.data_tab, self.training_tab, self.prediction_tab, 
                   self.analysis_tab, self.comparison_tab]:
            if hasattr(tab, 'update_status'):
                tab.update_status(message, status_type)
                
    def refresh_all_tabs(self):
        """Refresh all tabs after data changes"""
        for tab in [self.training_tab, self.prediction_tab, 
                   self.analysis_tab, self.comparison_tab]:
            if hasattr(tab, 'refresh'):
                tab.refresh()
