#!/usr/bin/env python3
"""
Quadratic neural network application.

Advanced neural network analysis for quadratic equations.

This application provides comprehensive neural network analysis for quadratic equations,
including training, prediction, analysis, and comparision capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path

# add the neural engine path
current_dir = Path(__file__).parent
neural_engine_path = current_dir.parent.parent
sys.path.insert(0, str(neural_engine_path))

# import application components
try:
    from gui.main_window import QuadraticNeuralNetworkApp
except ImportError as e:
    print(f"Error importing GUI components: {e}")
    print("Please ensure all required modules are installed and paths are correct.")
    sys.exit(1)


def check_dependencies():
    """Check if all required dependancies are available"""
    required_modules = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'tkinter'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print("Missing required dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install missing dependencies using:")
        print("pip install " + " ".join(missing_modules))
        return False

    return True


def check_neural_engine():
    """Check if neural engine components are available"""
    try:
        from nn_core import NeuralNetwork
        from autodiff import TrainingEngine, Adam
        from data_utils import DataPreprocessor
        return True
    except ImportError as e:
        print(f"Neural Engine not found: {e}")
        print("Please ensure the Neural Engine is in the parent directory.")
        return False


def setup_application():
    """Setup application enviroment"""
    print("Quadratic Neural Network Application")
    print("=" * 50)
    print("Initializing application...")

    # check dependencies
    if not check_dependencies():
        return False

    # check neural engine
    if not check_neural_engine():
        return False

    print("All dependencies available")
    return True


def main():
    """Main entry point"""
    try:
        # setup application
        if not setup_application():
            input("Press Enter to exit...")
            return

        # create main window
        root = tk.Tk()

        # configure window
        root.title("Quadratic Neural Network - Advanced Analysis")
        root.geometry("1400x900")

        # set window icon (if available)
        try:
            root.iconbitmap('icon.ico')
        except:
            pass  # icon not available

        # configure styles
        style = ttk.Style()
        try:
            style.theme_use('clam')  # use modern theme
        except:
            pass  # theme not available

        # create application
        app = QuadraticNeuralNetworkApp(root)

        # show welcome message
        welcome_message = """
Welcome to Quadratic Neural Network!

This application provides comprehensive neural network analysis for quadratic equations.

Features:
• Data loading and preprocessing
• Multi-scenario neural network training
• Interactive prediction with confidence estimaton
• Advanced analysis and visualizations
• Model comparision and benchmarking

To get started:
1. Load a quadratic equation dataset (CSV format: a,b,c,x1,x2)
2. Train models for different prediction scenarios
3. Make predictions and analyze results
4. Compare model performence
        """

        messagebox.showinfo("Welcome", welcome_message)

        # start application
        print("Application started succesfully!")
        print("Close this window to exit the application.")

        # run main loop
        root.mainloop()

    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {str(e)}")

    finally:
        print("Application closed. Goodbye!")


if __name__ == "__main__":
    main()
