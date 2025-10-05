"""
Universal character recognizer - results showcase.

Professional presentation of neural network performence and capabilities.
Displays comprehensive test results without user interaction.
"""

import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk


class ResultsShowcase:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.create_showcase_interface()

    def setup_window(self):
        self.root.title("NeuralEngine Universal Character Recognizer - Results Showcase")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1A1A1A')

    def create_showcase_interface(self):
        # header section
        header_frame = tk.Frame(self.root, bg='#1A1A1A')
        header_frame.pack(fill='x', pady=20)

        title_label = tk.Label(
            header_frame,
            text="NeuralEngine Universal Character Recognizer",
            font=('Arial', 24, 'bold'),
            bg='#1A1A1A',
            fg='white'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Professional OCR System • 62 Character Classes • Custom Neural Architecture",
            font=('Arial', 14),
            bg='#1A1A1A',
            fg='#B8B8B8'
        )
        subtitle_label.pack(pady=(5, 0))

        # results grid section
        results_frame = tk.Frame(self.root, bg='#1A1A1A')
        results_frame.pack(fill='both', expand=True, padx=20)

        # performance metrics display
        self.create_metrics_section(results_frame)

        # load and display test visualizations
        self.create_visualizations_section(results_frame)

    def create_metrics_section(self, parent):
        metrics_frame = tk.LabelFrame(
            parent,
            text="Performance Achievements",
            font=('Arial', 16, 'bold'),
            bg='#2C1810',
            fg='white',
            bd=2,
            relief='raised'
        )
        metrics_frame.pack(fill='x', pady=10)

        # key metrics to display
        metrics = [
            ("Overall Accuracy", "79.11%", "62-class classification"),
            ("Digits Accuracy", "91.7%", "Professional-grade performence"),
            ("vs Random Baseline", "49.1x", "Improvement factor"),
            ("Average Confidence", "76.0%", "High prediction certainty"),
            ("Architecture", "784->512->256->128->62", "Custom NeuralEngine design"),
            ("Parameters", "574,142", "Optimized model size")
        ]

        for i, (metric, value, desc) in enumerate(metrics):
            row = i // 3
            col = i % 3

            metric_frame = tk.Frame(metrics_frame, bg='#2C1810')
            metric_frame.grid(row=row, column=col, padx=10, pady=10, sticky='ew')

            # value display in purple
            tk.Label(
                metric_frame,
                text=value,
                font=('Arial', 18, 'bold'),
                bg='#2C1810',
                fg='#9966CC'
            ).pack()

            # metric name
            tk.Label(
                metric_frame,
                text=metric,
                font=('Arial', 12, 'bold'),
                bg='#2C1810',
                fg='white'
            ).pack()

            # description text
            tk.Label(
                metric_frame,
                text=desc,
                font=('Arial', 10),
                bg='#2C1810',
                fg='#B8B8B8'
            ).pack()

        # make columns equal width
        for i in range(3):
            metrics_frame.grid_columnconfigure(i, weight=1)

    def create_visualizations_section(self, parent):
        viz_frame = tk.LabelFrame(
            parent,
            text="Comprehensive Test Results",
            font=('Arial', 16, 'bold'),
            bg='#2C1810',
            fg='white',
            bd=2,
            relief='raised'
        )
        viz_frame.pack(fill='both', expand=True, pady=10)

        # check if visualization files exist
        viz_files = [
            'test_results/character_type_performance.png',
            'test_results/worst_characters.png',
            'test_results/universal_confidence_analysis.png'
        ]

        if all(os.path.exists(f) for f in viz_files):
            self.display_visualization_grid(viz_frame, viz_files)
        else:
            self.display_placeholder_results(viz_frame)

    def display_visualization_grid(self, parent, viz_files):
        # create notebook for multiple charts
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        for viz_file in viz_files:
            try:
                # load and resize image
                img = Image.open(viz_file)
                img = img.resize((800, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                frame = tk.Frame(notebook, bg='white')
                label = tk.Label(frame, image=photo, bg='white')
                label.image = photo  # keep reference to avoid garbage colection
                label.pack(pady=10)

                # create tab name from filename
                tab_name = os.path.basename(viz_file).replace('.png', '').replace('_', ' ').title()
                notebook.add(frame, text=tab_name)

            except Exception as e:
                print(f"Could not load {viz_file}: {e}")

    def display_placeholder_results(self, parent):
        placeholder_frame = tk.Frame(parent, bg='#2C1810')
        placeholder_frame.pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(
            placeholder_frame,
            text="Test Results Summary",
            font=('Arial', 18, 'bold'),
            bg='#2C1810',
            fg='white'
        ).pack(pady=20)

        results_text = """
UNIVERSAL CHARACTER RECOGNITION ACHIEVEMENTS:

Successfully trained on EMNIST ByClass dataset (814,255 samples)
Achieved 79.11% accuracy across 62 character classes
Outperformed random baseline by 49.1x improvement factor
Demonstrated strong digit recogntion (91.7% accuracy)
Built custom NeuralEngine architecture from scratch
Implemented professional OCR pipeline with preprocessing
Created comprehensive testing and visualization suite

CHARACTER TYPE PERFORMANCE:
• Digits (0-9): 91.7% accuracy - Excellent performance
• Uppercase (A-Z): 70.7% accuracy - Good performance  
• Lowercase (a-z): 61.8% accuracy - Room for improvement

TECHNICAL ACHIEVEMENTS:
• Custom neural network implementation
• Multi-phase training strategy
• Comprehensive evaluation framework
• Professional visualization system

This project demonstrates advanced machine learning engineering
with custom implementation of neural network components.
        """

        text_widget = tk.Text(
            placeholder_frame,
            font=('Courier', 11),
            bg='#1A1A1A',
            fg='#B8B8B8',
            wrap='word',
            state='normal'
        )
        text_widget.pack(fill='both', expand=True)
        text_widget.insert('1.0', results_text)
        text_widget.config(state='disabled')

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    showcase = ResultsShowcase()
    showcase.run()
