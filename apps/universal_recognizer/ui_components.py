"""
Universal character recognition UI components.

Tkinter UI components for complete character recogntion (0-9, A-Z, a-z).
Built on proven digit recognizer UI with 62-class support.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import time
from typing import Optional, List, Tuple, Callable


class UniversalDrawingCanvas:
    """
    Enhanced drawing canvas for universal character recogntion.
    Supports drawing any character with optimized stroke handling.
    """

    def __init__(self, parent, width=400, height=400, bg_color='black', fg_color='white'):
        self.parent = parent
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.fg_color = fg_color

        # create canvas frame with purple theme
        self.canvas_frame = tk.Frame(parent, bg='#2C1810', relief='raised', bd=2)
        self.canvas_frame.pack(pady=10, padx=10)

        # canvas title
        title_label = tk.Label(
            self.canvas_frame, 
            text="Draw Any Character (0-9, A-Z, a-z)",
            font=('Arial', 14, 'bold'), 
            bg='#2C1810', 
            fg='white'
        )
        title_label.pack(pady=(10, 5))

        # main drawing canvas
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=width,
            height=height,
            bg=bg_color,
            cursor='crosshair',
            relief='sunken',
            bd=3
        )
        self.canvas.pack(padx=10, pady=5)

        # drawing state variables
        self.is_drawing = False
        self.last_x = 0
        self.last_y = 0
        self.brush_size = 15
        self.strokes = []

        # bind drawing events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # create controls
        self.create_controls()
        self.create_character_guide()

    def create_controls(self):
        """Create canvas controls."""
        controls_frame = tk.Frame(self.canvas_frame, bg='#2C1810')
        controls_frame.pack(pady=5, fill='x', padx=10)
        
        # brush size control
        brush_frame = tk.Frame(controls_frame, bg='#2C1810')
        brush_frame.pack(side='left', padx=5)
        
        tk.Label(brush_frame, text="Brush Size:", font=('Arial', 10), bg='#2C1810', fg='white').pack(side='left')
        self.brush_scale = tk.Scale(
            brush_frame,
            from_=5, to=30,
            orient='horizontal',
            command=self.update_brush_size,
            bg='#4A4A4A',
            fg='white',
            highlightbackground='#2C1810',
            length=120
        )
        self.brush_scale.set(15)
        self.brush_scale.pack(side='left', padx=5)
        
        # Create custom style for button
        style = ttk.Style()
        style.configure('Purple.TButton',
                    background='#8B4A9C',
                    foreground='white',
                    borderwidth=2,
                    focuscolor='none',
                    font=('Arial', 10, 'bold'))
        style.map('Purple.TButton',
                background=[('active', '#A366CC'), ('pressed', '#6B2A7C')])
        
        # clear button using ttk
        self.clear_btn = ttk.Button(
            controls_frame,
            text="Clear",
            command=self.clear_canvas,
            style='Purple.TButton',
            cursor='hand2'
        )
        self.clear_btn.pack(side='right', padx=5)

    def create_character_guide(self):
        """Create a comprehensive character guide."""
        guide_frame = tk.Frame(self.canvas_frame, bg='#2C1810')
        guide_frame.pack(pady=5, fill='x', padx=10)

        tk.Label(
            guide_frame, 
            text="Universal Recognition: Draw digits, uppercase, or lowercase letters",
            font=('Arial', 9), 
            bg='#2C1810', 
            fg='#B8B8B8'
        ).pack()

        # character examples
        examples_frame = tk.Frame(guide_frame, bg='#2C1810')
        examples_frame.pack(pady=2)

        tk.Label(
            examples_frame,
            text="Digits: 0 1 2 3 4 5 6 7 8 9",
            font=('Arial', 10, 'bold'),
            bg='#2C1810',
            fg='#9966CC'
        ).pack()

        tk.Label(
            examples_frame,
            text="Uppercase: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
            font=('Arial', 10, 'bold'),
            bg='#2C1810',
            fg='#9966CC'
        ).pack()

        tk.Label(
            examples_frame,
            text="Lowercase: a b c d e f g h i j k l m n o p q r s t u v w x y z",
            font=('Arial', 10, 'bold'),
            bg='#2C1810',
            fg='#9966CC'
        ).pack()

    def update_brush_size(self, value):
        """Update brush size from scale."""
        self.brush_size = int(value)

    def start_drawing(self, event):
        """Start a new drawing stroke."""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y

        # start new stroke
        current_stroke = [(self.last_x, self.last_y)]
        self.strokes.append(current_stroke)

        # draw initial point
        x1 = event.x - self.brush_size // 2
        y1 = event.y - self.brush_size // 2
        x2 = event.x + self.brush_size // 2
        y2 = event.y + self.brush_size // 2

        self.canvas.create_oval(x1, y1, x2, y2, fill=self.fg_color, outline=self.fg_color)

    def draw(self, event):
        """Continue drawing stroke."""
        if self.is_drawing:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size,
                fill=self.fg_color,
                capstyle='round',
                smooth=True
            )

            if self.strokes:
                self.strokes[-1].append((event.x, event.y))

            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        """Stop current drawing stroke."""
        self.is_drawing = False

    def clear_canvas(self):
        """Clear the drawing canvas."""
        self.canvas.delete('all')
        self.strokes = []

    def get_image_data(self):
        """Get the drawn image as numpy array for neural network."""
        # create PIL image from canvas
        img = Image.new('RGB', (self.width, self.height), 'black')
        draw = ImageDraw.Draw(img)

        # recreate drawing on PIL image
        for stroke in self.strokes:
            if len(stroke) > 1:
                for i in range(len(stroke) - 1):
                    x1, y1 = stroke[i]
                    x2, y2 = stroke[i + 1]
                    draw.line([x1, y1, x2, y2], fill='white', width=self.brush_size)
            elif len(stroke) == 1:
                x, y = stroke[0]
                r = self.brush_size // 2
                draw.ellipse([x-r, y-r, x+r, y+r], fill='white')

        # resize to 28x28 for neural network
        img = img.convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 255.0

        return img_array


class UniversalPredictionDisplay:
    """
    Display prediction results for universal character recogntion (62 classes).
    Organized by character type with smart grouping and filtering.
    """

    def __init__(self, parent):
        self.parent = parent

        # main prediction frame
        self.prediction_frame = tk.Frame(parent, bg='#2C1810', relief='raised', bd=2)
        self.prediction_frame.pack(pady=10, padx=10, fill='both', expand=True)

        # title
        title_label = tk.Label(
            self.prediction_frame,
            text="Universal Character Recognition Results",
            font=('Arial', 16, 'bold'),
            bg='#2C1810',
            fg='white'
        )
        title_label.pack(pady=(15, 10))

        # create display components
        self.create_top_prediction_display()
        self.create_tabbed_confidence_display()
        self.create_metrics_display()

    def create_top_prediction_display(self):
        """Create the main prediction display."""
        top_frame = tk.Frame(self.prediction_frame, bg='#2C1810')
        top_frame.pack(pady=10, fill='x', padx=20)

        # predicted character (large display)
        self.predicted_char_label = tk.Label(
            top_frame,
            text="?",
            font=('Arial', 48, 'bold'),
            bg='#2C1810',
            fg='#9966CC'
        )
        self.predicted_char_label.pack(pady=10)

        # character type and confidence
        self.type_confidence_frame = tk.Frame(top_frame, bg='#2C1810')
        self.type_confidence_frame.pack()

        self.char_type_label = tk.Label(
            self.type_confidence_frame,
            text="Character Type: --",
            font=('Arial', 12, 'bold'),
            bg='#2C1810',
            fg='white'
        )
        self.char_type_label.pack()

        self.confidence_label = tk.Label(
            self.type_confidence_frame,
            text="Confidence: --%",
            font=('Arial', 14, 'bold'),
            bg='#2C1810',
            fg='white'
        )
        self.confidence_label.pack()

        self.time_label = tk.Label(
            self.type_confidence_frame,
            text="Prediction time: -- ms",
            font=('Arial', 10),
            bg='#2C1810',
            fg='#B8B8B8'
        )
        self.time_label.pack(pady=5)

    def create_tabbed_confidence_display(self):
        """Create tabbed display for organized confidence viewing."""
        # notebook for tabs
        style = ttk.Style()
        style.theme_use('default')

        self.notebook = ttk.Notebook(self.prediction_frame)
        self.notebook.pack(pady=15, fill='both', expand=True, padx=20)

        # create tabs
        self.create_digits_tab()
        self.create_uppercase_tab()
        self.create_lowercase_tab()
        self.create_all_tab()

    def create_digits_tab(self):
        """Create digits confidence tab (0-9)."""
        digits_frame = tk.Frame(self.notebook, bg='#2C1810')
        self.notebook.add(digits_frame, text='Digits (0-9)')

        self.digits_bars = self.create_confidence_section(digits_frame, range(10))

    def create_uppercase_tab(self):
        """Create uppercase letters tab (A-Z)."""
        upper_frame = tk.Frame(self.notebook, bg='#2C1810')
        self.notebook.add(upper_frame, text='Uppercase (A-Z)')

        self.upper_bars = self.create_confidence_section(upper_frame, range(10, 36))

    def create_lowercase_tab(self):
        """Create lowercase letters tab (a-z)."""
        lower_frame = tk.Frame(self.notebook, bg='#2C1810')
        self.notebook.add(lower_frame, text='Lowercase (a-z)')

        self.lower_bars = self.create_confidence_section(lower_frame, range(36, 62))

    def create_all_tab(self):
        """Create tab showing top predictions from all categories."""
        all_frame = tk.Frame(self.notebook, bg='#2C1810')
        self.notebook.add(all_frame, text='Top Predictions')

        # top 10 predictions display
        tk.Label(
            all_frame,
            text="Top 10 Predictions Across All Characters",
            font=('Arial', 12, 'bold'),
            bg='#2C1810',
            fg='white'
        ).pack(pady=10)

        self.top_predictions_frame = tk.Frame(all_frame, bg='#2C1810')
        self.top_predictions_frame.pack(fill='both', expand=True, padx=10)

    def create_confidence_section(self, parent, indices):
        """Create confidence bars for a specific character range."""
        # scrollable frame
        canvas = tk.Canvas(parent, bg='#2C1810', height=300)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2C1810')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # store confidence elements
        confidence_bars = {}
        confidence_labels = {}

        from load_data import index_to_character

        for i in indices:
            char = index_to_character(i)

            # create frame for each character
            char_frame = tk.Frame(scrollable_frame, bg='#2C1810')
            char_frame.pack(fill='x', pady=2, padx=10)

            # character label
            char_label = tk.Label(
                char_frame,
                text=f"{char}:",
                font=('Arial', 11, 'bold'),
                bg='#2C1810',
                fg='white',
                width=4
            )
            char_label.pack(side='left', padx=(0, 10))

            # progress bar frame
            bar_frame = tk.Frame(char_frame, bg='#1A1A1A', height=20, relief='sunken', bd=1)
            bar_frame.pack(side='left', fill='x', expand=True, padx=5)

            # actual progress bar
            progress_bar = tk.Frame(bar_frame, bg='#9966CC', height=18)
            progress_bar.place(x=0, y=0, width=0, height=18)
            confidence_bars[i] = progress_bar

            # percentage label
            percentage_label = tk.Label(
                char_frame,
                text="0%",
                font=('Arial', 9),
                bg='#2C1810',
                fg='#B8B8B8',
                width=6
            )
            percentage_label.pack(side='right', padx=(10, 0))
            confidence_labels[i] = percentage_label

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return {'bars': confidence_bars, 'labels': confidence_labels}

    def create_metrics_display(self):
        """Create metrics display for additional informaton."""
        metrics_frame = tk.Frame(self.prediction_frame, bg='#2C1810')
        metrics_frame.pack(pady=10, fill='x', padx=20)

        tk.Label(
            metrics_frame,
            text="Recognition Metrics",
            font=('Arial', 11, 'bold'),
            bg='#2C1810',
            fg='white'
        ).pack()

        self.metrics_text = tk.Text(
            metrics_frame,
            height=4,
            bg='#1A1A1A',
            fg='#B8B8B8',
            font=('Arial', 9),
            relief='sunken',
            bd=1,
            state='disabled'
        )
        self.metrics_text.pack(fill='x', pady=5)

    def update_prediction(self, predicted_char: str, confidence: float, 
                         all_confidences: List[float], prediction_time: float):
        """Update the prediction display with new results."""
        from load_data import character_to_index, get_character_type

        # update top prediction
        self.predicted_char_label.config(text=predicted_char)

        char_index = character_to_index(predicted_char)
        char_type = get_character_type(char_index) if char_index >= 0 else "Unknown"

        self.char_type_label.config(text=f"Character Type: {char_type}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        self.time_label.config(text=f"Prediction time: {prediction_time:.1f} ms")

        # color code confidence
        if confidence >= 80:
            color = '#4CAF50'  # green
        elif confidence >= 60:
            color = '#FF9800'  # orange
        else:
            color = '#F44336'  # red

        self.predicted_char_label.config(fg=color)

        # update all confidence bars
        self.update_all_confidence_bars(all_confidences, predicted_char)

        # update top predictions
        self.update_top_predictions(all_confidences)

        # update metrics
        self.update_metrics(all_confidences, prediction_time, char_type)

    def update_all_confidence_bars(self, confidences: List[float], predicted_char: str):
        """Update confidence bars across all tabs."""
        from load_data import index_to_character, character_to_index

        predicted_index = character_to_index(predicted_char)
        bar_frame_width = 200

        # update digits bars (0-9)
        for i in range(10):
            if i in self.digits_bars['bars']:
                confidence_pct = confidences[i] * 100
                bar_width = int((confidence_pct / 100) * bar_frame_width)

                self.digits_bars['bars'][i].config(width=bar_width)
                self.digits_bars['labels'][i].config(text=f"{confidence_pct:.1f}%")

                # highlight predicted character
                if i == predicted_index:
                    self.digits_bars['bars'][i].config(bg='#FF6B6B')
                    self.digits_bars['labels'][i].config(fg='white', font=('Arial', 9, 'bold'))
                else:
                    self.digits_bars['bars'][i].config(bg='#9966CC')
                    self.digits_bars['labels'][i].config(fg='#B8B8B8', font=('Arial', 9))

        # update uppercase bars (A-Z)
        for i in range(10, 36):
            if i in self.upper_bars['bars']:
                confidence_pct = confidences[i] * 100
                bar_width = int((confidence_pct / 100) * bar_frame_width)

                self.upper_bars['bars'][i].config(width=bar_width)
                self.upper_bars['labels'][i].config(text=f"{confidence_pct:.1f}%")

                if i == predicted_index:
                    self.upper_bars['bars'][i].config(bg='#FF6B6B')
                    self.upper_bars['labels'][i].config(fg='white', font=('Arial', 9, 'bold'))
                else:
                    self.upper_bars['bars'][i].config(bg='#9966CC')
                    self.upper_bars['labels'][i].config(fg='#B8B8B8', font=('Arial', 9))

        # update lowercase bars (a-z)
        for i in range(36, 62):
            if i in self.lower_bars['bars']:
                confidence_pct = confidences[i] * 100
                bar_width = int((confidence_pct / 100) * bar_frame_width)

                self.lower_bars['bars'][i].config(width=bar_width)
                self.lower_bars['labels'][i].config(text=f"{confidence_pct:.1f}%")

                if i == predicted_index:
                    self.lower_bars['bars'][i].config(bg='#FF6B6B')
                    self.lower_bars['labels'][i].config(fg='white', font=('Arial', 9, 'bold'))
                else:
                    self.lower_bars['bars'][i].config(bg='#9966CC')
                    self.lower_bars['labels'][i].config(fg='#B8B8B8', font=('Arial', 9))

    def update_top_predictions(self, confidences: List[float]):
        """Update the top predictions display."""
        from load_data import index_to_character, get_character_type

        # clear previous top predictions
        for widget in self.top_predictions_frame.winfo_children():
            widget.destroy()

        # get top 10 predictions
        top_indices = np.argsort(confidences)[-10:][::-1]

        for rank, idx in enumerate(top_indices):
            char = index_to_character(idx)
            confidence = confidences[idx] * 100
            char_type = get_character_type(idx)

            # create frame for this prediction
            pred_frame = tk.Frame(self.top_predictions_frame, bg='#2C1810')
            pred_frame.pack(fill='x', pady=1, padx=10)

            # rank
            rank_label = tk.Label(
                pred_frame,
                text=f"#{rank+1}",
                font=('Arial', 10, 'bold'),
                bg='#2C1810',
                fg='#9966CC',
                width=4
            )
            rank_label.pack(side='left')

            # character
            char_label = tk.Label(
                pred_frame,
                text=f"'{char}'",
                font=('Arial', 12, 'bold'),
                bg='#2C1810',
                fg='white',
                width=6
            )
            char_label.pack(side='left', padx=5)

            # type
            type_label = tk.Label(
                pred_frame,
                text=char_type,
                font=('Arial', 9),
                bg='#2C1810',
                fg='#B8B8B8',
                width=10
            )
            type_label.pack(side='left', padx=5)

            # confidence
            conf_label = tk.Label(
                pred_frame,
                text=f"{confidence:.1f}%",
                font=('Arial', 10, 'bold'),
                bg='#2C1810',
                fg='#4CAF50' if rank == 0 else '#B8B8B8'
            )
            conf_label.pack(side='right', padx=5)

    def update_metrics(self, confidences: List[float], prediction_time: float, char_type: str):
        """Update the metrics display."""
        avg_confidence = np.mean(confidences) * 100
        max_confidence = np.max(confidences) * 100
        std_confidence = np.std(confidences) * 100

        # character type specific metrics
        if char_type == "Digit":
            type_confidences = confidences[:10]
        elif char_type == "Uppercase":
            type_confidences = confidences[10:36]
        elif char_type == "Lowercase":
            type_confidences = confidences[36:62]
        else:
            type_confidences = confidences

        type_avg = np.mean(type_confidences) * 100

        metrics_text = f"""Overall Average: {avg_confidence:.1f}% | {char_type} Average: {type_avg:.1f}%
Maximum Confidence: {max_confidence:.1f}% | Confidence Std Dev: {std_confidence:.1f}%
Character Type: {char_type} | Processing Speed: {1000/prediction_time:.1f} chars/sec"""

        self.metrics_text.config(state='normal')
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
        self.metrics_text.config(state='disabled')

    def reset_display(self):
        """Reset the prediction display."""
        self.predicted_char_label.config(text="?", fg='#9966CC')
        self.char_type_label.config(text="Character Type: --")
        self.confidence_label.config(text="Confidence: --%")
        self.time_label.config(text="Prediction time: -- ms")

        # reset all confidence bars
        for bars_dict in [self.digits_bars, self.upper_bars, self.lower_bars]:
            for bar in bars_dict['bars'].values():
                bar.config(width=0, bg='#9966CC')
            for label in bars_dict['labels'].values():
                label.config(text="0%", fg='#B8B8B8', font=('Arial', 9))

        # clear top predictions
        for widget in self.top_predictions_frame.winfo_children():
            widget.destroy()

        # clear metrics
        self.metrics_text.config(state='normal')
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.config(state='disabled')