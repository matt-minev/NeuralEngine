"""
Digit Recognizer UI Components - Interactive Elements for Digit Recognition
========================================================================

This module provides professional UI components specifically designed for
the digit recognizer application:

- DrawingCanvas: Smooth digit drawing interface with 28x28 output
- PredictionDisplay: Real-time digit prediction with confidence bars
- TrainingMonitor: Live training progress for digit recognition models

Features:
- Beautiful purple theme matching modern UI trends
- Tkinter-based for cross-platform compatibility
- Optimized for real-time neural network feedback
- Professional appearance with smooth animations
- Educational focus with clear visual feedback
"""

import tkinter as tk
from tkinter import ttk, Canvas, Frame, Label, Button, Scale, StringVar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import sys
import os
from typing import Callable, Optional, List, Dict, Tuple, Any
from collections import deque

# Add parent directory to path to import NeuralEngine components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class DrawingCanvas:
    """
    Professional drawing canvas for digit input with purple theme.
    
    Features:
    - Smooth drawing optimized for digit recognition
    - Real-time 28x28 array conversion for neural networks
    - Purple-themed interface for modern appearance
    - Configurable brush sizes for different drawing styles
    - Export to neural network compatible format
    - Anti-aliased rendering for better visual quality
    """
    
    def __init__(self, parent, width: int = 280, height: int = 280, 
                 bg_color: str = "black", brush_color: str = "white",
                 brush_size: int = 20, on_draw_callback: Optional[Callable] = None):
        """
        Initialize digit drawing canvas.
        
        Args:
            parent: Parent tkinter widget
            width: Canvas width in pixels (optimized for 28x28 digit recognition)
            height: Canvas height in pixels
            bg_color: Background color (black for digit recognition)
            brush_color: Drawing brush color (white for digit contrast)
            brush_size: Brush size in pixels
            on_draw_callback: Function called when drawing occurs for real-time prediction
        """
        self.parent = parent
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.brush_color = brush_color
        self.brush_size = brush_size
        self.on_draw_callback = on_draw_callback
        
        # Drawing state
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create UI elements with purple theme
        self._create_widgets()
        self._bind_events()
        
        # Drawing data for neural network (28x28 for MNIST compatibility)
        self.drawing_data = np.zeros((28, 28), dtype=np.float32)
        
    def _create_widgets(self):
        """Create the digit drawing interface with purple theme."""
        # Main frame with deep purple background
        self.frame = Frame(self.parent, bg="#4a148c", relief=tk.RAISED, borderwidth=2)
        
        # Title with digit-specific text
        title_label = Label(self.frame, text="✏️ Draw a Digit (0-9)", 
                           font=("Arial", 14, "bold"), 
                           bg="#4a148c", fg="white")
        title_label.pack(pady=(10, 5))
        
        # Canvas frame with medium purple border
        canvas_frame = Frame(self.frame, bg="#7b1fa2", relief=tk.SUNKEN, borderwidth=3)
        canvas_frame.pack(padx=20, pady=10)
        
        # Drawing canvas optimized for digits
        self.canvas = Canvas(canvas_frame, 
                           width=self.width, 
                           height=self.height,
                           bg=self.bg_color,
                           cursor="crosshair")
        self.canvas.pack()
        
        # Controls frame
        controls_frame = Frame(self.frame, bg="#4a148c")
        controls_frame.pack(pady=10)
        
        # Brush size control with purple styling
        brush_frame = Frame(controls_frame, bg="#4a148c")
        brush_frame.pack(side=tk.LEFT, padx=10)
        
        Label(brush_frame, text="Brush Size:", 
              font=("Arial", 10), bg="#4a148c", fg="white").pack()
        
        self.brush_scale = Scale(brush_frame, from_=5, to=40, 
                                orient=tk.HORIZONTAL, 
                                bg="#7b1fa2", fg="white",
                                troughcolor="#9c27b0",
                                command=self._update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack()
        
        # Clear button with purple accent
        self.clear_button = Button(controls_frame, text="🗑️ Clear", 
                                  font=("Arial", 12, "bold"),
                                  bg="#9c27b0", fg="white",
                                  activebackground="#ab47bc",
                                  relief=tk.RAISED, borderwidth=3,
                                  command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10)
        
        # Status label with light purple text
        self.status_var = StringVar(value="Ready to draw a digit...")
        self.status_label = Label(self.frame, textvariable=self.status_var,
                                 font=("Arial", 10), 
                                 bg="#4a148c", fg="#e1bee7")
        self.status_label.pack(pady=(0, 10))
    
    def _bind_events(self):
        """Bind mouse events for digit drawing."""
        self.canvas.bind("<Button-1>", self._start_drawing)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drawing)
        
        # Touch support for tablets (useful for digit drawing)
        self.canvas.bind("<Motion>", self._on_hover)
    
    def _start_drawing(self, event):
        """Start digit drawing operation."""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.status_var.set("Drawing digit...")
        
        # Draw initial point
        self._draw_point(event.x, event.y)
    
    def _draw(self, event):
        """Continue digit drawing operation."""
        if self.is_drawing and self.last_x and self.last_y:
            # Draw smooth line from last position to current
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size, 
                fill=self.brush_color,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Update drawing data for neural network
            self._update_drawing_data(event.x, event.y)
            
            # Update position
            self.last_x = event.x
            self.last_y = event.y
            
            # Callback for real-time digit prediction
            if self.on_draw_callback:
                threading.Thread(target=self._safe_callback, daemon=True).start()
    
    def _stop_drawing(self, event):
        """Stop digit drawing operation."""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        self.status_var.set("Digit complete! Predicting...")
        
        # Final callback for digit prediction
        if self.on_draw_callback:
            threading.Thread(target=self._safe_callback, daemon=True).start()
    
    def _on_hover(self, event):
        """Update coordinates display on hover."""
        if not self.is_drawing:
            self.status_var.set(f"Position: ({event.x}, {event.y}) - Ready to draw")
    
    def _draw_point(self, x, y):
        """Draw a single point for digit."""
        radius = self.brush_size // 2
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill=self.brush_color, outline=self.brush_color
        )
        self._update_drawing_data(x, y)
    
    def _update_drawing_data(self, x, y):
        """Update the 28x28 array for MNIST-compatible digit recognition."""
        # Convert canvas coordinates to 28x28 grid (MNIST standard)
        grid_x = int((x / self.width) * 28)
        grid_y = int((y / self.height) * 28)
        
        # Ensure coordinates are within bounds
        grid_x = max(0, min(27, grid_x))
        grid_y = max(0, min(27, grid_y))
        
        # Add brush effect (small area around point for better digit quality)
        brush_radius = max(1, self.brush_size // 20)
        for dx in range(-brush_radius, brush_radius + 1):
            for dy in range(-brush_radius, brush_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < 28 and 0 <= ny < 28:
                    # Use Gaussian-like falloff for smoother digit edges
                    distance = np.sqrt(dx*dx + dy*dy)
                    intensity = np.exp(-distance / brush_radius) if brush_radius > 0 else 1.0
                    self.drawing_data[ny, nx] = min(1.0, self.drawing_data[ny, nx] + intensity)
    
    def _update_brush_size(self, value):
        """Update brush size from scale widget."""
        self.brush_size = int(value)
    
    def _safe_callback(self):
        """Safely execute digit prediction callback in separate thread."""
        try:
            if self.on_draw_callback:
                self.on_draw_callback(self.get_drawing_array())
        except Exception as e:
            print(f"Digit prediction callback error: {e}")
    
    def clear_canvas(self):
        """Clear the digit drawing canvas."""
        self.canvas.delete("all")
        self.drawing_data = np.zeros((28, 28), dtype=np.float32)
        self.status_var.set("Canvas cleared. Ready to draw a new digit...")
        
        # Callback with empty drawing
        if self.on_draw_callback:
            threading.Thread(target=self._safe_callback, daemon=True).start()
    
    def get_drawing_array(self) -> np.ndarray:
        """
        Get the current digit drawing as a 28x28 numpy array.
        
        Returns:
            28x28 array normalized to [0, 1] range, ready for MNIST-trained models
        """
        return self.drawing_data.copy()
    
    def get_flattened_array(self) -> np.ndarray:
        """
        Get the current digit drawing as a flattened 784-element array.
        
        Returns:
            784-element array ready for neural network input (28*28=784)
        """
        return self.drawing_data.flatten()
    
    def pack(self, **kwargs):
        """Pack the drawing canvas frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the drawing canvas frame."""
        self.frame.grid(**kwargs)


class PredictionDisplay:
    """
    Real-time digit prediction display with purple theme.
    
    Features:
    - Live confidence bars for each digit (0-9)
    - Highlighted top prediction with large display
    - Purple-themed modern interface
    - Performance metrics for prediction speed
    - Smooth animations and color transitions
    """
    
    def __init__(self, parent, width: int = 300, height: int = 400):
        """
        Initialize digit prediction display.
        
        Args:
            parent: Parent tkinter widget
            width: Display width
            height: Display height
        """
        self.parent = parent
        self.width = width
        self.height = height
        
        # Prediction data for digits 0-9
        self.predictions = np.zeros(10)
        self.prediction_history = deque(maxlen=50)
        
        # Create UI elements with purple theme
        self._create_widgets()
        
        # Animation state
        self.animation_running = False
    
    def _create_widgets(self):
        """Create the digit prediction display interface with purple theme."""
        # Main frame with deep purple background
        self.frame = Frame(self.parent, bg="#4a148c", relief=tk.RAISED, borderwidth=2)
        
        # Title
        title_label = Label(self.frame, text="🧠 Digit Prediction", 
                           font=("Arial", 14, "bold"), 
                           bg="#4a148c", fg="white")
        title_label.pack(pady=(10, 5))
        
        # Top prediction display with medium purple background
        self.top_prediction_frame = Frame(self.frame, bg="#7b1fa2", relief=tk.SUNKEN, borderwidth=3)
        self.top_prediction_frame.pack(padx=20, pady=10, fill=tk.X)
        
        self.top_prediction_var = StringVar(value="?")
        self.top_prediction_label = Label(self.top_prediction_frame, 
                                         textvariable=self.top_prediction_var,
                                         font=("Arial", 48, "bold"),
                                         bg="#7b1fa2", fg="#e91e63")  # Pink accent for predictions
        self.top_prediction_label.pack(pady=20)
        
        self.confidence_var = StringVar(value="Confidence: --%")
        self.confidence_label = Label(self.top_prediction_frame,
                                     textvariable=self.confidence_var,
                                     font=("Arial", 12),
                                     bg="#7b1fa2", fg="white")
        self.confidence_label.pack(pady=(0, 10))
        
        # Confidence bars frame for digits 0-9
        bars_frame = Frame(self.frame, bg="#4a148c")
        bars_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        Label(bars_frame, text="Digit Confidence Levels:", 
              font=("Arial", 12, "bold"), 
              bg="#4a148c", fg="white").pack(pady=(0, 5))
        
        # Create confidence bars for each digit (0-9)
        self.confidence_bars = {}
        self.confidence_labels = {}
        
        for digit in range(10):
            # Bar frame for each digit
            bar_frame = Frame(bars_frame, bg="#4a148c")
            bar_frame.pack(fill=tk.X, pady=2)
            
            # Digit label
            digit_label = Label(bar_frame, text=f"{digit}:", 
                               font=("Arial", 10, "bold"),
                               bg="#4a148c", fg="white", width=3)
            digit_label.pack(side=tk.LEFT)
            
            # Progress bar background
            progress_frame = Frame(bar_frame, bg="#7b1fa2", height=20, relief=tk.SUNKEN, borderwidth=1)
            progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Confidence bar with purple gradient
            confidence_bar = Frame(progress_frame, bg="#9c27b0", height=18)
            self.confidence_bars[digit] = confidence_bar
            
            # Percentage label
            percent_label = Label(bar_frame, text="0%", 
                                 font=("Arial", 9),
                                 bg="#4a148c", fg="#e1bee7", width=5)
            percent_label.pack(side=tk.RIGHT)
            self.confidence_labels[digit] = percent_label
        
        # Performance metrics with purple styling
        metrics_frame = Frame(self.frame, bg="#4a148c")
        metrics_frame.pack(pady=10)
        
        self.prediction_time_var = StringVar(value="Prediction Time: --ms")
        prediction_time_label = Label(metrics_frame, textvariable=self.prediction_time_var,
                                     font=("Arial", 10), bg="#4a148c", fg="#e1bee7")
        prediction_time_label.pack()
        
        self.total_predictions_var = StringVar(value="Total Predictions: 0")
        total_predictions_label = Label(metrics_frame, textvariable=self.total_predictions_var,
                                       font=("Arial", 10), bg="#4a148c", fg="#e1bee7")
        total_predictions_label.pack()
    
    def update_predictions(self, predictions: np.ndarray, prediction_time: float = 0.0):
        """
        Update the digit prediction display with new results.
        
        Args:
            predictions: Array of 10 confidence values (0-1) for digits 0-9
            prediction_time: Time taken for prediction in milliseconds
        """
        self.predictions = np.array(predictions)
        
        # Store in history for analysis
        self.prediction_history.append(predictions.copy())
        
        # Update top prediction
        top_digit = np.argmax(predictions)
        confidence = predictions[top_digit] * 100
        
        self.top_prediction_var.set(str(top_digit))
        self.confidence_var.set(f"Confidence: {confidence:.1f}%")
        
        # Update color based on confidence level
        if confidence > 80:
            color = "#4caf50"  # Green for high confidence
        elif confidence > 60:
            color = "#ff9800"  # Orange for medium confidence  
        elif confidence > 40:
            color = "#ffc107"  # Yellow for low-medium confidence
        else:
            color = "#f44336"  # Red for low confidence
        
        self.top_prediction_label.config(fg=color)
        
        # Update confidence bars for all digits
        for digit in range(10):
            confidence = predictions[digit]
            
            # Update bar width based on confidence
            bar_width = confidence  # 0 to 1 scale
            self.confidence_bars[digit].place(relwidth=bar_width, relheight=1)
            
            # Update percentage label
            self.confidence_labels[digit].config(text=f"{confidence*100:.1f}%")
            
            # Color coding based on prediction strength
            if digit == top_digit:
                self.confidence_bars[digit].config(bg="#e91e63")  # Pink for top prediction
            elif confidence > 0.1:
                self.confidence_bars[digit].config(bg="#9c27b0")  # Purple for significant
            else:
                self.confidence_bars[digit].config(bg="#ce93d8")  # Light purple for low
        
        # Update performance metrics
        self.prediction_time_var.set(f"Prediction Time: {prediction_time:.1f}ms")
        self.total_predictions_var.set(f"Total Predictions: {len(self.prediction_history)}")
    
    def clear_predictions(self):
        """Clear all digit predictions and reset display."""
        self.predictions = np.zeros(10)
        self.prediction_history.clear()
        
        self.top_prediction_var.set("?")
        self.confidence_var.set("Confidence: --%")
        self.top_prediction_label.config(fg="#e1bee7")
        
        # Reset all digit confidence bars
        for digit in range(10):
            self.confidence_bars[digit].place(relwidth=0, relheight=1)
            self.confidence_labels[digit].config(text="0%")
            self.confidence_bars[digit].config(bg="#ce93d8")
        
        self.prediction_time_var.set("Prediction Time: --ms")
        self.total_predictions_var.set("Total Predictions: 0")
    
    def pack(self, **kwargs):
        """Pack the prediction display frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the prediction display frame."""
        self.frame.grid(**kwargs)


class TrainingMonitor:
    """
    Live training progress monitor for digit recognition models with purple theme.
    
    Features:
    - Real-time loss plotting for digit recognition training
    - Training metrics display (accuracy, loss, time)
    - Purple-themed progress bars and styling
    - Performance monitoring optimized for digit classification
    """
    
    def __init__(self, parent, width: int = 500, height: int = 300):
        """
        Initialize digit recognition training monitor.
        
        Args:
            parent: Parent tkinter widget
            width: Monitor width
            height: Monitor height
        """
        self.parent = parent
        self.width = width
        self.height = height
        
        # Training data for digit recognition
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.current_epoch = 0
        self.total_epochs = 0
        
        # Create UI elements with purple theme
        self._create_widgets()
        self._create_plot()
    
    def _create_widgets(self):
        """Create the digit training monitor interface with purple theme."""
        # Main frame with deep purple background
        self.frame = Frame(self.parent, bg="#4a148c", relief=tk.RAISED, borderwidth=2)
        
        # Title
        title_label = Label(self.frame, text="📈 Digit Recognition Training", 
                           font=("Arial", 14, "bold"), 
                           bg="#4a148c", fg="white")
        title_label.pack(pady=(10, 5))
        
        # Progress section
        progress_frame = Frame(self.frame, bg="#4a148c")
        progress_frame.pack(padx=20, pady=10, fill=tk.X)
        
        # Epoch progress
        self.epoch_var = StringVar(value="Epoch: 0 / 0")
        epoch_label = Label(progress_frame, textvariable=self.epoch_var,
                           font=("Arial", 12, "bold"), bg="#4a148c", fg="white")
        epoch_label.pack()
        
        # Progress bar with purple styling
        style = ttk.Style()
        style.configure("Purple.Horizontal.TProgressbar", 
                       background="#9c27b0",
                       troughcolor="#7b1fa2")
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate',
                                           style="Purple.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Metrics frame with medium purple background
        metrics_frame = Frame(self.frame, bg="#7b1fa2", relief=tk.SUNKEN, borderwidth=2)
        metrics_frame.pack(padx=20, pady=10, fill=tk.X)
        
        # Current training metrics
        self.train_loss_var = StringVar(value="Train Loss: --")
        self.val_loss_var = StringVar(value="Val Loss: --")
        self.accuracy_var = StringVar(value="Accuracy: --%")
        self.time_var = StringVar(value="Time: --s")
        
        Label(metrics_frame, textvariable=self.train_loss_var,
              font=("Arial", 10), bg="#7b1fa2", fg="white").pack(pady=2)
        Label(metrics_frame, textvariable=self.val_loss_var,
              font=("Arial", 10), bg="#7b1fa2", fg="white").pack(pady=2)
        Label(metrics_frame, textvariable=self.accuracy_var,
              font=("Arial", 10), bg="#7b1fa2", fg="white").pack(pady=2)
        Label(metrics_frame, textvariable=self.time_var,
              font=("Arial", 10), bg="#7b1fa2", fg="white").pack(pady=2)
    
    def _create_plot(self):
        """Create the loss curve plot with purple theme."""
        # Create matplotlib figure with purple styling
        self.fig = Figure(figsize=(6, 3), dpi=80, facecolor='#4a148c')
        self.ax = self.fig.add_subplot(111, facecolor='#7b1fa2')
        
        # Purple theme styling
        self.ax.set_xlabel('Epoch', color='white')
        self.ax.set_ylabel('Loss', color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, alpha=0.3, color='white')
        
        # Initialize empty plots with purple/pink colors
        self.train_line, = self.ax.plot([], [], color='#e91e63', linewidth=2, label='Training Loss')
        self.val_line, = self.ax.plot([], [], color='#9c27b0', linewidth=2, label='Validation Loss')
        self.ax.legend(facecolor='#7b1fa2', edgecolor='white', labelcolor='white')
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack(padx=20, pady=10)
    
    def start_training(self, total_epochs: int):
        """
        Start monitoring digit recognition training.
        
        Args:
            total_epochs: Total number of epochs to train
        """
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        self.progress_bar.config(maximum=total_epochs)
        self.epoch_var.set(f"Epoch: 0 / {total_epochs}")
        
        # Clear and reset plot
        self.ax.clear()
        self.ax.set_xlabel('Epoch', color='white')
        self.ax.set_ylabel('Loss', color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, alpha=0.3, color='white')
        self.train_line, = self.ax.plot([], [], color='#e91e63', linewidth=2, label='Training Loss')
        self.val_line, = self.ax.plot([], [], color='#9c27b0', linewidth=2, label='Validation Loss')
        self.ax.legend(facecolor='#7b1fa2', edgecolor='white', labelcolor='white')
        self.canvas.draw()
    
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float = None, 
                    accuracy: float = None, epoch_time: float = None):
        """
        Update digit recognition training progress for current epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            accuracy: Training accuracy percentage (optional)
            epoch_time: Time taken for epoch in seconds (optional)
        """
        self.current_epoch = epoch
        
        # Update training data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        # Update progress bar
        self.progress_bar.config(value=epoch)
        self.epoch_var.set(f"Epoch: {epoch} / {self.total_epochs}")
        
        # Update metrics display
        self.train_loss_var.set(f"Train Loss: {train_loss:.6f}")
        if val_loss is not None:
            self.val_loss_var.set(f"Val Loss: {val_loss:.6f}")
        if accuracy is not None:
            self.accuracy_var.set(f"Digit Accuracy: {accuracy:.1f}%")
        if epoch_time is not None:
            self.time_var.set(f"Time: {epoch_time:.1f}s")
        
        # Update loss curve plot
        self._update_plot()
    
    def _update_plot(self):
        """Update the loss curve plot with new data."""
        if not self.epochs:
            return
        
        # Update training loss line
        self.train_line.set_data(self.epochs, self.train_losses)
        
        # Update validation loss line if available
        if self.val_losses:
            self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits and refresh
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
    
    def pack(self, **kwargs):
        """Pack the training monitor frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the training monitor frame."""
        self.frame.grid(**kwargs)


# Example usage and testing for digit recognizer
if __name__ == "__main__":
    """
    Test the digit recognizer UI components with purple theme.
    """
    print("🧪 Testing Digit Recognizer UI Components")
    print("=" * 50)
    
    # Create test window with purple theme
    root = tk.Tk()
    root.title("Digit Recognizer - UI Components Test")
    root.configure(bg="#2e1065")  # Dark purple background
    
    # Test digit drawing canvas
    def on_digit_draw(drawing_array):
        """Test callback for digit drawing."""
        digit_sum = np.sum(drawing_array)
        print(f"Digit drawing updated: shape={drawing_array.shape}, intensity={digit_sum:.2f}")
    
    # Create digit drawing canvas
    canvas = DrawingCanvas(root, on_draw_callback=on_digit_draw)
    canvas.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Create digit prediction display
    prediction_display = PredictionDisplay(root)
    prediction_display.pack(side=tk.RIGHT, padx=10, pady=10)
    
    # Test digit prediction updates
    def update_digit_predictions():
        """Test digit prediction updates with realistic confidence patterns."""
        # Generate realistic digit predictions (one dominant, others low)
        predictions = np.random.dirichlet(np.ones(10) * 0.1)  # More realistic distribution
        prediction_time = np.random.uniform(5, 25)  # 5-25ms for digit prediction
        
        prediction_display.update_predictions(predictions, prediction_time)
        
        # Schedule next update
        root.after(3000, update_digit_predictions)
    
    # Start prediction updates
    root.after(1000, update_digit_predictions)
    
    print("✅ Digit Recognizer UI Components test window created!")
    print("   🎨 Beautiful purple theme applied")
    print("   ✏️  Draw digits on the left canvas")
    print("   🧠 Watch digit predictions update on the right")
    print("   📊 Observe confidence levels for each digit (0-9)")
    print("   🚀 Close window to continue with digit recognizer development")
    
    # Run the test
    root.mainloop()
