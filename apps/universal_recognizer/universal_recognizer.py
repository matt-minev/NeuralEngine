"""
Universal Character Recognizer - Main Application
===============================================

Complete character recognition system using NeuralEngine.
Recognizes all 62 characters (0-9, A-Z, a-z) with real-time prediction.
Built on your proven digit recognizer architecture.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pickle
import os
import sys
import time
import threading
from PIL import Image, ImageTk

# Add NeuralEngine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nn_core import NeuralNetwork

# Import our universal components
from ui_components import UniversalDrawingCanvas, UniversalPredictionDisplay
from load_data import index_to_character, character_to_index, get_character_type
from train_model import train_universal_model

class UniversalCharacterRecognizer:
    """
    Main application class for universal character recognition.
    Integrates drawing, prediction, and model management.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.model = None
        self.model_info = {}
        self.is_training = False
        
        # Configure main window
        self.setup_main_window()
        
        # Create UI components
        self.create_menu_bar()
        self.create_main_interface()
        
        # Try to load existing model
        self.load_model_on_startup()
        
    def setup_main_window(self):
        """Configure the main application window."""
        self.root.title("🧠 NeuralEngine - Universal Character Recognizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1A1A1A')
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1400x900+{x}+{y}")
        
        # Configure window icon and properties
        try:
            self.root.iconbitmap(default='icon.ico')  # Add icon if available
        except:
            pass
        
        self.root.resizable(True, True)
        self.root.minsize(1200, 700)
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model...", command=self.load_model_dialog)
        file_menu.add_command(label="Save Image...", command=self.save_drawing)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Train New Model...", command=self.start_training)
        model_menu.add_command(label="Model Information", command=self.show_model_info)
        model_menu.add_command(label="Test Model", command=self.run_comprehensive_test)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Clear Canvas", command=self.clear_canvas)
        tools_menu.add_command(label="Predict Current Drawing", command=self.predict_current)
        tools_menu.add_separator()
        tools_menu.add_command(label="Character Recognition Guide", command=self.show_character_guide)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="How to Use", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_interface(self):
        """Create the main application interface."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1A1A1A')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title section
        title_frame = tk.Frame(main_frame, bg='#1A1A1A')
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="🧠 NeuralEngine Universal Character Recognizer",
            font=('Arial', 24, 'bold'),
            bg='#1A1A1A',
            fg='white'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Professional OCR • Recognizes 0-9, A-Z, a-z • Powered by Your NeuralEngine",
            font=('Arial', 12),
            bg='#1A1A1A',
            fg='#B8B8B8'
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Status indicator
        self.status_frame = tk.Frame(title_frame, bg='#1A1A1A')
        self.status_frame.pack(pady=(10, 0))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="🔴 No Model Loaded",
            font=('Arial', 11, 'bold'),
            bg='#1A1A1A',
            fg='#F44336'
        )
        self.status_label.pack()
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg='#1A1A1A')
        content_frame.pack(fill='both', expand=True)
        
        # Left side: Drawing canvas
        left_frame = tk.Frame(content_frame, bg='#1A1A1A')
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        self.drawing_canvas = UniversalDrawingCanvas(left_frame)
        
        # Add prediction button
        predict_frame = tk.Frame(left_frame, bg='#1A1A1A')
        predict_frame.pack(pady=10)
        
        self.predict_btn = tk.Button(
            predict_frame,
            text="🎯 Predict Character",
            command=self.predict_current,
            bg='#9966CC',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3,
            padx=30,
            pady=10,
            state='disabled'
        )
        self.predict_btn.pack()
        
        # Right side: Prediction display
        right_frame = tk.Frame(content_frame, bg='#1A1A1A')
        right_frame.pack(side='right', fill='both', expand=True)
        
        self.prediction_display = UniversalPredictionDisplay(right_frame)
    
    def load_model_on_startup(self):
        """Try to load existing model on application startup."""
        model_path = 'models/universal_character_model.pkl'
        
        if os.path.exists(model_path):
            if self.load_model(model_path):
                print("✅ Universal model loaded automatically on startup")
            else:
                print("⚠️ Failed to load existing model")
        else:
            print("ℹ️ No existing universal model found")
            self.prompt_for_training()
    
    def prompt_for_training(self):
        """Prompt user to train a new model if none exists."""
        response = messagebox.askyesno(
            "No Model Found",
            "No trained universal character model found.\n\n"
            "Would you like to train a new model now?\n"
            "(This will take 20-30 minutes with your EMNIST ByClass dataset)",
            icon='question'
        )
        
        if response:
            self.start_training()
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained universal character model."""
        try:
            print(f"📥 Loading universal model from {model_path}...")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_info = {
                'accuracy': model_data.get('accuracy', 0.0),
                'architecture': model_data.get('architecture', 'unknown'),
                'classes': model_data.get('classes', 62),
                'avg_confidence': model_data.get('avg_confidence', 0.0),
                'training_time': model_data.get('training_time', 0.0),
                'character_accuracies': model_data.get('character_type_accuracies', {})
            }
            
            # Update UI
            self.status_label.config(
                text="🟢 Universal Model Ready",
                fg='#4CAF50'
            )
            self.predict_btn.config(state='normal')
            
            print(f"✅ Universal model loaded successfully!")
            print(f"   Accuracy: {self.model_info['accuracy']:.2f}%")
            print(f"   Classes: {self.model_info['classes']}")
            print(f"   Architecture: {self.model.layer_sizes}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            messagebox.showerror(
                "Model Loading Error",
                f"Failed to load universal character model:\n{str(e)}"
            )
            return False
    
    def load_model_dialog(self):
        """Open dialog to load a model file."""
        file_path = filedialog.askopenfilename(
            title="Load Universal Character Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir="models"
        )
        
        if file_path:
            self.load_model(file_path)
    
    def start_training(self):
        """Start training a new universal character model."""
        if self.is_training:
            messagebox.showwarning("Training in Progress", "Model training is already in progress.")
            return
        
        # Confirm training
        response = messagebox.askyesno(
            "Train Universal Model",
            "This will train a new universal character recognition model.\n\n"
            "Training details:\n"
            "• Dataset: EMNIST ByClass (814,255 samples)\n"
            "• Classes: 62 (0-9, A-Z, a-z)\n"
            "• Training time: 20-30 minutes\n"
            "• Phases: 3 (150 epochs total)\n\n"
            "Continue with training?",
            icon='question'
        )
        
        if not response:
            return
        
        # Start training in separate thread
        self.is_training = True
        self.status_label.config(
            text="🔄 Training Universal Model...",
            fg='#FF9800'
        )
        self.predict_btn.config(state='disabled')
        
        training_thread = threading.Thread(target=self.train_model_worker)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model_worker(self):
        """Worker thread for model training."""
        try:
            print("🚀 Starting universal character model training...")
            
            # Train the model
            model, accuracy = train_universal_model()
            
            # Load the trained model
            self.root.after(0, self.on_training_complete, True, accuracy)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.root.after(0, self.on_training_complete, False, str(e))
    
    def on_training_complete(self, success: bool, result):
        """Handle training completion."""
        self.is_training = False
        
        if success:
            accuracy = result
            # Automatically load the new model
            model_path = 'models/universal_character_model.pkl'
            if self.load_model(model_path):
                messagebox.showinfo(
                    "Training Complete",
                    f"Universal character model training completed successfully!\n\n"
                    f"Final accuracy: {accuracy:.2f}%\n"
                    f"Model saved and loaded automatically."
                )
            else:
                messagebox.showerror(
                    "Training Complete",
                    f"Training completed (accuracy: {accuracy:.2f}%) but failed to load the model."
                )
        else:
            error_msg = result
            messagebox.showerror(
                "Training Failed",
                f"Model training failed:\n{error_msg}"
            )
            self.status_label.config(
                text="🔴 Training Failed",
                fg='#F44336'
            )
    
    def predict_current(self):
        """Predict the currently drawn character."""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load or train a model first.")
            return
        
        try:
            # Get image data from canvas
            img_data = self.drawing_canvas.get_image_data()
            
            # Prepare for prediction
            img_flattened = img_data.flatten().reshape(1, -1)
            
            # Make prediction
            start_time = time.time()
            predictions = self.model.forward(img_flattened)
            prediction_time = (time.time() - start_time) * 1000
            
            # Process results
            predictions = predictions.flatten()
            predicted_index = np.argmax(predictions)
            predicted_char = index_to_character(predicted_index)
            confidence = predictions[predicted_index] * 100
            
            # Update display
            self.prediction_display.update_prediction(
                predicted_char, confidence, predictions.tolist(), prediction_time
            )
            
            print(f"🎯 Prediction: '{predicted_char}' ({confidence:.1f}% confidence)")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            messagebox.showerror("Prediction Error", f"Failed to predict character:\n{str(e)}")
    
    def clear_canvas(self):
        """Clear the drawing canvas and reset prediction display."""
        self.drawing_canvas.clear_canvas()
        self.prediction_display.reset_display()
    
    def save_drawing(self):
        """Save the current drawing to a file."""
        try:
            img_data = self.drawing_canvas.get_image_data()
            
            # Convert to PIL Image
            img = Image.fromarray((img_data * 255).astype(np.uint8), mode='L')
            img = img.resize((280, 280), Image.Resampling.NEAREST)  # Scale up for visibility
            
            # Save dialog
            file_path = filedialog.asksaveasfilename(
                title="Save Drawing",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if file_path:
                img.save(file_path)
                messagebox.showinfo("Saved", f"Drawing saved to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save drawing:\n{str(e)}")
    
    def show_model_info(self):
        """Display detailed model information."""
        if self.model is None:
            messagebox.showinfo("No Model", "No model is currently loaded.")
            return
        
        # Create info window
        info_window = tk.Toplevel(self.root)
        info_window.title("Universal Model Information")
        info_window.geometry("500x400")
        info_window.configure(bg='#2C1810')
        info_window.resizable(False, False)
        
        # Center the window
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (info_window.winfo_screenheight() // 2) - (400 // 2)
        info_window.geometry(f"500x400+{x}+{y}")
        
        # Title
        title_label = tk.Label(
            info_window,
            text="🤖 Universal Model Information",
            font=('Arial', 16, 'bold'),
            bg='#2C1810',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Info text
        info_text = tk.Text(
            info_window,
            height=20,
            width=60,
            bg='#1A1A1A',
            fg='#B8B8B8',
            font=('Courier', 10),
            relief='sunken',
            bd=2,
            state='normal'
        )
        info_text.pack(padx=20, pady=(0, 20), fill='both', expand=True)
        
        # Populate info
        info_content = f"""
🧠 NEURALENGINE UNIVERSAL CHARACTER RECOGNIZER
{'=' * 50}

📊 MODEL PERFORMANCE:
   Overall Accuracy: {self.model_info['accuracy']:.2f}%
   Average Confidence: {self.model_info['avg_confidence']:.1f}%
   Training Time: {self.model_info['training_time']/60:.1f} minutes

🏗️ ARCHITECTURE:
   Input Layer: 784 neurons (28×28 pixels)
   Hidden Layers: {' → '.join(map(str, self.model.layer_sizes[1:-1]))}
   Output Layer: 62 neurons (62 classes)
   Total Parameters: {self.model.count_parameters():,}
   
🎯 CHARACTER COVERAGE:
   Classes: 62 total
   Digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 classes)
   Uppercase: A-Z (26 classes)
   Lowercase: a-z (26 classes)

📈 CHARACTER TYPE PERFORMANCE:
   Digits (0-9): {self.model_info['character_accuracies'].get('digits', 0):.1f}%
   Uppercase (A-Z): {self.model_info['character_accuracies'].get('uppercase', 0):.1f}%
   Lowercase (a-z): {self.model_info['character_accuracies'].get('lowercase', 0):.1f}%

🔧 TRAINING DETAILS:
   Dataset: EMNIST ByClass
   Training Samples: 814,255
   Validation Samples: ~73,000
   Test Samples: 116,323
   
⚙️ TECHNICAL SPECIFICATIONS:
   Framework: Your Custom NeuralEngine
   Loss Function: Cross-entropy
   Optimizer: Adam (multi-phase learning rates)
   Activation: ReLU + Softmax
   Data Format: 28×28 grayscale images
"""
        
        info_text.insert(1.0, info_content)
        info_text.config(state='disabled')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(info_window, orient="vertical", command=info_text.yview)
        info_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
    
    def run_comprehensive_test(self):
        """Run comprehensive model testing."""
        if self.model is None:
            messagebox.showinfo("No Model", "Please load a model first.")
            return
        
        response = messagebox.askyesno(
            "Run Comprehensive Test",
            "This will run a comprehensive test of the universal character model.\n\n"
            "The test will:\n"
            "• Evaluate performance on test dataset\n"
            "• Generate detailed statistics\n"
            "• Create visualization charts\n"
            "• Save results to test_results/\n\n"
            "This may take a few minutes. Continue?",
            icon='question'
        )
        
        if response:
            try:
                # Import and run comprehensive test
                from comprehensive_test import main as run_comprehensive_test
                run_comprehensive_test()
                
                messagebox.showinfo(
                    "Test Complete",
                    "Comprehensive testing completed!\n\n"
                    "Results saved to test_results/ directory:\n"
                    "• Detailed performance report\n"
                    "• Visualization charts\n"
                    "• Confusion matrices"
                )
            except ImportError:
                messagebox.showinfo(
                    "Test Not Available",
                    "Comprehensive test module not found.\n"
                    "Please ensure comprehensive_test.py is in the project directory."
                )
            except Exception as e:
                messagebox.showerror("Test Error", f"Testing failed:\n{str(e)}")
    
    def show_character_guide(self):
        """Show character recognition guide."""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Character Recognition Guide")
        guide_window.geometry("600x500")
        guide_window.configure(bg='#2C1810')
        
        # Center window
        guide_window.update_idletasks()
        x = (guide_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (guide_window.winfo_screenheight() // 2) - (500 // 2)
        guide_window.geometry(f"600x500+{x}+{y}")
        
        # Title
        title_label = tk.Label(
            guide_window,
            text="📚 Character Recognition Guide",
            font=('Arial', 16, 'bold'),
            bg='#2C1810',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Guide content
        guide_text = tk.Text(
            guide_window,
            height=25,
            width=70,
            bg='#1A1A1A',
            fg='#B8B8B8',
            font=('Arial', 11),
            relief='sunken',
            bd=2,
            wrap='word'
        )
        guide_text.pack(padx=20, pady=(0, 20), fill='both', expand=True)
        
        guide_content = """
🎯 HOW TO USE THE UNIVERSAL CHARACTER RECOGNIZER

✏️ DRAWING TIPS:
• Draw characters clearly in the center of the canvas
• Use a brush size appropriate for the character (10-20 works well)
• Make strokes bold and clear - avoid thin, faint lines
• Leave some space around the character edges

📝 CHARACTER TYPES SUPPORTED:

🔢 DIGITS (0-9):
Write numbers clearly. The system recognizes all digits 0 through 9.
Examples: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

🔤 UPPERCASE LETTERS (A-Z):
Draw capital letters with clear, distinct strokes.
Examples: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

🔡 LOWERCASE LETTERS (a-z):
Write lowercase letters with proper proportions.
Examples: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z

💡 BEST PRACTICES:
• Keep characters centered and properly sized
• Draw with consistent stroke thickness
• Avoid decorative or overly stylized writing
• Use the brush size slider to adjust line thickness
• Clear the canvas between different characters

🎯 PREDICTION CONFIDENCE:
• Green text (80%+): High confidence prediction
• Orange text (60-79%): Medium confidence
• Red text (<60%): Low confidence - consider redrawing

📊 UNDERSTANDING RESULTS:
• The tabbed interface shows confidence for all character types
• "Top Predictions" tab shows the most likely characters across all categories
• Character type analysis helps understand what the model sees

🔧 TROUBLESHOOTING:
• If predictions are poor, try redrawing more clearly
• Adjust brush size for better line quality
• Ensure characters are well-centered on the canvas
• Check that the model is properly loaded (green status indicator)

🎉 ENJOY YOUR UNIVERSAL CHARACTER RECOGNIZER!
"""
        
        guide_text.insert(1.0, guide_content)
        guide_text.config(state='disabled')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(guide_window, orient="vertical", command=guide_text.yview)
        guide_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
    
    def show_help(self):
        """Show help information."""
        messagebox.showinfo(
            "Universal Character Recognizer Help",
            "🧠 NeuralEngine Universal Character Recognizer\n\n"
            "QUICK START:\n"
            "1. Ensure a model is loaded (green status indicator)\n"
            "2. Draw any character (0-9, A-Z, a-z) on the canvas\n"
            "3. Click 'Predict Character' or it will predict automatically\n"
            "4. View results in the tabbed prediction display\n\n"
            "FEATURES:\n"
            "• Real-time character recognition\n"
            "• 62 character classes supported\n"
            "• Confidence analysis and visualization\n"
            "• Model training and testing tools\n"
            "• Professional OCR capabilities\n\n"
            "For detailed guidance, see Tools → Character Recognition Guide"
        )
    
    def show_about(self):
        """Show about information."""
        messagebox.showinfo(
            "About Universal Character Recognizer",
            "🧠 NeuralEngine Universal Character Recognizer\n"
            "Version 1.0.0\n\n"
            "Professional-grade optical character recognition\n"
            "supporting complete alphanumeric recognition.\n\n"
            "CAPABILITIES:\n"
            "• 62 character classes (0-9, A-Z, a-z)\n"
            "• Real-time prediction with confidence analysis\n"
            "• EMNIST ByClass dataset training\n"
            "• Advanced neural network architecture\n"
            "• Professional OCR performance\n\n"
            "Built with your custom NeuralEngine framework\n"
            "Designed for accuracy, speed, and reliability.\n\n"
            "🎯 Transforming handwritten characters into digital text"
        )
    
    def run(self):
        """Start the application."""
        print("🚀 Starting Universal Character Recognizer...")
        print("✨ Ready for professional character recognition!")
        self.root.mainloop()

if __name__ == "__main__":
    app = UniversalCharacterRecognizer()
    app.run()