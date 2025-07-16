import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
from core.predictor import QuadraticPredictor

class TrainingTab:
    """Model training tab"""
    
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="🧠 Training")
        
        self.training_thread = None
        self.is_training = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup training tab interface"""
        self.create_scenario_selection()
        self.create_training_controls()
        self.create_progress_section()
        
    def create_scenario_selection(self):
        """Create scenario selection checkboxes"""
        scenario_frame = ttk.LabelFrame(self.frame, text="Training Scenarios")
        scenario_frame.pack(fill='x', padx=10, pady=5)
        
        # Scenario checkboxes
        self.scenario_vars = {}
        scenario_grid = ttk.Frame(scenario_frame)
        scenario_grid.pack(fill='x', padx=5, pady=5)
        
        for i, (key, scenario) in enumerate(self.app.scenarios.items()):
            var = tk.BooleanVar(value=True)
            self.scenario_vars[key] = var
            
            # Checkbox
            cb = ttk.Checkbutton(scenario_grid, text=scenario.name, variable=var)
            cb.grid(row=i//2, column=(i%2)*2, sticky='w', padx=5, pady=2)
            
            # Description
            desc_label = ttk.Label(scenario_grid, text=f"  {scenario.description}", 
                                 font=('TkDefaultFont', 8), foreground='gray')
            desc_label.grid(row=i//2, column=(i%2)*2+1, sticky='w', padx=5, pady=2)
            
    def create_training_controls(self):
        """Create training control buttons and settings"""
        control_frame = ttk.LabelFrame(self.frame, text="Training Controls")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Epochs:").pack(side='left', padx=5)
        self.epochs_var = tk.IntVar(value=1000)
        epochs_entry = ttk.Entry(settings_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.pack(side='left', padx=5)
        
        ttk.Label(settings_frame, text="Learning Rate:").pack(side='left', padx=(20, 5))
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_entry = ttk.Entry(settings_frame, textvariable=self.lr_var, width=10)
        lr_entry.pack(side='left', padx=5)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        self.train_button = ttk.Button(button_frame, text="🚀 Start Training", 
                                      command=self.start_training)
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop Training", 
                                     command=self.stop_training, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🗑 Clear Results", 
                  command=self.clear_results).pack(side='left', padx=5)
        
    def create_progress_section(self):
        """Create training progress display"""
        progress_frame = ttk.LabelFrame(self.frame, text="Training Progress")
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        # Training log
        log_container = ttk.Frame(progress_frame)
        log_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.training_log = tk.Text(log_container, height=20, width=80, font=('Courier', 9))
        log_scroll = ttk.Scrollbar(log_container, orient='vertical', command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scroll.set)
        
        self.training_log.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')
        
    def start_training(self):
        """Start training selected models"""
        if self.app.data_processor.data is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
            
        # Get selected scenarios
        selected_scenarios = [key for key, var in self.scenario_vars.items() if var.get()]
        
        if not selected_scenarios:
            messagebox.showwarning("Warning", "Please select at least one scenario to train!")
            return
            
        # Update UI state
        self.is_training = True
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.start()
        
        # Clear previous results
        self.app.predictors.clear()
        self.app.results.clear()
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._train_models,
            args=(selected_scenarios,)
        )
        self.training_thread.start()
        
    def _train_models(self, selected_scenarios):
        """Train models (runs in separate thread)"""
        try:
            self.log_message("🎯 Starting training session...")
            self.log_message(f"Selected scenarios: {len(selected_scenarios)}")
            self.log_message("")
            
            for i, scenario_key in enumerate(selected_scenarios):
                if not self.is_training:  # Check for stop signal
                    break
                    
                scenario = self.app.scenarios[scenario_key]
                self.log_message(f"[{i+1}/{len(selected_scenarios)}] Training {scenario.name}...")
                
                # Create predictor
                predictor = QuadraticPredictor(scenario, self.app.data_processor)
                
                # Train model
                try:
                    training_results = predictor.train(
                        epochs=self.epochs_var.get(),
                        verbose=False
                    )
                    
                    # Store results
                    self.app.predictors[scenario_key] = predictor
                    self.app.results[scenario_key] = training_results['test_results']
                    
                    # Log success
                    test_r2 = training_results['test_results']['r2']
                    training_time = training_results['training_time']
                    self.log_message(f"   ✅ Completed! R²: {test_r2:.4f}, Time: {training_time:.2f}s")
                    
                except Exception as e:
                    self.log_message(f"   ❌ Failed: {str(e)}")
                    
                self.log_message("")
                
            if self.is_training:
                self.log_message("🎉 Training session completed!")
            else:
                self.log_message("⏹ Training stopped by user")
                
        except Exception as e:
            self.log_message(f"❌ Training error: {str(e)}")
        finally:
            # Update UI in main thread
            self.app.root.after(0, self._training_finished)
            
    def _training_finished(self):
        """Update UI after training completion (main thread)"""
        self.is_training = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_bar.stop()
        
        # Update status
        if self.app.predictors:
            self.app.update_status(f"Training complete! {len(self.app.predictors)} models trained", 'success')
        
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.log_message("⏹ Stopping training...")
        
    def clear_results(self):
        """Clear training results"""
        self.app.predictors.clear()
        self.app.results.clear()
        self.training_log.delete(1.0, tk.END)
        self.app.update_status("Results cleared", 'info')
        
    def log_message(self, message: str):
        """Add message to training log (thread-safe)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.app.root.after(0, lambda: self._update_log(full_message))
        
    def _update_log(self, message: str):
        """Update training log in main thread"""
        self.training_log.insert(tk.END, message)
        self.training_log.see(tk.END)
        
    def refresh(self):
        """Refresh tab after data changes"""
        # Enable/disable controls based on data availability
        has_data = self.app.data_processor.data is not None
        state = 'normal' if has_data and not self.is_training else 'disabled'
        self.train_button.config(state=state)
        
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status (interface for main app)"""
        pass  # Training tab doesn't have a separate status display
