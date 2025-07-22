import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Import your Neural Engine modules
from nn_core import NeuralNetwork, Layer, mean_squared_error
from autodiff import SGD, Adam, TrainingEngine
from data_utils import DataPreprocessor, DataSplitter, BatchProcessor
from utils import ActivationFunctions, MathUtils, NetworkVisualizer, create_test_data

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def create_optimizer_visualizations():
    """Create comprehensive optimizer visualizations."""
    
    # 3.1 SGD and Momentum Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Create synthetic 2D loss landscape
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * X**2 + 2 * Y**2 + 0.3 * np.sin(5*X) * np.cos(5*Y)  # Ravine-like function
    
    # SGD trajectory (more oscillatory)
    sgd_path_x = [-1.5, -1.2, -0.9, -0.8, -0.5, -0.4, -0.1, 0, 0.1]
    sgd_path_y = [0.7, 0.4, 0.6, 0.2, 0.4, 0.1, 0.2, 0, 0.05]
    
    # Momentum trajectory (smoother)
    momentum_path_x = [-1.5, -1.1, -0.8, -0.5, -0.2, 0, 0.1]
    momentum_path_y = [0.7, 0.5, 0.3, 0.15, 0.05, 0, 0.02]
    
    # Plot contours and trajectories
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
    ax1.plot(sgd_path_x, sgd_path_y, 'ro-', linewidth=2, markersize=4, label='SGD', alpha=0.8)
    ax1.plot(momentum_path_x, momentum_path_y, 'bo-', linewidth=2, markersize=4, label='SGD + Momentum', alpha=0.8)
    ax1.set_title('Optimization Trajectories', fontweight='bold')
    ax1.set_xlabel('Parameter θ₁')
    ax1.set_ylabel('Parameter θ₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Convergence curves
    epochs = np.arange(1, 101)
    sgd_loss = 1.0 * np.exp(-epochs/30) + 0.1 * np.sin(epochs/5) * np.exp(-epochs/40)
    momentum_loss = 1.0 * np.exp(-epochs/20) + 0.05 * np.sin(epochs/8) * np.exp(-epochs/50)
    
    ax2.plot(epochs, sgd_loss, 'r-', linewidth=2, label='SGD', alpha=0.8)
    ax2.plot(epochs, momentum_loss, 'b-', linewidth=2, label='SGD + Momentum', alpha=0.8)
    ax2.set_title('Loss Convergence', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Momentum visualization
    velocities = np.array([0, 0.3, 0.5, 0.6, 0.65, 0.68, 0.7])
    gradients = np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1])
    steps = np.arange(len(velocities))
    
    ax3.bar(steps, gradients, alpha=0.6, label='Gradient', color='red')
    ax3.bar(steps, velocities, alpha=0.6, label='Velocity', color='blue', bottom=gradients)
    ax3.set_title('Momentum Accumulation', fontweight='bold')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate scheduling
    lr_schedules = {
        'Constant': np.ones(100) * 0.01,
        'Step Decay': 0.01 * (0.5 ** (epochs // 25)),
        'Exponential': 0.01 * (0.95 ** epochs),
        'Cosine': 0.001 + 0.009 * (1 + np.cos(epochs * np.pi / 100)) / 2
    }
    
    for name, lr_schedule in lr_schedules.items():
        ax4.plot(epochs, lr_schedule, linewidth=2, label=name)
    
    ax4.set_title('Learning Rate Schedules', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle('SGD and Momentum Optimization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimizer_sgd_momentum_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3.2 Detailed Adam Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Adam moment evolution
    epochs = np.arange(1, 101)
    gradients_sim = 0.5 * np.exp(-epochs/20) + 0.1 * np.sin(epochs/5)
    
    # Simulate Adam moments
    beta1, beta2 = 0.9, 0.999
    m = np.zeros_like(epochs, dtype=float)
    v = np.zeros_like(epochs, dtype=float)
    m_hat = np.zeros_like(epochs, dtype=float)
    v_hat = np.zeros_like(epochs, dtype=float)
    
    for i in range(1, len(epochs)):
        g = gradients_sim[i]
        m[i] = beta1 * m[i-1] + (1 - beta1) * g
        v[i] = beta2 * v[i-1] + (1 - beta2) * g**2
        m_hat[i] = m[i] / (1 - beta1**i)
        v_hat[i] = v[i] / (1 - beta2**i)
    
    ax1.plot(epochs, m, 'b-', linewidth=2, label='First moment (m)')
    ax1.plot(epochs, m_hat, 'b--', linewidth=2, label='Bias-corrected (m̂)')
    ax1.set_title('First Moment Evolution', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Moment Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, v, 'r-', linewidth=2, label='Second moment (v)')
    ax2.plot(epochs, v_hat, 'r--', linewidth=2, label='Bias-corrected (v̂)')
    ax2.set_title('Second Moment Evolution', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Moment Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Effective learning rates
    lr_base = 0.001
    epsilon = 1e-8
    effective_lr = lr_base * m_hat / (np.sqrt(v_hat) + epsilon)
    
    ax3.plot(epochs, effective_lr, 'g-', linewidth=2, label='Effective Learning Rate')
    ax3.axhline(lr_base, color='k', linestyle='--', alpha=0.5, label='Base LR')
    ax3.set_title('Adaptive Learning Rate', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Parameter-specific adaptation
    n_params = 10
    param_indices = np.arange(n_params)
    gradient_magnitudes = np.random.exponential(0.5, n_params)
    effective_lrs = lr_base / (np.sqrt(gradient_magnitudes) + epsilon)
    
    ax4.bar(param_indices, gradient_magnitudes, alpha=0.6, color='red', label='Gradient Magnitude')
    ax4_twin = ax4.twinx()
    ax4_twin.bar(param_indices + 0.3, effective_lrs, alpha=0.6, color='blue', 
                 width=0.4, label='Effective LR')
    
    ax4.set_title('Per-Parameter Adaptation', fontweight='bold')
    ax4.set_xlabel('Parameter Index')
    ax4.set_ylabel('Gradient Magnitude', color='red')
    ax4_twin.set_ylabel('Effective LR', color='blue')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Adam Optimizer Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimizer_adam_detailed_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3.3 Comprehensive Optimizer Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convergence comparison on different functions
    epochs = np.arange(1, 101)
    
    # Quadratic function (convex)
    sgd_quadratic = 1.0 * np.exp(-epochs/40)
    momentum_quadratic = 1.0 * np.exp(-epochs/25)
    adam_quadratic = 1.0 * np.exp(-epochs/15)
    
    ax1.plot(epochs, sgd_quadratic, 'r-', linewidth=2, label='SGD')
    ax1.plot(epochs, momentum_quadratic, 'b-', linewidth=2, label='SGD+Momentum')
    ax1.plot(epochs, adam_quadratic, 'g-', linewidth=2, label='Adam')
    ax1.set_title('Convex Function (Quadratic)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Non-convex function (Rosenbrock-like)
    sgd_rosenbrock = 10.0 * np.exp(-epochs/60) + 0.5
    momentum_rosenbrock = 10.0 * np.exp(-epochs/35) + 0.2
    adam_rosenbrock = 10.0 * np.exp(-epochs/20) + 0.1
    
    ax2.plot(epochs, sgd_rosenbrock, 'r-', linewidth=2, label='SGD')
    ax2.plot(epochs, momentum_rosenbrock, 'b-', linewidth=2, label='SGD+Momentum')
    ax2.plot(epochs, adam_rosenbrock, 'g-', linewidth=2, label='Adam')
    ax2.set_title('Non-Convex Function (Rosenbrock)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Memory usage comparison
    optimizers = ['SGD', 'SGD+Momentum', 'Adam', 'RMSprop']
    memory_multipliers = [1, 2, 3, 2]  # Relative to parameter count
    
    bars = ax3.bar(optimizers, memory_multipliers, 
                   color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax3.set_title('Memory Usage (Relative to Parameters)', fontweight='bold')
    ax3.set_ylabel('Memory Multiplier')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, memory_multipliers):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value}×', ha='center', va='bottom', fontweight='bold')
    
    # Hyperparameter sensitivity
    hyperparams = ['Learning Rate', 'Batch Size', 'Architecture']
    sgd_sensitivity = [0.9, 0.3, 0.2]
    adam_sensitivity = [0.3, 0.2, 0.1]
    
    x_pos = np.arange(len(hyperparams))
    width = 0.35
    
    ax4.bar(x_pos - width/2, sgd_sensitivity, width, label='SGD', color='red', alpha=0.7)
    ax4.bar(x_pos + width/2, adam_sensitivity, width, label='Adam', color='green', alpha=0.7)
    
    ax4.set_title('Hyperparameter Sensitivity', fontweight='bold')
    ax4.set_ylabel('Sensitivity Score')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(hyperparams)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Optimizer Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimizer_comprehensive_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

create_optimizer_visualizations()
