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

# Import the Neural Engine modules
from nn_core import NeuralNetwork, Layer, mean_squared_error
from autodiff import SGD, Adam, TrainingEngine
from data_utils import DataPreprocessor, DataSplitter, BatchProcessor
from utils import ActivationFunctions, MathUtils, NetworkVisualizer, create_test_data

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")


def create_activation_visualizations():
    """Create comprehensive activation function visualizations."""
    x = np.linspace(-5, 5, 1000)
    
    # 2.1 Detailed ReLU Analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # ReLU function
    relu_y = ActivationFunctions.relu(x)
    ax1.plot(x, relu_y, 'b-', linewidth=3, label='ReLU(z)')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('ReLU Function', fontweight='bold')
    ax1.set_xlabel('Input (z)')
    ax1.set_ylabel('Output')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ReLU derivative (approximated)
    relu_deriv = np.where(x > 0, 1, 0)
    ax2.plot(x, relu_deriv, 'r-', linewidth=3, label="ReLU'(z)")
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('ReLU Derivative', fontweight='bold')
    ax2.set_xlabel('Input (z)')
    ax2.set_ylabel('Derivative')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Dead neuron illustration
    ax3.fill_between(x[x < 0], 0, 1, alpha=0.3, color='red', label='Dead Region')
    ax3.fill_between(x[x >= 0], 0, 1, alpha=0.3, color='green', label='Active Region')
    ax3.axvline(0, color='k', linestyle='--', alpha=0.7)
    ax3.set_title('ReLU Activation Regions', fontweight='bold')
    ax3.set_xlabel('Pre-activation Value')
    ax3.set_ylabel('Gradient Flow')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed ReLU Activation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_relu_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2.2 ReLU Family Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Functions
    relu_y = ActivationFunctions.relu(x)
    leaky_relu_y = ActivationFunctions.leaky_relu(x, alpha=0.01)
    elu_y = ActivationFunctions.elu(x, alpha=1.0)
    
    ax1.plot(x, relu_y, 'b-', linewidth=2, label='ReLU')
    ax1.plot(x, leaky_relu_y, 'r-', linewidth=2, label='Leaky ReLU (α=0.01)')
    ax1.plot(x, elu_y, 'g-', linewidth=2, label='ELU (α=1.0)')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('Function Values', fontweight='bold')
    ax1.set_xlabel('Input (z)')
    ax1.set_ylabel('f(z)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Derivatives (approximated for visualization)
    relu_deriv = np.where(x > 0, 1, 0)
    leaky_relu_deriv = np.where(x > 0, 1, 0.01)
    elu_deriv = np.where(x > 0, 1, np.exp(x))
    
    ax2.plot(x, relu_deriv, 'b-', linewidth=2, label="ReLU'")
    ax2.plot(x, leaky_relu_deriv, 'r-', linewidth=2, label="Leaky ReLU'")
    ax2.plot(x, elu_deriv, 'g-', linewidth=2, label="ELU'")
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Derivatives', fontweight='bold')
    ax2.set_xlabel('Input (z)')
    ax2.set_ylabel("f'(z)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gradient flow comparison
    ax3.bar(['ReLU', 'Leaky ReLU', 'ELU'], [50, 1, 0], 
            color=['red', 'orange', 'green'], alpha=0.7)
    ax3.set_title('Dead Neuron Problem', fontweight='bold')
    ax3.set_ylabel('% Dead Neurons (Negative Inputs)')
    ax3.grid(True, alpha=0.3)
    
    # Computational cost comparison
    costs = [1, 1.1, 2.5]  # Relative computational costs
    ax4.bar(['ReLU', 'Leaky ReLU', 'ELU'], costs, 
            color=['blue', 'orange', 'red'], alpha=0.7)
    ax4.set_title('Computational Cost (Relative)', fontweight='bold')
    ax4.set_ylabel('Relative Cost')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('ReLU Family Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_relu_family.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2.3 Classical Smooth Activations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    sigmoid_y = ActivationFunctions.sigmoid(x)
    tanh_y = ActivationFunctions.tanh(x)
    
    ax1.plot(x, sigmoid_y, 'purple', linewidth=3, label='Sigmoid')
    ax1.plot(x, tanh_y, 'orange', linewidth=3, label='Tanh')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('Function Values', fontweight='bold')
    ax1.set_xlabel('Input (z)')
    ax1.set_ylabel('f(z)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Derivatives
    sigmoid_deriv = sigmoid_y * (1 - sigmoid_y)
    tanh_deriv = 1 - tanh_y**2
    
    ax2.plot(x, sigmoid_deriv, 'purple', linewidth=3, label="Sigmoid'")
    ax2.plot(x, tanh_deriv, 'orange', linewidth=3, label="Tanh'")
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Derivatives', fontweight='bold')
    ax2.set_xlabel('Input (z)')
    ax2.set_ylabel("f'(z)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Saturation regions
    ax3.fill_between(x, 0, np.where(np.abs(x) > 3, 1, 0), alpha=0.3, color='red', label='Saturation')
    ax3.fill_between(x, 0, np.where(np.abs(x) <= 3, 1, 0), alpha=0.3, color='green', label='Active')
    ax3.axvline(-3, color='r', linestyle='--', alpha=0.7)
    ax3.axvline(3, color='r', linestyle='--', alpha=0.7)
    ax3.set_title('Saturation Analysis', fontweight='bold')
    ax3.set_xlabel('Input (z)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gradient magnitude comparison
    max_grads = [0.25, 1.0]  # Maximum gradients for sigmoid and tanh
    ax4.bar(['Sigmoid', 'Tanh'], max_grads, color=['purple', 'orange'], alpha=0.7)
    ax4.set_title('Maximum Gradient Values', fontweight='bold')
    ax4.set_ylabel('Max Gradient')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Classical Smooth Activations: Sigmoid vs Tanh', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_classical_smooth.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2.4 Modern Advanced Activations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    swish_y = ActivationFunctions.swish(x)
    gelu_y = ActivationFunctions.gelu(x)
    
    ax1.plot(x, swish_y, 'red', linewidth=3, label='Swish')
    ax1.plot(x, gelu_y, 'blue', linewidth=3, label='GELU')
    ax1.plot(x, relu_y, 'gray', linewidth=2, alpha=0.5, label='ReLU (reference)')
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('Function Values', fontweight='bold')
    ax1.set_xlabel('Input (z)')
    ax1.set_ylabel('f(z)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show non-monotonic behavior
    x_nm = np.linspace(-2, 1, 300)
    swish_nm = ActivationFunctions.swish(x_nm)
    ax2.plot(x_nm, swish_nm, 'red', linewidth=3, label='Swish')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Swish Non-Monotonic Behavior', fontweight='bold')
    ax2.set_xlabel('Input (z)')
    ax2.set_ylabel('Swish(z)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Performance comparison
    functions = ['ReLU', 'Swish', 'GELU']
    performance = [100, 87, 92]  # Relative performance
    computational_cost = [1, 3, 3.5]  # Relative cost
    
    ax3.bar(functions, performance, color=['gray', 'red', 'blue'], alpha=0.7)
    ax3.set_title('Empirical Performance', fontweight='bold')
    ax3.set_ylabel('Performance Score (%)')
    ax3.grid(True, alpha=0.3)
    
    ax4.bar(functions, computational_cost, color=['gray', 'red', 'blue'], alpha=0.7)
    ax4.set_title('Computational Cost (Relative)', fontweight='bold')
    ax4.set_ylabel('Relative Cost')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Modern Advanced Activations: Swish vs GELU', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_modern_advanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2.5 Specialized Output Activations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Softmax demonstration
    z_values = np.array([1, 2, 3, 4, 5])
    softmax_output = ActivationFunctions.softmax(z_values)
    
    ax1.bar(range(len(z_values)), z_values, alpha=0.7, color='lightblue', label='Input (logits)')
    ax1.set_title('Softmax Input (Logits)', fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Class Index')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(range(len(softmax_output)), softmax_output, alpha=0.7, color='lightcoral', label='Softmax Output')
    ax2.set_title('Softmax Output (Probabilities)', fontweight='bold')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Class Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Linear activation
    linear_y = ActivationFunctions.linear(x)
    ax3.plot(x, linear_y, 'green', linewidth=3, label='Linear(z) = z')
    ax3.plot(x, x, 'gray', linewidth=1, linestyle='--', alpha=0.5, label='y = x reference')
    ax3.set_title('Linear Activation', fontweight='bold')
    ax3.set_xlabel('Input (z)')
    ax3.set_ylabel('f(z)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Use case comparison
    use_cases = ['Binary\nClassification', 'Multi-class\nClassification', 'Regression']
    activations = ['Sigmoid', 'Softmax', 'Linear']
    colors = ['purple', 'red', 'green']
    
    ax4.bar(use_cases, [1, 1, 1], color=colors, alpha=0.7)
    for i, (use_case, activation) in enumerate(zip(use_cases, activations)):
        ax4.text(i, 0.5, activation, ha='center', va='center', fontweight='bold', color='white')
    ax4.set_title('Specialized Output Activations', fontweight='bold')
    ax4.set_ylabel('Applicability')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Specialized Output Activation Functions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_specialized_output.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2.6 Comprehensive Activation Functions Overview
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # All activation functions
    activations_data = {
        'ReLU': ActivationFunctions.relu(x),
        'Leaky ReLU': ActivationFunctions.leaky_relu(x),
        'ELU': ActivationFunctions.elu(x),
        'Sigmoid': ActivationFunctions.sigmoid(x),
        'Tanh': ActivationFunctions.tanh(x),
        'Swish': ActivationFunctions.swish(x),
        'GELU': ActivationFunctions.gelu(x),
        'Linear': ActivationFunctions.linear(x)
    }
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(activations_data)))
    
    for i, (name, y_values) in enumerate(activations_data.items()):
        ax1.plot(x, y_values, linewidth=2, label=name, color=colors[i])
    
    ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title('All Activation Functions', fontweight='bold')
    ax1.set_xlabel('Input (z)')
    ax1.set_ylabel('f(z)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-2, 5)
    
    # Gradient preservation analysis
    gradient_preservation = [0, 1, 100, 25, 100, 85, 90, 100]  # % gradient preservation
    function_names = list(activations_data.keys())
    
    ax2.bar(function_names, gradient_preservation, color=colors, alpha=0.7)
    ax2.set_title('Gradient Preservation (%)', fontweight='bold')
    ax2.set_ylabel('Gradient Preservation')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Computational cost comparison
    computational_costs = [1, 1.1, 2.5, 2, 2, 3, 3.5, 1]  # Relative costs
    
    ax3.bar(function_names, computational_costs, color=colors, alpha=0.7)
    ax3.set_title('Computational Cost (Relative)', fontweight='bold')
    ax3.set_ylabel('Relative Cost')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Use case matrix
    use_case_matrix = np.array([
        [0.9, 0.1, 0.8],  # ReLU: [Hidden, Output, Modern]
        [0.9, 0.1, 0.6],  # Leaky ReLU
        [0.8, 0.2, 0.7],  # ELU
        [0.2, 0.9, 0.1],  # Sigmoid
        [0.3, 0.3, 0.1],  # Tanh
        [0.9, 0.2, 0.9],  # Swish
        [0.9, 0.2, 0.9],  # GELU
        [0.1, 0.9, 0.5],  # Linear
    ])
    
    im = ax4.imshow(use_case_matrix, cmap='RdYlGn', aspect='auto')
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['Hidden Layers', 'Output Layers', 'Modern Archs'])
    ax4.set_yticks(range(len(function_names)))
    ax4.set_yticklabels(function_names)
    ax4.set_title('Use Case Suitability Matrix', fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, shrink=0.6)
    
    plt.suptitle('Comprehensive Activation Function Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_functions_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

create_activation_visualizations()
