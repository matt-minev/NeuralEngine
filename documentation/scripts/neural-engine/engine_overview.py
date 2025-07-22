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


def create_engine_overview_diagram():
    """Generate the comprehensive architectural diagram of the Neural Engine."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Define module positions and properties with better spacing
    modules = {
        "Data Management\n(data_utils.py)": (1, 4, 3, 2, '#E8F4FD', 'navy'),
        "Core Neural Network\n(nn_core.py)": (6, 4, 3, 2, '#E8F8E8', 'darkgreen'),
        "Automatic Differentiation\n(autodiff.py)": (11, 4, 3, 2, '#FFF0E8', 'darkorange'),
        "Utility Framework\n(utils.py)": (6, 1, 3, 1.5, '#F8E8F8', 'purple'),
        "Training Pipeline": (6, 7.5, 3, 1.5, '#FFEBE8', 'darkred')
    }
    
    # Draw modules with better styling
    for name, (x, y, w, h, facecolor, edgecolor) in modules.items():
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5,
            alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center',
                fontsize=12, fontweight='bold', color=edgecolor)
    
    # Improved data flow arrows with better positioning
    arrows = [
        # Data Management to Core NN (horizontal, middle height)
        ((4, 5), (6, 5), 'Data Input\n& Processing', 'right'),
        # Core NN to AutoDiff (horizontal, middle height)
        ((9, 5), (11, 5), 'Forward Pass\n& Gradients', 'right'),
        # AutoDiff back to Core NN (curved back)
        ((11, 4.5), (9, 4.5), 'Backprop\nGradients', 'left'),
        # Utilities to Core NN (vertical up)
        ((7.5, 2.5), (7.5, 4), 'Helper\nFunctions', 'up'),
        # Training Pipeline to Core NN (vertical down)
        ((7.5, 7.5), (7.5, 6), 'Training\nControl', 'down'),
    ]
    
    # Draw arrows with better styling
    for i, (start, end, label, direction) in enumerate(arrows):
        # Different arrow styles for different connections
        if direction in ['left', 'right']:
            if direction == 'left':
                arrow_style = '<-'
                color = 'darkred'
            else:
                arrow_style = '->'
                color = 'darkblue'
            connectionstyle = "arc3,rad=0.1" if i == 2 else "arc3,rad=0"
        else:
            arrow_style = '->' if direction == 'up' else '<-'
            color = 'darkgreen'
            connectionstyle = "arc3,rad=0"
        
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(
                       arrowstyle=arrow_style, 
                       lw=2.5, 
                       color=color, 
                       alpha=0.8,
                       connectionstyle=connectionstyle
                   ))
        
        # Better label positioning to avoid overlaps
        if direction == 'right':
            mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2 + 0.4
        elif direction == 'left':
            mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2 - 0.4
        elif direction == 'up':
            mid_x, mid_y = (start[0] + end[0])/2 + 1.2, (start[1] + end[1])/2  # Moved further right
        else:  # down
            mid_x, mid_y = (start[0] + end[0])/2 - 1.2, (start[1] + end[1])/2  # Moved further left
        
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=color, alpha=0.9),
                color=color)
    
    # Add component details as annotations with better positioning and larger text
    details = [
        (2.5, 3.2, "• Load/preprocess data\n• Batch management\n• Data validation", 'navy'),
        (7.5, 2.8, "• Layer definitions\n• Forward propagation\n• Model architecture", 'darkgreen'),
        (12.5, 3.2, "• Gradient computation\n• Backpropagation\n• Parameter updates", 'darkorange'),
        (7.5, 0.2, "• Math operations\n• Visualization\n• Helper functions", 'purple'),
        (7.5, 9.5, "• Training loops\n• Optimization\n• Model evaluation", 'darkred')
    ]
    
    for x, y, text, color in details:
        ax.text(x, y, text, ha='center', va='center', fontsize=11,  # Increased font size
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white',  # More padding
                         edgecolor=color, alpha=0.8, linestyle='--', linewidth=1.5),
                color=color, fontweight='normal')  # Made text normal weight for better readability
    
    # Set limits with more padding
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 10.2)
    
    # Enhanced title
    ax.set_title('Neural Network Engine Architecture Overview', 
                 fontsize=18, fontweight='bold', pad=30,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Add legend for arrow meanings
    legend_elements = [
        plt.Line2D([0], [0], color='darkblue', lw=3, label='Data Flow →'),
        plt.Line2D([0], [0], color='darkred', lw=3, label='← Gradient Flow'),
        plt.Line2D([0], [0], color='darkgreen', lw=3, label='↕ Control Flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.savefig('engine_overview_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

create_engine_overview_diagram()