import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as patches

def create_detailed_network_architecture():
    # Create larger figure with better proportions
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 12)
    
    # Network configuration
    layer_sizes = [4, 8, 6, 4, 2]
    activations = ['ReLU', 'ReLU', 'ReLU', 'Softmax']
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Hidden\nLayer 3', 'Output\nLayer']
    
    # Professional color scheme - much better gradients
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    edge_colors = ['#1B4D66', '#6B2649', '#A66200', '#8B2A1A', '#3D1F5A']
    
    # Better spacing
    x_positions = [1, 4, 7, 10, 13]
    neuron_positions = []
    
    # Draw layers with improved styling
    for i, (x, size, color, edge_color, layer_name) in enumerate(zip(x_positions, layer_sizes, colors, edge_colors, layer_names)):
        # Calculate y positions for better vertical centering
        if size > 1:
            y_positions = np.linspace(2, 10, size)
        else:
            y_positions = [6]
            
        positions = []
        
        # Draw neurons with gradient effect
        for j, y in enumerate(y_positions):
            # Main neuron circle - larger and more prominent
            neuron = Circle((x, y), 0.4, 
                          facecolor=color, 
                          edgecolor=edge_color, 
                          linewidth=2.5, 
                          zorder=10)
            ax.add_patch(neuron)
            
            # Add inner highlight for 3D effect
            highlight = Circle((x-0.1, y+0.1), 0.15, 
                             facecolor='white', 
                             alpha=0.6, 
                             zorder=11)
            ax.add_patch(highlight)
            
            # Neuron labels with better formatting
            ax.text(x, y, f'{j+1}', 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold', 
                   color='white', zorder=12)
            
            positions.append((x, y))
        
        neuron_positions.append(positions)
        
        # Layer labels with better styling
        ax.text(x, 0.5, layer_name, 
               ha='center', va='center', 
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor=color, 
                        alpha=0.8,
                        edgecolor=edge_color))
        
        # Neuron count
        ax.text(x, -0.2, f'({size} neurons)', 
               ha='center', va='center', 
               fontsize=11, color='#444',
               style='italic')
    
    # Draw ALL connections - this was the main issue in your original
    connection_colors = ['#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7']
    
    for layer_idx in range(len(layer_sizes)-1):
        # Get connection color for this layer
        conn_color = connection_colors[min(layer_idx, len(connection_colors)-1)]
        
        # Draw EVERY connection between adjacent layers
        for i, (x1, y1) in enumerate(neuron_positions[layer_idx]):
            for j, (x2, y2) in enumerate(neuron_positions[layer_idx + 1]):
                # Vary line width and alpha based on connection
                line_width = 1.0 + 0.5 * np.random.random()
                alpha = 0.3 + 0.4 * np.random.random()
                
                # Create smooth curved connections
                connection = FancyArrowPatch(
                    (x1 + 0.4, y1), (x2 - 0.4, y2),
                    connectionstyle=f"arc3,rad={0.1 * (i - j) * 0.1}",
                    arrowstyle="-",
                    color=conn_color,
                    linewidth=line_width,
                    alpha=alpha,
                    zorder=1
                )
                ax.add_patch(connection)
    
    # Enhanced weight matrix annotations
    weight_matrices = []
    for i in range(len(layer_sizes)-1):
        rows, cols = layer_sizes[i+1], layer_sizes[i]
        weight_matrices.append((rows, cols))
    
    for i, (rows, cols) in enumerate(weight_matrices):
        x_mid = (x_positions[i] + x_positions[i+1]) / 2
        
        # Weight matrix symbol with better styling
        ax.text(x_mid, 11, f"$\\mathbf{{W}}^{{({i+1})}}$",
               ha="center", va="center", 
               fontsize=16, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor='#2c3e50',
                        linewidth=2))
        
        # Dimensions
        ax.text(x_mid, 10.4, f"${rows} \\times {cols}$",
               ha="center", va="center", 
               fontsize=12, color='#2c3e50')
    
    # Activation function labels with better positioning
    for i, activation in enumerate(activations):
        x_mid = (x_positions[i+1] + x_positions[i+2]) / 2 if i < len(activations)-1 else x_positions[i+1] + 1
        
        ax.text(x_mid, 8.5, activation,
               ha="center", va="center",
               fontsize=13, fontweight='bold',
               color='#e74c3c',
               bbox=dict(boxstyle="round,pad=0.2", 
                        facecolor='#fff5f5', 
                        alpha=0.8,
                        edgecolor='#e74c3c'))
    
    # Calculate and display parameters with better styling
    total_params = sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1] 
                      for i in range(len(layer_sizes)-1))
    
    ax.text(7, -0.8, f"Total Parameters: {total_params:,}",
           ha="center", va="center",
           fontsize=16, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='#f39c12', 
                    alpha=0.9,
                    edgecolor='#d68910',
                    linewidth=2))
    
    # Enhanced title
    plt.suptitle('Detailed Neural Network Architecture', 
                fontsize=24, fontweight='bold', 
                y=0.95, color='#2c3e50')
    
    # Add subtitle
    ax.text(7, 11.5, 'Fully Connected Deep Neural Network',
           ha="center", va="center",
           fontsize=16, style='italic', color='#7f8c8d')
    
    # Better layout and save
    plt.tight_layout()
    plt.savefig('network_architecture_detailed.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# Run the function
if __name__ == "__main__":
    create_detailed_network_architecture()
