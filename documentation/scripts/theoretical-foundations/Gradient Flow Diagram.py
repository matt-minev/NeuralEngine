import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Define network with better spacing
layers = [2, 3, 1]
x_positions = [2, 6, 10]
colors = ['#E8F5E8', '#E8F0FF', '#FFE8E8']  # Softer colors
layer_names = ['Input Layer', 'Hidden Layer', 'Output Layer']

# Draw neurons with improved styling
neuron_positions = {}
for i, (x, n_neurons, color, name) in enumerate(zip(x_positions, layers, colors, layer_names)):
    y_positions = np.linspace(-1.5, 1.5, n_neurons) if n_neurons > 1 else [0]
    neuron_positions[i] = []
    
    for j, y in enumerate(y_positions):
        # Draw neuron circle
        circle = patches.Circle((x, y), 0.35, fill=True, color=color, 
                              ec='black', linewidth=2)
        ax.add_patch(circle)
        neuron_positions[i].append((x, y))
        
        # Add neuron content without overlap issues
        if i == 0:
            ax.text(x, y, f'$x_{j+1}$', ha='center', va='center', 
                   fontsize=11, weight='bold')
        elif i == len(layers) - 1:
            ax.text(x, y, '$\\hat{y}$', ha='center', va='center', 
                   fontsize=12, weight='bold')
        else:
            ax.text(x, y, f'$h_{j+1}$', ha='center', va='center', 
                   fontsize=10, weight='bold')
    
    # Add layer labels
    ax.text(x, -2.5, name, ha='center', va='center', fontsize=12, weight='bold')

# Draw forward pass connections (blue) - positioned above
for i in range(len(layers) - 1):
    for j, start_pos in enumerate(neuron_positions[i]):
        for k, end_pos in enumerate(neuron_positions[i + 1]):
            # Offset arrows to avoid overlap
            start_point = (start_pos[0] + 0.35, start_pos[1] + 0.1)
            end_point = (end_pos[0] - 0.35, end_pos[1] + 0.1)
            
            ax.annotate('', xy=end_point, xytext=start_point,
                       arrowprops=dict(arrowstyle='->', color='#4472C4', 
                                     lw=2, alpha=0.8))

# Draw backward pass connections (red) - positioned below
for i in range(len(layers) - 1, 0, -1):
    for j, start_pos in enumerate(neuron_positions[i]):
        for k, end_pos in enumerate(neuron_positions[i - 1]):
            # Offset arrows to avoid overlap
            start_point = (start_pos[0] - 0.35, start_pos[1] - 0.15)
            end_point = (end_pos[0] + 0.35, end_pos[1] - 0.15)
            
            ax.annotate('', xy=end_point, xytext=start_point,
                       arrowprops=dict(arrowstyle='->', color='#C55A5A', 
                                     lw=2, alpha=0.8))

# Add process descriptions in separate areas
ax.text(4, 2.5, 'Forward Pass\n(Compute Activations)', ha='center', va='center', 
        fontsize=12, weight='bold', color='#4472C4',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F0FF', 
                 edgecolor='#4472C4', alpha=0.9))

ax.text(8, -3.2, 'Backward Pass\n(Compute Gradients)', ha='center', va='center', 
        fontsize=12, weight='bold', color='#C55A5A',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFE8E8', 
                 edgecolor='#C55A5A', alpha=0.9))

# Add mathematical equations in clear positions
ax.text(4, 0.5, '$\\mathbf{a}^{(l)} = f(\\mathbf{W}^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)})$', 
        ha='center', va='center', fontsize=11, style='italic',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

ax.text(8, 0.5, '$\\frac{\\partial L}{\\partial \\mathbf{W}^{(l)}} = \\boldsymbol{\\delta}^{(l)} (\\mathbf{a}^{(l-1)})^T$', 
        ha='center', va='center', fontsize=11, style='italic',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

# Add directional indicators
ax.text(1, 0, 'Input\n$\\mathbf{x}$', ha='center', va='center', fontsize=11, 
        weight='bold', color='darkgreen')
ax.text(11, 0, 'Output\n$\\hat{\\mathbf{y}}$', ha='center', va='center', fontsize=11, 
        weight='bold', color='darkred')

ax.set_xlim(0, 12)
ax.set_ylim(-4, 3.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Forward and Backward Propagation in Neural Networks', 
             fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('gradient_flow_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white')
plt.show()
