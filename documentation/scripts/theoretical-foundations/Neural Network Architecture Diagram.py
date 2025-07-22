import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 9))

# Define layer configuration
layers = [3, 4, 4, 2]
layer_names = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
layer_colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
x_positions = [1, 4, 7, 10]

# Draw neurons with color coding
neuron_positions = {}
for i, (x, n_neurons, color, name) in enumerate(zip(x_positions, layers, layer_colors, layer_names)):
    y_positions = np.linspace(-2, 2, n_neurons) if n_neurons > 1 else [0]
    neuron_positions[i] = []
    
    for j, y in enumerate(y_positions):
        circle = patches.Circle((x, y), 0.25, fill=True, color=color, 
                              ec='black', linewidth=1.5)
        ax.add_patch(circle)
        neuron_positions[i].append((x, y))
    
    # Add layer labels below
    ax.text(x, -3, name, ha='center', va='center', fontsize=11, weight='bold')

# Draw connections between layers with reduced opacity
for i in range(len(layers) - 1):
    for start_pos in neuron_positions[i]:
        for end_pos in neuron_positions[i + 1]:
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   'gray', alpha=0.4, linewidth=0.8)

# Add comprehensive weight and bias annotations
weight_annotations = [
    (2.5, 1.5, '$\\mathbf{W}^{(1)}$'),
    (5.5, 1.5, '$\\mathbf{W}^{(2)}$'),
    (8.5, 1.5, '$\\mathbf{W}^{(3)}$')
]

bias_annotations = [
    (4, -2.5, '$\\mathbf{b}^{(1)}$'),
    (7, -2.5, '$\\mathbf{b}^{(2)}$'),
    (10, -2.5, '$\\mathbf{b}^{(3)}$')
]

for x, y, label in weight_annotations:
    ax.text(x, y, label, ha='center', va='center', fontsize=12, 
           weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='white', alpha=0.8))

for x, y, label in bias_annotations:
    ax.text(x, y, label, ha='center', va='center', fontsize=11, 
           weight='bold', color='darkblue')

# Add input and output labels
ax.text(0.2, 0, '$\\mathbf{x}$', ha='center', va='center', fontsize=16, 
        weight='bold', color='darkgreen')
ax.text(10.8, 0, '$\\hat{\\mathbf{y}}$', ha='center', va='center', fontsize=16, 
        weight='bold', color='darkred')

# Add mathematical flow annotations
ax.text(2.5, -1, '$\\mathbf{z}^{(1)} = \\mathbf{W}^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}$', 
        ha='center', fontsize=9, style='italic')
ax.text(5.5, -1, '$\\mathbf{z}^{(2)} = \\mathbf{W}^{(2)}\\mathbf{a}^{(1)} + \\mathbf{b}^{(2)}$', 
        ha='center', fontsize=9, style='italic')
ax.text(8.5, -1, '$\\mathbf{z}^{(3)} = \\mathbf{W}^{(3)}\\mathbf{a}^{(2)} + \\mathbf{b}^{(3)}$', 
        ha='center', fontsize=9, style='italic')

# Add activation function notation
ax.text(4, 2.5, '$\\mathbf{a}^{(1)} = f(\\mathbf{z}^{(1)})$', ha='center', 
        fontsize=10, weight='bold', color='blue')
ax.text(7, 2.5, '$\\mathbf{a}^{(2)} = f(\\mathbf{z}^{(2)})$', ha='center', 
        fontsize=10, weight='bold', color='blue')

ax.set_xlim(-0.5, 11.5)
ax.set_ylim(-3.5, 3)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Comprehensive Neural Network Architecture', fontsize=16, 
             weight='bold', pad=20)

plt.tight_layout()
plt.savefig('neural_network_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white')
plt.show()
