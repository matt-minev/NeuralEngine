import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

def create_engine_overview_diagram_svg():
    """Generate the architectural diagram of the Neural Engine for dark background in SVG format."""
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300, constrained_layout=True)
    ax.axis('off')
    
    # No background for SVG transparency
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Dark background compatible colors - darker fills with bright edges
    modules = {
        "Data Management\n(data_utils.py)": (1, 4, 3, 2, '#1E3A8A', 'deepskyblue'),
        "Core Neural Network\n(nn_core.py)": (6, 4, 3, 2, '#14532D', 'limegreen'),
        "Automatic Differentiation\n(autodiff.py)": (11, 4, 3, 2, '#78350F', 'darkorange'),
        "Utility Framework\n(utils.py)": (6, 1, 3, 1.5, '#5B21B6', 'mediumorchid'),
        "Training Pipeline": (6, 7.5, 3, 1.5, '#7F1D1D', 'orangered')
    }

    # Draw modules with improved styling for dark backgrounds
    for name, (x, y, w, h, facecolor, edgecolor) in modules.items():
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5, alpha=0.85
        )
        ax.add_patch(box)
        # White text for dark background compatibility
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', 
                fontsize=15, fontweight='bold', color='white')

    # Improved arrow positioning with better spacing
    arrows = [
        ((4, 5), (6, 5), 'Data Input\n& Processing', 'right'),
        ((9, 5), (11, 5), 'Forward Pass\n& Gradients', 'right'),
        ((11, 4.5), (9, 4.5), 'Backprop\nGradients', 'left'),
        ((7.5, 2.5), (7.5, 4), 'Helper\nFunctions', 'up'),
        ((7.5, 7.5), (7.5, 6), 'Training\nControl', 'down'),
    ]

    # Draw arrows with bright colors for dark background
    for i, (start, end, label, direction) in enumerate(arrows):
        if direction in ['left', 'right']:
            if direction == 'left':
                arrow_style = '<-'
                color = 'orangered'
            else:
                arrow_style = '->'
                color = 'deepskyblue'
            connectionstyle = "arc3,rad=0.1" if i == 2 else "arc3,rad=0"
        else:
            arrow_style = '->' if direction == 'up' else '<-'
            color = 'limegreen'
            connectionstyle = "arc3,rad=0"

        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=arrow_style, lw=3, color=color, 
                                  alpha=0.9, connectionstyle=connectionstyle))

        # Better spacing for arrow labels
        if direction == 'right':
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.5
        elif direction == 'left':
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 0.5
        elif direction == 'up':
            mid_x, mid_y = (start[0] + end[0]) / 2 + 1.5, (start[1] + end[1]) / 2
        else:  # down
            mid_x, mid_y = (start[0] + end[0]) / 2 - 1.5, (start[1] + end[1]) / 2

        # Dark background for label boxes with bright text
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', 
                         edgecolor=color, alpha=0.85),
                color=color)

    # Component details with larger fonts and better spacing
    details = [
        (2.5, 3.2, "• Load/preprocess data\n• Batch management\n• Data validation", 'deepskyblue'),
        (7.5, 2.8, "• Layer definitions\n• Forward propagation\n• Model architecture", 'limegreen'),
        (12.5, 3.2, "• Gradient computation\n• Backpropagation\n• Parameter updates", 'darkorange'),
        (7.5, 0.2, "• Math operations\n• Visualization\n• Helper functions", 'mediumorchid'),
        (7.5, 9.5, "• Training loops\n• Optimization\n• Model evaluation", 'orangered')
    ]

    for x, y, text, color in details:
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=13, fontweight='normal',
                bbox=dict(boxstyle="round,pad=0.6", facecolor='black', 
                         edgecolor=color, alpha=0.85),
                color=color)

    # Generous spacing
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 10.2)

    # White title for dark background
    ax.set_title('Neural Network Engine Architecture Overview', 
                 fontsize=24, fontweight='bold', pad=30, color='white')

    # Legend with bright colors and larger fonts
    legend_elements = [
        plt.Line2D([0], [0], color='deepskyblue', lw=4, label='Data Flow →'),
        plt.Line2D([0], [0], color='orangered', lw=4, label='← Gradient Flow'),
        plt.Line2D([0], [0], color='limegreen', lw=4, label='↕ Control Flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
              fontsize=14, frameon=False, labelcolor='white')

    # Save as SVG with transparent background
    plt.savefig('engine_overview_diagram.svg', format='svg', dpi=300, 
                bbox_inches='tight', facecolor='none')
    plt.close()

# Execute the function
create_engine_overview_diagram_svg()
