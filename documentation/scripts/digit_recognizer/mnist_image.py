import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def load_csv_data(filepath):
    """Load data from CSV format (works for both MNIST and EMNIST)"""
    data = pd.read_csv(filepath)
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values.reshape(-1, 28, 28)
    return images, labels

def apply_emnist_transforms_fixed(image):
    """
    EMNIST transformation: 90° CCW rotation, then vertical flip (fixes 'upside down' bug)
    """
    rotated = np.rot90(image)
    # Flip vertically for correct orientation
    flipped = np.flipud(rotated) 
    return flipped

def create_dataset_samples_visualization():
    """Create comprehensive dataset samples visualization with improved layout and vibrant colors."""
    fig = plt.figure(figsize=(16, 13), facecolor='white')
    
    # Try loading data, or fall back to synthetic data for demo
    try:
        mnist_images, mnist_labels = load_csv_data(
            r'C:\Users\Matt\Desktop\NeuralEngine\apps\digit_recognizer\data\mnist_train.csv')
        emnist_images, emnist_labels = load_csv_data(
            r'C:\Users\Matt\Desktop\NeuralEngine\apps\digit_recognizer\data\emnist-digits-train.csv')
    except Exception as e:
        print('Error loading data, using synthetic placeholders.')
        np.random.seed(42)
        mnist_images = np.random.randint(0, 255, (1000, 28, 28))
        mnist_labels = np.random.randint(0, 10, 1000)
        emnist_images = np.random.randint(0, 255, (1000, 28, 28))
        emnist_labels = np.random.randint(0, 10, 1000)

    # Updated pipeline colors (keep blue and purple, change yellow/green/orange for readability)
    vibrant_pipeline_colors = ['#4062FA', '#6A0DAD', '#3B8350', '#916A44', '#9D0191']
    bar_colors = ['#00C1D4', '#FD6F96']
    
    gs = fig.add_gridspec(5, 10, height_ratios=[1, 1, 1, 0.8, 0.8], hspace=0.66, wspace=0.25)
    
    # Section 1: MNIST Samples - CENTERED TITLE
    fig.text(0.5, 0.89, 'MNIST Dataset Samples', fontsize=16, fontweight='bold', color='#1E40AF', ha='center')
    for i in range(10):
        ax = fig.add_subplot(gs[0, i])
        idx = np.where(mnist_labels == i)[0]
        if len(idx) > 0:
            ax.imshow(mnist_images[idx[0]], cmap='gray')
            ax.set_title(f'{i}', fontsize=13, color='#1E40AF', fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No\n{i}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        
    # Section 2: EMNIST Samples (raw) - CENTERED TITLE
    fig.text(0.5, 0.73, 'EMNIST Dataset Samples (Raw)', fontsize=16, fontweight='bold', color='#D90429', ha='center')
    for i in range(10):
        ax = fig.add_subplot(gs[1, i])
        idx = np.where(emnist_labels == i)[0]
        if len(idx) > 0:
            ax.imshow(emnist_images[idx[0]], cmap='gray')
            ax.set_title(f'{i}', fontsize=13, color='#D90429', fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No\n{i}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Section 3: EMNIST Samples (fixed transformation) - CENTERED TITLE
    fig.text(0.5, 0.57, 'EMNIST Samples (Fixed Transformation)', 
             fontsize=16, fontweight='bold', color='#119822', ha='center')
    for i in range(10):
        ax = fig.add_subplot(gs[2, i])
        idx = np.where(emnist_labels == i)[0]
        if len(idx) > 0:
            img = apply_emnist_transforms_fixed(emnist_images[idx[0]])
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{i}', fontsize=13, color='#119822', fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No\n{i}', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Badge moved to right of EMNIST title (no overlap)
    fig.text(0.72, 0.73, 'EMNIST has 4× more training data', fontsize=11, color='#FF5400',
             bbox=dict(boxstyle='round,pad=0.24', facecolor='#fffbe6', edgecolor='#FF5400', alpha=0.8))
    
    # Section 4: Processing Pipeline (fixed colors for readability)
    pipeline_ax = fig.add_subplot(gs[3, :6])
    pipeline_ax.set_xlim(0, 10)
    pipeline_ax.set_ylim(0, 3)
    pipeline_ax.axis('off')

    steps = [
        ('Raw\n28×28', 1, 1.5, vibrant_pipeline_colors[0]),  # Blue - good
        ('Normalize\n÷255', 3, 1.5, vibrant_pipeline_colors[1]),  # Purple - good
        ('Reshape\n784×1', 5, 1.5, vibrant_pipeline_colors[2]),  # Dark green - readable
        ('One-hot\nLabels', 7, 1.5, vibrant_pipeline_colors[3]),  # Brown - readable
        ('Training\nReady', 9, 1.5, vibrant_pipeline_colors[4])  # Dark purple - good
    ]
    
    # Draw boxes first
    for i, (label, x, y, color) in enumerate(steps):
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                             boxstyle='round,pad=0.16', facecolor=color, edgecolor='black', linewidth=1.7)
        pipeline_ax.add_patch(box)
        pipeline_ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Draw arrows between boxes
    for i in range(len(steps) - 1):
        x_start = steps[i][1] + 0.6  # Right edge of current box
        x_end = steps[i+1][1] - 0.6  # Left edge of next box
        y = steps[i][2]  # Same y coordinate
        
        pipeline_ax.arrow(x_start, y, x_end - x_start, 0, 
                         head_width=0.15, head_length=0.15, 
                         fc='black', ec='black', linewidth=2)

    pipeline_ax.set_title('Data Preprocessing Pipeline', fontsize=15, fontweight='bold', y=0.98)
    
    # Section 5: Dataset Size Comparison (fixed positioning to avoid overlap)
    stats_ax = fig.add_subplot(gs[3, 6:])
    datasets = ['MNIST', 'EMNIST']
    train_samples = [60000, 240000]
    test_samples = [10000, 40000]
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    stats_ax.bar(x_pos - width/2, np.array(train_samples)/1000, width, 
                 label='Training', color=bar_colors[0], alpha=0.85, zorder=2)
    stats_ax.bar(x_pos + width/2, np.array(test_samples)/1000, width, 
                 label='Testing', color=bar_colors[1], alpha=0.85, zorder=2)
    
    # Fixed y-label positioning to avoid overlap with pipeline
    stats_ax.set_ylabel('Samples\n(thousands)', fontweight='bold', fontsize=10)
    stats_ax.set_xticks(x_pos)
    stats_ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    stats_ax.legend(frameon=False, fontsize=11)
    stats_ax.grid(True, axis='y', linestyle=':', alpha=0.35, zorder=0)
    stats_ax.set_axisbelow(True)
    stats_ax.set_title('Dataset Size Comparison', fontweight='bold', fontsize=14, pad=13)
    
    # Value labels
    for i, (tr, ts) in enumerate(zip(train_samples, test_samples)):
        stats_ax.text(i-width/2, tr/1000+2.8, f'{tr//1000}K', ha='center', va='bottom', fontsize=11, fontweight='bold')
        stats_ax.text(i+width/2, ts/1000+2.4, f'{ts//1000}K', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Section 6: Transformation Steps Illustration
    transform_ax = fig.add_subplot(gs[4, :])
    transform_ax.axis('off')
    sample_idx = np.where(emnist_labels == 3)[0]
    if len(sample_idx) > 0:
        original = emnist_images[sample_idx[0]]
        ax1 = fig.add_subplot(gs[4, 1])
        ax1.imshow(original, cmap='gray')
        ax1.set_title('1. Original', fontsize=12)
        ax1.axis('off')
        # 90° CCW
        rotated = np.rot90(original)
        ax2 = fig.add_subplot(gs[4, 4])
        ax2.imshow(rotated, cmap='gray')
        ax2.set_title('2. 90° CCW', fontsize=12)
        ax2.axis('off')
        # Vertical flip (the fix)
        flipped = np.flipud(rotated)
        ax3 = fig.add_subplot(gs[4, 7])
        ax3.imshow(flipped, cmap='gray')
        ax3.set_title('3. Vertical Flip', fontsize=12)
        ax3.axis('off')
        # Arrows for step flow
        fig.text(0.28, 0.17, '→', fontsize=20, fontweight='bold', ha='center', color='#00C1D4')
        fig.text(0.57, 0.17, '→', fontsize=20, fontweight='bold', ha='center', color='#FD6F96')
    fig.text(0.5, 0.08, 'EMNIST Transformation: 90° CCW Rotation, then Vertical Flip (Orientation Fix)', 
             fontsize=14, fontweight='bold', ha='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF176', alpha=0.85))
    
    # Main title
    fig.suptitle('MNIST vs EMNIST: Samples, Transformation & Preprocessing Pipeline', 
                 fontsize=19, fontweight='bold', y=0.97, color='#35477D')
    
    # Custom legend MOVED TO BOTTOM RIGHT CORNER
    mnist_patch = mpatches.Patch(color='#1E40AF', label='MNIST Sample')
    emnist_raw_patch = mpatches.Patch(color='#D90429', label='EMNIST Raw')
    emnist_trans_patch = mpatches.Patch(color='#119822', label='EMNIST Transformed')
    fig.legend(handles=[mnist_patch, emnist_raw_patch, emnist_trans_patch], 
               loc='lower right', bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=13)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig('digit_recognizer_dataset_samples_final.png', dpi=310, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Visualization saved as 'digit_recognizer_dataset_samples_final.png'")
    
if __name__ == "__main__":
    create_dataset_samples_visualization()
