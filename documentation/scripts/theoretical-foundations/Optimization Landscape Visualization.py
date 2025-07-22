import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create optimization landscape
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define parameter space
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Create complex loss landscape with multiple minima and saddle points
Z = (X**2 + Y**2) * 0.5 + 0.3 * np.sin(3*X) * np.cos(3*Y) + 0.2 * (X**2 * Y - Y**3/3)

# Plot surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)

# Add gradient descent trajectory
trajectory_x = [-1.5, -1.2, -0.8, -0.4, -0.1, 0.1]
trajectory_y = [1.2, 0.9, 0.6, 0.3, 0.1, 0.0]
trajectory_z = [((x**2 + y**2) * 0.5 + 0.3 * np.sin(3*x) * np.cos(3*y) + 0.2 * (x**2 * y - y**3/3)) 
                for x, y in zip(trajectory_x, trajectory_y)]

ax.plot(trajectory_x, trajectory_y, trajectory_z, 'r-', linewidth=3, alpha=0.9, label='Gradient Descent Path')
ax.scatter(trajectory_x, trajectory_y, trajectory_z, c='red', s=50, alpha=0.9)

# Mark important points
ax.scatter([0], [0], [0], c='blue', s=100, label='Global Minimum', alpha=0.9)
ax.scatter([-0.8, 0.8], [0.8, -0.8], [0.5, 0.5], c='orange', s=80, label='Local Minima', alpha=0.9)

ax.set_xlabel('Parameter $\\theta_1$', fontsize=12)
ax.set_ylabel('Parameter $\\theta_2$', fontsize=12)
ax.set_zlabel('Loss $L(\\theta_1, \\theta_2)$', fontsize=12)
ax.set_title('Neural Network Optimization Landscape', fontsize=14, weight='bold')

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Add legend
ax.legend(loc='upper left')

# Set viewing angle
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
plt.show()
