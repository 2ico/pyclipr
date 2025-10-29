
import matplotlib.pyplot as plt
import numpy as np

import pyclipr

# Define simpler sample paths
# A U-shape with a beveled top-left corner
path = np.array([(0, 0), (0, 90), (10, 100), (30, 100), (30, 2), (70, 30), (70, 100), (100, 100), (100, 0), (0, 0)])

# A rectangle for the pattern
path2 = np.array([(0, 0), (0, 4), (8, 4), (8, 0), (0, 0)])

# Test Minkowski Sum
sumOut = pyclipr.minkowskiSum(path2, path, isClosed=True)

# Test Minkowski Diff
diffOut = pyclipr.minkowskiDiff(path2, path, isClosed=True)

# Plot the results in subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Minkowski Operations Example')

# Minkowski Sum subplot
axs[0].set_title('Minkowski Sum')
axs[0].axis('equal')
axs[0].fill(path[:, 0], path[:, 1], 'b', alpha=0.1, label='Path', edgecolor='black')
axs[0].fill(path2[:, 0], path2[:, 1], 'r', alpha=0.1, label='Pattern', edgecolor='black')
for poly in sumOut:
    axs[0].fill(poly[:, 0], poly[:, 1], 'g', alpha=0.5, label='Sum' if 'Sum' not in axs[0].get_legend_handles_labels()[1] else '', edgecolor='black')
axs[0].legend()

# Minkowski Diff subplot
axs[1].set_title('Minkowski Difference')
axs[1].axis('equal')
axs[1].fill(path[:, 0], path[:, 1], 'b', alpha=0.1, label='Path', edgecolor='black')
axs[1].fill(path2[:, 0], path2[:, 1], 'r', alpha=0.1, label='Pattern', edgecolor='black')
for poly in diffOut:
    axs[1].fill(poly[:, 0], poly[:, 1], 'purple', alpha=0.5, label='Diff' if 'Diff' not in axs[1].get_legend_handles_labels()[1] else '', edgecolor='black')
axs[1].legend()

# plt.show()
