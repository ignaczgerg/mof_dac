import numpy as np
import matplotlib.pyplot as plt

# Define the number of nodes (including node 0)
num_nodes = 13  # Since node indexing starts at 0

# Given edge_index tensor (adjusted for 0-based indexing)
edge_index = np.array([[1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  4,  7,  7,  8,  8,  9,  9, 10,
                         10, 11, 11, 12,  6,  1, 12,  7],
                        [2,  1,  3,  2,  4,  3,  5,  4,  6,  5,  7,  4,  8,  7,  9,  8, 10,  9,
                         11, 10, 12, 11,  1,  6,  7, 12]])

# Create an adjacency matrix (including node 0)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# Fill adjacency matrix using edge_index
for i in range(edge_index.shape[1]):
    row, col = edge_index[:, i]
    adj_matrix[row, col] = 1  # Undirected graph
    adj_matrix[col, row] = 1  # Ensure symmetry

# Create a visualization of the adjacency matrix
fig, ax = plt.subplots(figsize=(8, 8))

# Display full adjacency matrix
ax.matshow(adj_matrix, cmap="Blues")

# Set background color to white
ax.set_facecolor("white")

# Increase font sizes
ax.set_xticks(np.arange(num_nodes))
ax.set_yticks(np.arange(num_nodes))
ax.set_xticklabels(np.arange(num_nodes), fontsize=20)
ax.set_yticklabels(np.arange(num_nodes), fontsize=20)

# Move x-axis labels to the bottom
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')

# Set title with increased font size
plt.title(" Adjacency Matrix", fontsize=20, pad=20)

# Save figure for PowerPoint
plt.savefig("adjacency_matrix.png", dpi=300, bbox_inches='tight')
