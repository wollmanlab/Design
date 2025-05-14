# %%
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import anndata
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

# %%
# Define the output path where data was saved
input_path = "/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/"

# Load the training data
X = torch.load(f"{input_path}/X_train.pt")
y = torch.load(f"{input_path}/y_train.pt")

# Load the categorical converter for cell types
categorical_converter = pd.read_csv(f"{input_path}/categorical_converter.csv", index_col=0)

# Print shapes to verify data loaded correctly
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %%
# Calculate average expression for each cell type more efficiently
unique_cell_types = torch.unique(y)
num_cell_types = len(unique_cell_types)
num_genes = X.shape[1]

# Use torch.zeros to initialize tensors for sum and count
X_sum = torch.zeros((num_cell_types, num_genes))
X_count = torch.zeros(num_cell_types)

# Map cell type IDs to consecutive indices if they aren't already
idx_map = {cell_type.item(): i for i, cell_type in enumerate(unique_cell_types)}

# Single pass through the data
for i in range(len(y)):
    cell_type_idx = idx_map[y[i].item()]
    X_sum[cell_type_idx] += X[i]
    X_count[cell_type_idx] += 1

# Compute averages
X_avg = X_sum / X_count.unsqueeze(1)

print(f"X_avg shape: {X_avg.shape}")
print(f"Number of unique cell types: {num_cell_types}")

# %%
# Select top 1000 highest var/mean genes across cell types
gene_means = torch.mean(X_avg, dim=0)  # Mean of each gene across cell types
gene_variances = torch.var(X_avg, dim=0)  # Variance of each gene across cell types

# Calculate variance/mean ratio (coefficient of variation)
# Add small epsilon to avoid division by zero
epsilon = 1e-10
cv_ratio = gene_variances / (gene_means + epsilon)

# Get indices of top 1000 genes with highest var/mean ratio
_, top_cv_indices = torch.topk(cv_ratio, k=1000)

# Select only the top var/mean genes
X_avg_filtered = X_avg[:, top_cv_indices]

print(f"X_avg_filtered shape: {X_avg_filtered.shape}")
print(f"Min var/mean selected: {cv_ratio[top_cv_indices[-1]]:.4f}")
print(f"Max var/mean selected: {cv_ratio[top_cv_indices[0]]:.4f}")

# %%
# Perform hierarchical clustering using Ward's method with Euclidean distance

# Convert to numpy for scipy compatibility
X_avg_filtered_np = X_avg_filtered.numpy()

# Compute the distance matrix using Euclidean distance
dist_matrix = pdist(X_avg_filtered_np, metric='euclidean')

# Perform hierarchical clustering with Ward's method
linkage_matrix = sch.linkage(dist_matrix, method='ward')

print(f"Linkage matrix shape: {linkage_matrix.shape}")

# %%
# Optional: Save the linkage matrix for later use
np.save(f"{input_path}/linkage_matrix.npy", linkage_matrix)

# %%
# Create a child-parent relationship CSV file
# First, convert categorical_converter from DataFrame to dictionary
categorical_dict = categorical_converter.to_dict()['label']
# Create reverse mapping from numeric ID to cell type label
id_to_label = {v: k for k, v in categorical_dict.items()}

# Total number of original cell types
n_cell_types = X_avg_filtered.shape[0]
print(f"Number of cell types: {n_cell_types}")

# Create lists to store child-parent relationships
child_labels = []
parent_labels = []

# Create labels for the inner nodes of the tree (clusters)
cluster_labels = {i: id_to_label[i] for i in range(n_cell_types) if i in id_to_label}

# Process the linkage matrix to build the hierarchy
for i, (cluster1, cluster2, distance, count) in enumerate(linkage_matrix):
    cluster1, cluster2 = int(cluster1), int(cluster2)
    
    # New cluster ID is n_cell_types + i (per scipy's convention)
    new_cluster_id = n_cell_types + i
    
    # Create cluster label (parent name)
    cluster_label = f"Cluster_{i}"
    cluster_labels[new_cluster_id] = cluster_label
    
    # Add relationships for first cluster
    if cluster1 < n_cell_types:
        # It's an original cell type
        child_labels.append(id_to_label[cluster1])
        parent_labels.append(cluster_label)
    else:
        # It's an intermediate cluster
        child_labels.append(cluster_labels[cluster1])
        parent_labels.append(cluster_label)
    
    # Add relationships for second cluster
    if cluster2 < n_cell_types:
        # It's an original cell type
        child_labels.append(id_to_label[cluster2])
        parent_labels.append(cluster_label)
    else:
        # It's an intermediate cluster
        child_labels.append(cluster_labels[cluster2])
        parent_labels.append(cluster_label)

# Create DataFrame
child_parent_df = pd.DataFrame({
    'child_label': child_labels,
    'parent_label': parent_labels
})

# Save to CSV
child_parent_df.to_csv(f"{input_path}/child_parent_relationships.csv", index=False)

print(f"Created child-parent relationship CSV with {len(child_parent_df)} rows")
print(f"Sample of the child-parent relationships:")
print(child_parent_df.head())

# %%
