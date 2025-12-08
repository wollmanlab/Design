# Data_Format Directory

This directory contains scripts for processing single-cell RNA sequencing (scRNA-seq) data from various sources into standardized training datasets for machine learning models.

## Overview

The scripts in this directory process raw scRNA-seq data (stored as AnnData `.h5ad` files) and convert them into PyTorch tensors suitable for training. Each script follows a similar pipeline:

1. **Data Loading**: Load expression matrices and metadata from AnnData files
2. **Gene Filtering**: Filter genes based on probe availability and shared genes across datasets
3. **Gene Constraints**: Generate constraints based on probe counts from reference data
4. **Normalization**: Scale expression data to 100,000 counts per cell
5. **Class Balancing**: Balance cell type representation in the dataset
6. **Train/Test Split**: Split data into training and testing sets
7. **Data Export**: Save processed data as PyTorch tensors and metadata as CSV files

## Scripts

### `data_format.py`

Processes the Allen Whole Mouse Brain (WMB) dataset from March 2024.

**Input Data:**
- Base path: `/u/project/rwollman/data/Allen_WMB_2024Mar06`
- Multiple 10X v2 and v3 expression matrices
- Cell metadata with subclass and region labels

**Key Features:**
- Handles multiple feature matrices (10Xv2 and 10Xv3)
- Processes both cell type (`subclass`) and region (`region_of_interest_acronym`) labels
- Uses ensemble gene IDs with mapping to gene symbols
- Outputs region labels (`r_train.pt`, `r_test.pt`) in addition to cell type labels

**Output Files:**
- `X_train.pt`, `X_test.pt`: Expression matrices (PyTorch tensors)
- `y_train.pt`, `y_test.pt`: Cell type labels (PyTorch tensors)
- `r_train.pt`, `r_test.pt`: Region labels (PyTorch tensors)
- `categorical_converter.csv`: Mapping from cell type names to integer labels
- `region_categorical_converter.csv`: Mapping from region names to integer labels
- `constraints.csv`: Gene probe constraints

**Output Path:** `/u/home/r/rwollman/project-rwollman/atlas_design/Design/Training_data/`

### `data_format_lymph.py`

Processes mouse lymph node datasets by combining multiple `.h5ad` files.

**Input Data:**
- Base path: `/u/home/z/zeh/rwollman/data/Mouse_Lymph/Mouse_LN_Datasets`
- Multiple `.h5ad` files in the directory

**Key Features:**
- Combines multiple datasets into a single AnnData object
- Finds shared genes across all datasets
- Standardizes cell type labels (e.g., `T_cell_CD4` → `CD4+ T cell`)
- Filters out doublet cells
- Handles datasets with different cell type column names

**Output Files:**
- `X_train.pt`, `X_test.pt`: Expression matrices (PyTorch tensors)
- `y_train.pt`, `y_test.pt`: Cell type labels (PyTorch tensors)
- `categorical_converter.csv`: Mapping from cell type names to integer labels
- `constraints.csv`: Gene probe constraints
- `cell_metadata.csv`: Full cell metadata

**Output Path:** `{base_path}/Training_data/`

**Usage:**
```bash
qsub /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/sub_python_script.sh /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/data_format_lymph.py
```

### `data_format_dev_mouse.py`

Processes developmental mouse scRNA-seq data from a single dataset.

**Input Data:**
- Base path: `/u/home/z/zeh/rwollman/data/Dev_Mouse`
- Single `.h5ad` file: `a5a85963-8004-41a1-8eb5-ca65266d89c3.h5ad`

**Key Features:**
- Converts ensemble gene IDs to gene symbols
- Uses `author_cell_type` as the cell type label
- Processes larger datasets (500,000 cells target)
- Memory-efficient processing with garbage collection

**Output Files:**
- `X_train.pt`, `X_test.pt`: Expression matrices (PyTorch tensors)
- `y_train.pt`, `y_test.pt`: Cell type labels (PyTorch tensors)
- `categorical_converter.csv`: Mapping from cell type names to integer labels
- `constraints.csv`: Gene probe constraints
- `cell_metadata.csv`: Full cell metadata with library size and scaling factors

**Output Path:** `{base_path}/Training_data/`

**Usage:**
```bash
qsub /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/sub_python_script.sh /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/data_format_dev_mouse.py
```

### `create_type_tree.py`

Creates a hierarchical clustering tree of cell types based on average expression profiles.

**Input Data:**
- Training data from Allen WMB dataset: `/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/`

**Process:**
1. Calculates average expression for each cell type
2. Selects top 1000 genes with highest variance/mean ratio across cell types
3. Performs hierarchical clustering using Ward's method with Euclidean distance
4. Generates child-parent relationships for the tree structure

**Output Files:**
- `linkage_matrix.npy`: Scipy linkage matrix for hierarchical clustering
- `child_parent_relationships.csv`: CSV file mapping child cell types/clusters to parent clusters

**Usage:**
Run as a Jupyter notebook or Python script after generating training data with `data_format.py`.

## Common Dependencies

All scripts require:
- `numpy`
- `pandas`
- `torch` (PyTorch)
- `sklearn` (scikit-learn)
- `anndata` (for reading `.h5ad` files)
- `scipy` (for `create_type_tree.py`)

## Gene Constraints

All scripts generate gene constraints based on probe availability from a reference file:
- **Reference file**: `/u/project/rwollman/data/mm10_probes_Oct28_2022.csv` (or `/scratchdata1/ExternalData/mm10_probes_Oct28_2022.csv`)
- Constraints represent the number of available probes per gene (clipped to 0-100)
- Only genes with constraints > 0 are typically used in downstream analysis

## Data Processing Details

### Normalization
All scripts normalize expression data to 100,000 counts per cell:
```python
scaling_factor = 100000 / library_size
X_normalized = X * scaling_factor
```

### Class Balancing
Cell types are balanced to ensure equal representation:
- Target number of cells per type: `n = total_cells / num_cell_types`
- If a cell type has fewer cells than `n`, sampling is done with replacement
- If a cell type has more cells than `n`, sampling is done without replacement

### Train/Test Split
- Test size: 20% (0.2)
- Random state: 42 (for reproducibility)
- Stratified split ensures balanced representation in both sets

## Output Format

### PyTorch Tensors
- `X_train.pt`, `X_test.pt`: Float tensors of shape `(n_cells, n_genes)`
- `y_train.pt`, `y_test.pt`: Long tensors of shape `(n_cells,)` containing integer cell type labels

### CSV Files
- `categorical_converter.csv`: Maps cell type names (index) to integer labels (column: `label`)
- `constraints.csv`: Maps gene names (index) to probe constraints (column: `constraints`)
- `cell_metadata.csv`: Full cell metadata with all original annotations

## Notes

- All scripts use hardcoded paths that may need to be updated for different environments
- Scripts are designed to run on cluster environments (HPC) with large memory requirements
- Some scripts include Jupyter notebook cell markers (`# %%`) for interactive development
- Memory management is important for large datasets; `data_format_dev_mouse.py` includes explicit garbage collection

