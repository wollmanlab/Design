# Design

## Overview

This repository contains **CIPHER** (Cell Identity Projection using Hybridization Encoding Rules), a deep learning framework for designing multiplexed in situ hybridization (ISH) probe sets that can accurately identify cell types from gene expression data.

## What is CIPHER?

**CIPHER** stands for **Cell Identity Projection using Hybridization Encoding Rules**. It is a PyTorch-based neural network system that learns to encode high-dimensional gene expression data into a low-dimensional "bit" representation (projection space) that can be measured using multiplexed ISH probes. The system simultaneously optimizes:

1. **Probe Design**: Determines how many probes to allocate to each gene for each bit
2. **Cell Type Classification**: Learns to decode the bit projections back to cell type identities
3. **Experimental Constraints**: Respects biological and experimental constraints (e.g., maximum probes per gene, target brightness, dynamic range)

### Key Concepts

- **Encoding**: Maps gene expression vectors (n_genes dimensions) to bit projections (n_bit dimensions, typically 3-96 bits)
- **Projection**: The bit values represent the expected signal intensity for each bit in the multiplexed ISH experiment
- **Decoding**: Reconstructs cell type identity from the bit projections

CIPHER enforces three sets of rules through its loss functions:

1. **LA (Accuracy Rules)**: Ensures the ability to decode the correct cell type from the projections
2. **LH (Hybridization Rules)**: Enforces constraints on gene-level probe limits and total number of probes
3. **LM (Measurability Rules)**: Ensures the projections can be accurately measured experimentally:
   - **Brightness**: Signal intensity must reach detectable levels
   - **Dynamic Range**: Sufficient fold-change between low and high expression levels
   - **Separability**: Different cell types must have sufficiently distinct projection patterns

Additionally, CIPHER includes **robustness/training rules** that don't directly affect accuracy, hybridization, or measurability but ensure robust design and prevent overfitting:
- **Training regularization**: Bit usage, bit correlation, sparsity, and gene importance constraints
- **Noise injection**: Various noise terms applied during training to simulate experimental conditions and minimize overfitting

## Architecture

CIPHER consists of two main components:

### 1. Encoder (Projection Layer)
- **Type**: Embedding layer (n_genes × n_bit)
- **Purpose**: Maps each gene to a contribution weight for each bit
- **Activation**: Configurable (tanh, sigmoid, linear, relu) with constraints applied
- **Output**: Encoding weights E (n_genes × n_bit) representing probe fractions

The encoder weights are constrained by:
- Gene-level constraints (maximum probes per gene)
- Total probe budget (target total number of probes)
- Activation functions that ensure valid probe fractions

### 2. Decoder (Classification Network)
- **Type**: Multi-layer neural network
- **Input**: Bit projections P (n_samples × n_bit)
- **Output**: Cell type logits (n_samples × n_categories)
- **Architecture**: Configurable number of hidden layers (0-3+)
- **Activation**: Configurable (relu, leaky_relu, gelu, swish, tanh)

### Workflow

```
Gene Expression (X) → Encoder (E) → Projection (P) → Decoder → Cell Type Prediction
     (n_genes)         (n_genes×n_bit)   (n_bit)        (n_categories)
```

1. **Projection**: `P = X @ E` (matrix multiplication of gene expression with encoding weights)
2. **Normalization**: Optional sum normalization or bit-wise normalization
3. **Noise Injection**: During training, various noise types are applied to simulate experimental conditions
4. **Decoding**: `R = Decoder(P)` → cell type predictions

## Loss Functions and Optimization Objectives

CIPHER optimizes multiple objectives simultaneously through a weighted loss function. The losses are organized into three main rule categories (LA, LH, LM) plus additional robustness/training rules:

### LA: Accuracy Rules

These losses focus on the ability to decode the correct cell type from the projections:

1. **Categorical Classification Loss** (`categorical_wt`)
   - Cross-entropy loss with label smoothing
   - Ensures accurate cell type identification from projections
   - Primary measure of classification accuracy

### LH: Hybridization Rules

These losses enforce constraints related to probe design and hybridization:

5. **Gene Constraint Loss** (`gene_constraint_wt`)
   - Ensures no gene exceeds its maximum allowed probe count
   - Critical for respecting biological/experimental constraints
   - Enforces per-gene probe limits

6. **Probe Count Loss** (`probe_wt`)
   - Penalizes deviations from target total probe count (`n_probes`)
   - Uses ELU activation to allow flexibility while encouraging target
   - Enforces total probe budget constraint

### LM: Measurability Rules

These losses ensure that the projections can be accurately measured in experimental conditions:

7. **Brightness Loss** (`brightness_wt`)
   - Ensures median signal brightness reaches target level (log10 scale)
   - Prevents signals from being too dim to detect experimentally

8. **Dynamic Range Loss** (`dynamic_wt`)
   - Encourages sufficient fold-change between low and high expression levels
   - Ensures bits can distinguish different expression states

9. **Separation Loss** (`separation_wt`)
   - Ensures minimum fold-change between different cell type projections
   - Critical for distinguishing cell types

10. **Step Size Loss** (`step_size_wt`)
    - Ensures minimum step sizes between cell types in projection space
    - Improves separability and ensures measurable differences between cell types

### Robustness and Training Rules

These rules don't directly affect accuracy, hybridization, or measurability but ensure robust design and prevent overfitting:

10. **Bit Usage Loss** (`bit_usage_wt`)
    - Encourages decoder to use all bits
    - Prevents bit collapse and ensures all bits contribute to classification
    - Promotes robust encoding by preventing unused bits

11. **Bit Correlation Loss** (`bit_corr_wt`)
    - Penalizes high correlation between bits
    - Ensures bits capture independent information
    - Prevents redundant encoding and improves generalization

12. **Sparsity Loss** (`sparsity_wt`)
    - Encourages sparse encoding weights (many zeros)
    - Reduces complexity and probe requirements
    - Promotes simpler, more generalizable designs

13. **Gene Importance Loss** (`gene_importance_wt`)
    - Prevents any single gene from dominating a bit (>25% contribution)
    - Promotes distributed encoding and reduces dependency on single genes
    - Improves robustness to gene expression variation

### Noise Terms (Regularization, Not Loss Terms)

CIPHER includes extensive noise injection during training to simulate experimental conditions and prevent overfitting. These are not loss terms but regularization mechanisms:

- **Gene-level noise**: `X_drp_s/e` (dropout), `X_noise_s/e` (expression noise)
- **Weight-level noise**: `E_drp_s/e` (encoding weight dropout), `E_noise_s/e` (weight noise)
- **Projection-level noise**: `P_drp_s/e` (projection dropout), `P_noise_s/e` (measurement noise)
- **Constant noise**: `P_add_s/e` (background signal, log10 scale)
- **Decoder dropout**: `D_drp_s/e`

These noise terms force the model to learn robust encodings that work under various experimental conditions, minimizing overfitting to the training data.

## Key Parameters

### Core Model Parameters

- `n_cpu`: Number of CPU threads to use for PyTorch (typically 3-12). Note: This codebase runs on CPU, not GPU/CUDA
- `n_bit`: Number of bits in the encoding (dimensionality of projection space). Typical values: 3-96
- `n_iters`: Total number of training iterations. Typical values: 10,000-100,000+
- `batch_size`: Batch size for training (0 = use full dataset)
- `n_probes`: Target total number of probes across all genes (e.g., 50,000, 500,000)
- `decoder_n_lyr`: Number of hidden layers in decoder (0-3+)

### Loss Function Weights

- `categorical_wt`: Weight for classification accuracy (typically 1-3)
- `probe_wt`: Weight for probe count constraint (typically 0-1)
- `gene_constraint_wt`: Weight for gene constraint violations (typically 1)
- `brightness_wt`: Weight for target brightness (typically 1)
- `dynamic_wt`: Weight for dynamic range (typically 0-1)
- `separation_wt`: Weight for cell type separation (typically 1)

### Training Parameters

- `lr_s` / `lr_e`: Initial and final learning rates (linear interpolation)
- `gradient_clip`: Maximum gradient norm for clipping
- `saturation`: When to reach final values for all _s/_e parameters (0.0-1.0)
- `report_rt`: How often to report training progress (iterations)

### Brightness and Signal Parameters

- `brightness_s` / `brightness_e`: Initial and final target brightness (log10 scale)
- `dynamic_fold_s` / `dynamic_fold_e`: Initial and final target fold change for dynamic range
- `separation_fold_s` / `separation_fold_e`: Initial and final minimum fold change between cell types

### Noise Parameters (for Robustness)

CIPHER includes extensive noise modeling to simulate experimental conditions:

- **Gene-level noise**: `X_drp_s/e` (dropout), `X_noise_s/e` (expression noise)
- **Weight-level noise**: `E_drp_s/e` (encoding weight dropout), `E_noise_s/e` (weight noise)
- **Projection-level noise**: `P_drp_s/e` (projection dropout), `P_noise_s/e` (measurement noise)
- **Constant noise**: `P_add_s/e` (background signal, log10 scale)
- **Decoder dropout**: `D_drp_s/e`

### Data Parameters

- `X_train` / `X_test`: Paths to training/test feature tensors (.pt files)
- `y_train` / `y_test`: Paths to training/test label tensors (.pt files)
- `constraints`: Path to gene constraints CSV file
- `y_label_converter_path`: Path to categorical label mapping CSV
- `top_n_genes`: Number of top genes to keep (0 = keep all)

### Advanced Parameters

- `device`: Device to run computations on ('cpu' - default, this codebase is designed for CPU execution)
- `encoder_act`: Activation function for encoding weights ('tanh', 'sigmoid', 'linear', 'relu')
- `decoder_act`: Activation function for decoder ('relu', 'leaky_relu', 'gelu', 'swish', 'tanh')
- `sum_norm`: Whether to normalize projection by sum (0 or 1)
- `bit_norm`: Whether to normalize projection by bit-wise statistics (0 or 1)
- `use_noise`: Whether to apply noise/dropout during training (0 or 1)
- `continue_training`: Whether to continue training if model loaded from file (0 or 1)
- `best_model`: Whether to save the best model during training (0 or 1)

## Installation

### Create Conda Environment

To set up the necessary environment, follow these steps:

1. **Create the Conda environment:**
   ```bash
   conda create -n "designer_3.12" python=3.12
   ```

2. **Activate the Conda environment:**
   ```bash
   conda activate designer_3.12
   ```

3. **Install required packages:**
   ```bash
   pip install torch
   pip install scikit-learn
   pip install anndata
   pip install ipykernel
   pip install matplotlib
   pip install seaborn
   pip install scanpy
   ```

## Usage

### Running Single Jobs

Follow these steps to execute CIPHER on a single parameter configuration:

1. **Format Data:**
   - Open and run the Jupyter Notebook located at `/Design/Notebooks/data_format.ipynb`
   - This prepares your gene expression data into the required format (X_train.pt, X_test.pt, y_train.pt, y_test.pt)

2. **Format Parameters:**
   - Open and run the Jupyter Notebook located at `/Design/Notebooks/parameters_format.ipynb`
   - This creates a CSV parameter file with all necessary configuration

3. **Run CIPHER:**
   - Execute the main script using the following command, replacing `"path/to/parameters/file"` with the actual path to your parameters CSV file:
   ```bash
   python /Design/CIPHER.py "path/to/parameters/file"
   ```

   The script will:
   - Load parameters from the CSV file
   - Initialize the encoder and decoder
   - Train the model for the specified number of iterations
   - Save model checkpoints, evaluation results, and visualizations
   - Generate comprehensive performance reports

### Running Multiple Jobs (Parameter Sweeps)

For systematic exploration of parameter space, use the batch processing workflow:

1. **Format Reference Scripts:**
   - `data_format.py` - Formats the data for the reference dataset
     - **Parameters to change:** `data_path`, `output_path`, `csv_file`
     - ```bash
       conda activate designer_3.12
       python ./Design/data_format.py
       ```
   - `create_type_tree.py` - Creates the type tree for the reference (if needed)

2. **Create Parameter Files and Submit Jobs:**
   - `create_parameter_file.py` - Creates parameter files for all combinations and automatically submits jobs
     - **Parameters to change:** `base_dir`, `parameter_variant_list` (in the script)
     - ```bash
       conda activate designer_3.12
       python ./Design/create_parameter_file.py [Run#]
       ```
     - **Run number:** You can specify a run number (e.g., `Run0`, `Run1`) or let the script automatically find the next available run number
     - This script:
       - Generates all parameter combinations from `parameter_variant_list`
       - Creates CSV parameter files in `Runs/Run#/params_files_to_scan/`
       - Automatically calls `sub_multi_param_file_optimization.sh` to submit jobs
   
   - `sub_multi_param_file_optimization.sh` - Submits and manages batch jobs
     - **Parameters to change:** `OPT_DIR`, `CODE_DIR`, replace `Run0` with your run number
     - **Note:** This script is automatically called by `create_parameter_file.py`, but can be run manually:
     - ```bash
       ./Design/sub_multi_param_file_optimization.sh Run0
       ```

### Parameter File Format

Parameter files are CSV files with two columns:
- Column 1: Parameter name (index)
- Column 2: Parameter value

Example:
```csv
values
n_cpu,3
n_bit,24
n_iters,100000
batch_size,500
...
```

### Input Data Format

CIPHER expects the following input files:

1. **Gene Expression Data:**
   - `X_train.pt`: PyTorch tensor of shape (n_train_samples, n_genes) - training gene expression
   - `X_test.pt`: PyTorch tensor of shape (n_test_samples, n_genes) - test gene expression

2. **Cell Type Labels:**
   - `y_train.pt`: PyTorch tensor of shape (n_train_samples,) - training cell type labels (integer indices)
   - `y_test.pt`: PyTorch tensor of shape (n_test_samples,) - test cell type labels (integer indices)

3. **Gene Constraints:**
   - `constraints.csv`: CSV file with maximum allowed probes per gene
   - Format: One column with constraint values (one per gene, in same order as genes in X)

4. **Label Mapping:**
   - `categorical_converter.csv`: CSV file mapping integer labels to cell type names
   - Format: Two columns - integer index and cell type name

### Output Files

CIPHER generates comprehensive output in the specified output directory:

1. **Model Files:**
   - `model_best.pt`: Best model checkpoint (if `best_model=1`)
   - `model_final.pt`: Final model checkpoint
   - `optimizer_state.pt`: Optimizer state

2. **Training Statistics:**
   - `learning_stats.csv`: Detailed training statistics for each iteration
   - `training_log.txt`: Comprehensive training log

3. **Evaluation Results:**
   - `evaluation_results.csv`: Final evaluation metrics
   - `P_test_averages.csv`: Average projections per cell type (no noise)
   - `confusion_matrix.csv`: Cell type classification confusion matrix

4. **Visualizations:**
   - `comprehensive_performance.pdf`: Training curves and performance metrics
   - `projection_heatmap.pdf`: Heatmap of projections by cell type
   - `confusion_matrix.pdf`: Visual confusion matrix

5. **Encoding Weights:**
   - `E_weights.csv`: Final encoding weights (probe fractions per gene per bit)
   - `E_weights.pt`: PyTorch tensor of encoding weights

## Understanding the Results

### Encoding Weights (E)
The encoding weights represent the learned probe design:
- Each row corresponds to a gene
- Each column corresponds to a bit
- Values represent the fraction of probes allocated to that gene for that bit
- Values are constrained by gene constraints and activation functions

### Projections (P)
The projections represent the expected signal for each bit:
- Each row is a cell/sample
- Each column is a bit
- Values represent expected signal intensity (can be normalized)

### Performance Metrics
- **Accuracy**: Cell type classification accuracy
- **Separation**: Minimum fold-change between cell type projections
- **Dynamic Range**: Fold-change between low and high expression levels
- **Brightness**: Median signal intensity
- **Probe Count**: Total number of probes used

## Tips and Best Practices

1. **Start with fewer bits**: Begin with 12-24 bits to understand the system before scaling up
2. **Monitor training**: Check `learning_stats.csv` and logs regularly
3. **Adjust loss weights**: If one objective isn't being met, increase its weight
4. **Use noise during training**: Set `use_noise=1` to improve robustness
5. **Respect constraints**: Ensure `gene_constraint_wt` is high enough to prevent violations
6. **Check convergence**: Look for plateau in loss and accuracy metrics
7. **Save best model**: Set `best_model=1` to automatically save the best checkpoint

## Troubleshooting

- **Training diverges**: Reduce learning rate, increase gradient clipping, or reduce loss weights
- **Constraints violated**: Increase `gene_constraint_wt` or check constraint file format
- **Low accuracy**: Increase `categorical_wt`, check data quality, or increase model capacity
- **Out of memory**: Reduce batch size, number of genes, or reduce `n_cpu` threads
- **Noise too high**: Reduce noise parameters or set `use_noise=0` for debugging
