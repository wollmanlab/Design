# Results Directory

This directory contains scripts for analyzing, aggregating, and visualizing results from CIPHER design runs. These tools help evaluate probe design performance, collect results across multiple parameter sweeps, and generate publication-quality figures.

## Overview

The Results directory provides a complete workflow for:
1. **Simulation Analysis** - Testing probe designs on simulated experimental data
2. **Result Aggregation** - Collecting results from multiple design runs
3. **Figure Generation** - Creating visualizations of design performance metrics

## Scripts

### 1. `simulation.py`

**Purpose**: Runs comprehensive simulation analysis to evaluate probe design performance on simulated experimental data.

**What it does**:
- Projects reference and simulation datasets using learned encoding weights
- Performs cell type classification using SCALE (Single Cell Alignment Leveraging Expectations)
- Tests multiple noise levels and model types (Logistic Regression, MLP)
- Evaluates classification accuracy under various experimental conditions
- Generates harmonized projections and classification results

**Key Features**:
- Processes large-scale reference datasets (Allen Brain Atlas)
- Handles chunked data files for memory efficiency
- Supports parallel processing with ThreadPoolExecutor
- Performs quantile harmonization to align measured and reference data
- Tests both likelihood-only and full Bayesian classification approaches

**Usage**:
```bash
conda activate dredfish_3.9
python simulation.py "/path/to/design/results/directory" [--data_path PATH] [--ccf_x_min FLOAT] [--ccf_x_max FLOAT]
```

**Arguments**:
- `input_path` (required): Path to the design results directory containing `E_constrained.csv`
- `--data_path`: Path to external data directory (default: `/scratchdata1/ExternalData`)
- `--ccf_x_min`: Minimum CCF x coordinate for filtering (default: 0)
- `--ccf_x_max`: Maximum CCF x coordinate for filtering (default: 20)

**Output**:
- `Simulation/Reference/{design_name}.h5ad` - Processed reference data projections
- `Simulation/Simulation/{design_name}.h5ad` - Processed simulation data projections
- `Simulation/Results/{model_type}_{design_name}_{noise_level}.h5ad` - Classification results for each condition
- `Simulation/Results/{design_name}.json` - Summary accuracy metrics

**Example**:
```bash
python simulation.py "/scratchdata1/GeneralStorage/Zach/Designs/Sync/params_fig_Probe_Number_Tradeoff_36_Bits_decoder_n_lyr_0_n_probes_50000_n_bit_36_replicate_1"
```

### 2. `get_results.py`

**Purpose**: Aggregates results from multiple design runs into a single JSON file for analysis.

**What it does**:
- Scans all runs in a base directory
- Collects `Results.csv` files from each design
- Extracts key metrics: Probes, Accuracy, Separation, Dynamic Range, Dynamic Fold
- Collects parameter files (`used_user_parameters.csv`) for each design
- Combines results and parameters into a structured JSON output

**Usage**:
```bash
conda activate designer_3.12
python get_results.py
```

**Configuration**:
Update the `base_path` variable in the script to point to your runs directory:
```python
base_path = f"/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs/"
```

**Output**:
- `{base_path}/results.json` - JSON file containing all collected results and parameters

**Structure**:
```json
{
  "design_name_1": {
    "results": {
      "Probes": 50000,
      "Accuracy": 0.95,
      "Separation": 2.3,
      "Dynamic Range": 4.5,
      "Dynamic Fold": 3.2
    },
    "parameters": {
      "n_bit": 24,
      "n_probes": 50000,
      ...
    }
  },
  ...
}
```

### 3. `create_figures.py`

**Purpose**: Generates publication-quality figures visualizing design performance across parameter sweeps.

**What it does**:
- Reads aggregated results from multiple design runs
- Identifies varying parameters across runs
- Creates multi-axis plots showing:
  - **Probes**: Number of probes used
  - **Accuracy**: Classification accuracy
  - **Separation**: Cell type separation metric
  - **Dynamic Range**: Dynamic range metric
- Supports subplot organization by decoder layer number
- Generates PDF figures with proper formatting

**Usage**:
```bash
conda activate designer_3.12
python create_figures.py
```

**Configuration**:
Update the following in the script:
- `notebook_name`: List of run names to process (e.g., `['Run14']`)
- `base_path`: Path to runs directory
- `subplot_parameter`: Parameter to use for subplot organization (default: `'decoder_n_lyr'`)

**Output**:
- PDF figures saved to `/u/home/z/zeh/project-rwollman/Projects/Design/Figures/{figure_name}.pdf`

**Figure Features**:
- Multiple y-axes for different metrics (Probes, Accuracy, Separation, Dynamic Range)
- Color-coded lines for different decoder configurations
- Error bars showing standard deviation across replicates
- Scientific notation for large numbers
- Proper axis labels and legends

**Example Output**:
Figures show how design performance (accuracy, separation, dynamic range, probe count) varies with parameters like number of bits, number of probes, or separation weight, organized by decoder architecture.

## Workflow

### Complete Analysis Pipeline

1. **Run Designs**: Execute CIPHER optimization runs using `create_parameter_file.py` and submission scripts

2. **Run Simulations** (Optional):
   ```bash
   # For each design result directory
   python simulation.py "/path/to/design/results"
   ```

3. **Aggregate Results**:
   ```bash
   python get_results.py
   ```

4. **Generate Figures**:
   ```bash
   python create_figures.py
   ```

## Dependencies

### `simulation.py`
- Python 3.9+ (conda environment: `dredfish_3.9`)
- numpy, pandas, torch, anndata
- scikit-learn (LogisticRegression, MLPClassifier, RANSACRegressor)
- scipy (interpolation, clustering, filtering)
- matplotlib, tqdm
- requests (for downloading manifests)

### `get_results.py` and `create_figures.py`
- Python 3.12+ (conda environment: `designer_3.12`)
- pandas, numpy, matplotlib

## Key Concepts

### SCALE (Single Cell Alignment Leveraging Expectations)
The `simulation.py` script uses SCALE for cell type classification:
- **Spatial Priors**: Uses KDE-based spatial priors from reference atlases
- **Harmonization**: Quantile normalization to align measured and reference data
- **Hierarchical Classification**: Tree-based classification using dendrograms
- **Bayesian Integration**: Combines likelihoods with spatial priors for posterior probabilities

### Noise Levels
Simulations test multiple noise levels:
- `0`: No noise (ideal conditions)
- `10e2`: Low noise
- `10e3`: Medium noise
- `10e4`: High noise

### Model Types
- **LR**: Logistic Regression (linear classifier)
- **MLP**: Multi-Layer Perceptron (non-linear classifier)

## File Structure

```
Results/
├── simulation.py          # Main simulation analysis script
├── get_results.py         # Result aggregation script
├── create_figures.py      # Figure generation script
└── README.md             # This file
```

## Notes

- `simulation.py` processes large datasets and may require significant memory and time
- Results are cached - existing output files are skipped to allow resuming interrupted runs
- The scripts use hardcoded paths that may need adjustment for your system
- `create_figures.py` expects specific parameter naming conventions (e.g., `decoder_n_lyr` for subplot organization)

## Troubleshooting

**Simulation fails with memory errors**:
- The script processes data in chunks, but very large datasets may still cause issues
- Consider processing smaller subsets or increasing available memory

**get_results.py finds no results**:
- Check that `Results.csv` files exist in `{run}/design_results/{design}/results/`
- Verify the `base_path` is correct

**Figures look incorrect**:
- Ensure `get_results.py` has been run first to generate `results.json`
- Check that parameter names match between results and figure generation code
- Verify that the figure name doesn't contain special characters that need escaping

