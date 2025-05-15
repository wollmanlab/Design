# %%
""" Set Parameters """
import pandas as pd
import os
import itertools
from datetime import datetime

# Create output directory for parameter files and all other requires directories under base_dir
base_dir = '/u/home/r/rwollman/project-rwollman/atlas_design/Runs/'
# Check if a command line argument was provided for the run directory
import sys
if len(sys.argv) > 1:
    run_dir = sys.argv[1]
    base_dir = os.path.join(base_dir, run_dir)
else:
    # Exit with an error message if no run directory is specified
    print("Error: You must provide a run directory name as an argument.")
    print("Usage: python create_parameter_file.py <run_directory_name>")
    sys.exit(1)

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)


input_param_path = os.path.join(base_dir,'params_files_to_scan')
os.makedirs(input_param_path, exist_ok=True)
os.makedirs(os.path.join(base_dir, 'params_files_scanned'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'design_results'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'job_logs'), exist_ok=True)

# Default parameter values
user_parameters = {
            'device': 'cpu',
            'Verbose': 1,
            'n_cpu': 1,
            'n_bit': 25,
            'n_iterations': 5000,
            'total_n_probes': 30e4,
            'probe_weight': 1,
            'probe_under_weight_factor': 0.05,
            'weight_dropout_proportion': 0.1,
            'projection_dropout_proportion': 0.1,
            'gene_constraint_weight': 1,
            'target_brightness_log': 4.5,
            'learning_rate': 0.05, 
            'learning_rate_start': 0.1,
            'learning_rate_end': 0.01,
            'report_freq': 500,
            'type_correlation_mean_weight': 0,
            'type_correlation_max_weight': 0.01,
            'noise_level': 3,
            'categorical_weight': 1,
            'batch_size': 2500,
            'use_region_info': 0, #region decoders (1=yes, 0=no)
            'region_embedding_dim': 0, 
            'correlation_thresh': 0.9,
            'pnorm_std_weight': 10,
            'hierarchical_scatter_weight': 0.5,  # Weight for the new hierarchical scatter loss
            'y_hierarchy_file_path': 'child_parent_relationships.csv',  # Path to the file defining cell type hierarchy
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results',
            'input':'/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/'
        }

# Define parameter variants - parameters to vary and their possible values
parameter_variants = {
    'hierarchical_scatter_weight': [0, 0.01],
    'type_correlation_max_weight' : [0,0.01]
}

# Generate all parameter combinations
param_names = list(parameter_variants.keys())
param_values = list(parameter_variants.values())

# Generate all combinations of parameter values
combinations = list(itertools.product(*param_values))
for i, combination in enumerate(combinations):
    # Create a copy of default parameters
    current_params = user_parameters.copy()
    
    # Update with current combination
    param_desc = []
    for j, param_name in enumerate(param_names):
        current_params[param_name] = combination[j]
        param_desc.append(f"{param_name}_{combination[j]}")
    
    # Create parameter file name
    param_file_name = f"params_{'_'.join(param_desc)}"
    fullfilepath = os.path.join(input_param_path, f"{param_file_name}.csv")
    
    # Save parameter file
    pd.DataFrame(current_params.values(), index=current_params.keys(), columns=['values']).to_csv(fullfilepath)

print(f"Generated {len(combinations)} parameter files in {input_param_path}")


