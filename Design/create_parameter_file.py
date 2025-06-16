# %%
""" Set Parameters """
import pandas as pd
import os
import itertools
from datetime import datetime
import getpass

# find user
current_user = getpass.getuser()
print(f"User identified as : {current_user}")
if current_user=='rwollman':
    base_dir = '/u/home/r/rwollman/project-rwollman/atlas_design/Runs/'
    input_dir = '/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/'
elif current_user=='zeh':
    base_dir = '/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs/'
    input_dir = '/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/'
else:
    base_dir = '/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs/'
    input_dir = '/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/'

print(f"base_dir: {base_dir}")
print(f"input_dir: {input_dir}")

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
            'n_cpu': 12,
            'n_bit': 24,
            'n_iterations': 50000, # any more than  100k will take more than 6 hours
            'total_n_probes': 30e4,
            'probe_weight': 1.0,
            'probe_under_weight_factor': 0.05,
            'weight_dropout_proportion': 0.1,
            'projection_dropout_proportion': 0.1,
            'gene_dropout_proportion':0.1,
            'gene_constraint_weight': 1.0,
            'target_brightness_log': 4.5,
            'target_brightness_weight':1.0,
            'tanh_slope_factor': 1.0, # Default slope factor
            'learning_rate': 0.05,
            'learning_rate_start': 0.1,
            'learning_rate_end': 0.01,
            'report_freq': 500,
            'type_correlation_mean_weight': 0.0, 
            'type_correlation_max_weight': 0.0, 
            'constant_noise': 3.0,
            'gene_fold_noise': 0.5,
            'categorical_weight': 2.0, 
            'batch_size': 2500,
            'correlation_thresh': 0.9,
            'pnorm_std_weight': 0, 
            'hierarchical_scatter_weight': 0.0,
            'y_hierarchy_file_path': 'child_parent_relationships.csv',  # Path to the file defining cell type hierarchy
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results', # Example, will be overridden per job
            'input':input_dir,
            'intra_type_variance_weight': 0.0,
            'bit_iqr_variance_weight': 0,
            'type_entropy_weight': 0.0,
            'decoder_hidden_layers': 0,
            'decoder_hidden_dim': 64,
            'decoder_dropout_rate': 0.3,
        }


# Define parameter variants - parameters to vary and their possible values
parameter_variants = {
    'learning_rate_start' : [0.1],
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
    param_desc_list = [] # Use a list to build description parts
    for j, param_name in enumerate(param_names):
        current_params[param_name] = combination[j]
        # Sanitize parameter values for filename (e.g., replace dots with 'p')
        value_str = str(combination[j]).replace('.', 'p')
        param_desc_list.append(f"{param_name}_{value_str}")
    
    # Check if learning_rate_end is greater than learning_rate_start
    if current_params['learning_rate_end'] > current_params['learning_rate_start']:
        # If end is greater than start, make end equal to start
        current_params['learning_rate_end'] = current_params['learning_rate_start']
        # Add this adjustment to the parameter description
        param_desc_list.append(f"lr_end_adjusted")

    # Create a unique identifier for the run based on combination and perhaps a timestamp or run_dir
    # The 'output' directory will be specific to each run.
    # We can base it on the param_desc_list and the main run_dir.
    
    run_specific_identifier = '_'.join(param_desc_list)
    
    # Create the base name for the parameter file and the corresponding output directory.
    # This ensures consistency with how sub_multi_param_file_optimization.sh constructs paths.
    param_file_name_base = f"params_{run_specific_identifier}" 
    
    # Update the 'output' path to be unique for this parameter combination.
    # This path will be written into the CSV and is what EncodingDesigner.py will use.
    # It should be: <base_dir>/design_results/params_<run_specific_identifier>
    correct_output_dir = os.path.join(base_dir, 'design_results', param_file_name_base)
    current_params['output'] = correct_output_dir
    
    # Create this directory. The .sh script might also try to create it (mkdir -p handles this).
    os.makedirs(correct_output_dir, exist_ok=True)

    # Define the full path for the CSV parameter file, which goes into 'params_files_to_scan'
    fullfilepath = os.path.join(input_param_path, f"{param_file_name_base}.csv")
    
    # Save parameter file
    pd.DataFrame(current_params.values(), index=current_params.keys(), columns=['values']).to_csv(fullfilepath)

print(f"Generated {len(combinations)} parameter files in {input_param_path}")