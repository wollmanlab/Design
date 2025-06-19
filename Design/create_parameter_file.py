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

user_parameters = {
            'n_cpu': 12,
            'n_bit': 24,
            'n_iterations': 50000,
            'batch_size': 2500,
            'target_brightness_log': 4.5,
            'total_n_probes': 30e4,
            'probe_weight': 1.0,
            'probe_under_weight_factor': 0.1,
            'gene_constraint_weight': 1.0,
            'target_brightness_weight':1.0,
            'gradient_clip_max_norm': 1.0, # Added for gradient clipping
            'learning_rate_start': 0.05, 
            'learning_rate_end': 0.005, 
            'report_freq': 100,
            'sparsity_target': 0.8, # Target sparsity ratio (80% zeros)
            'sparsity_weight': 0.0, # Weight for sparsity loss (increased from 0.1)
            'categorical_weight': 1.0,
            'weight_dropout_proportion_start': 0.0,
            'weight_dropout_proportion_end': 0.1,
            'projection_dropout_proportion_start': 0.0,
            'projection_dropout_proportion_end': 0.1,
            'gene_dropout_proportion_start': 0.0,
            'gene_dropout_proportion_end': 0.1,
            'decoder_dropout_rate_start': 0.0,
            'decoder_dropout_rate_end': 0.1,
            'constant_noise_start': 1.0,
            'constant_noise_end': 3.0,
            'gene_fold_noise_start': 0.0,
            'gene_fold_noise_end': 0.5,
            'perturbation_frequency': 500, # How often to perturb weights (every N iterations)
            'perturbation_percentage': 0.01, # Percentage of weights to perturb (0.0-1.0)
            'device': 'cpu',
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results',
            'input': './', 
            'Verbose': 1,
            'decoder_hidden_layers': 0,
            'decoder_hidden_dim': 128,
            'constraints': 'constraints.csv', 
            'X_test': 'X_test.pt',            
            'y_test': 'y_test.pt',            
            'X_train': 'X_train.pt',          
            'y_train': 'y_train.pt',          
            'y_label_converter_path': 'categorical_converter.csv', 
        }

user_parameters['input'] = input_dir
# Define parameter variants - parameters to vary and their possible values
parameter_variants = {
    'perturbation_frequency':[0,1000],
    'gene_fold_noise_end':[0.0,0.5],
    'gene_dropout_proportion_end':[0.0,0.1],
    'decoder_dropout_rate_end':[0.0,0.1],
    'constant_noise_end':[0.0,3.0],
    'weight_dropout_proportion_end':[0.0,0.1],
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