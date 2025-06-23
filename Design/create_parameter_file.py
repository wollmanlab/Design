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
            'n_cpu': 6,  # Number of CPU threads to use for PyTorch
            'n_bit': 24,  # Number of bits in the encoding (dimensionality of the projection)
            'n_iterations': 25000,  # Total number of training iterations
            'batch_size': 2500,  # Batch size for training (0 = use full dataset)
            'target_brightness_log': 4.5,  # Target brightness in log10 scale
            'total_n_probes': 30e4,  # Target total number of probes across all genes
            'probe_weight': 1.0,  # Weight for probe count loss term
            'probe_under_weight_factor': 0.1,  # Factor for under-utilization penalty
            'gene_constraint_weight': 1.0,  # Weight for gene constraint violation penalty
            'target_brightness_weight':1.0,  # Weight for target brightness loss term
            'gradient_clip_max_norm': 1.0,  # Maximum gradient norm for clipping
            'learning_rate_start': 0.05,  # Initial learning rate
            'learning_rate_end': 0.05,  # Final learning rate (linear interpolation)
            'report_freq': 100,  # How often to report training progress
            'sparsity_target': 0.95,  # Target sparsity ratio (fraction of zeros)
            'sparsity_weight': 0.0,  # Weight for sparsity loss term
            'categorical_weight': 1.0,  # Weight for categorical classification loss
            'label_smoothing': 0.1,  # Label smoothing factor for cross-entropy loss
            'best_model': 1,  # Whether to save the best model during training
            'device': 'cpu',  # Device to run computations on ('cpu' or 'cuda')
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results',  # Output directory path
            'input': './',  # Input directory path
            'Verbose': 1,  # Verbosity level (0 = quiet, 1 = verbose)
            'decoder_hidden_layers': 0,  # Number of hidden layers in decoder
            'decoder_hidden_dim': 128,  # Hidden dimension size in decoder
            'constraints': 'constraints.csv',  # Path to gene constraints file
            'X_test': 'X_test.pt',  # Path to test features tensor
            'y_test': 'y_test.pt',  # Path to test labels tensor
            'X_train': 'X_train.pt',  # Path to training features tensor
            'y_train': 'y_train.pt',  # Path to training labels tensor
            'y_label_converter_path': 'categorical_converter.csv',  # Path to label mapping file
            # Gene-level noise parameters
            'gene_dropout_proportion_start': 0.0,  # Initial proportion of genes to drop out
            'gene_dropout_proportion_end': 0.0,  # Final proportion of genes to drop out
            'gene_fold_noise_start': 0.0,  # Initial gene expression fold noise level
            'gene_fold_noise_end': 0.5,  # Final gene expression fold noise level
            # Weight-level noise parameters
            'weight_dropout_proportion_start': 0.0,  # Initial proportion of encoding weights to drop out
            'weight_dropout_proportion_end': 0.1,  # Final proportion of encoding weights to drop out
            'weight_fold_noise_start': 0.0,  # Initial encoding weight fold noise level
            'weight_fold_noise_end': 0.1,  # Final encoding weight fold noise level
            # Projection-level noise parameters
            'projection_dropout_proportion_start': 0.0,  # Initial proportion of projection values to drop out
            'projection_dropout_proportion_end': 0.0,  # Final proportion of projection values to drop out
            'projection_fold_noise_start': 0.0,  # Initial projection fold noise level
            'projection_fold_noise_end': 0.0,  # Final projection fold noise level
            # Decoder-level noise parameters
            'decoder_dropout_rate_start': 0.0,  # Initial decoder dropout rate
            'decoder_dropout_rate_end': 0.0,  # Final decoder dropout rate
            # Constant noise parameters
            'constant_noise_start': 0.0,  # Initial constant noise level (log10 scale)
            'constant_noise_end': 2.0,  # Final constant noise level (log10 scale)
            # Weight perturbation parameters
            'perturbation_frequency': 250,  # How often to perturb weights (every N iterations)
            'perturbation_percentage': 0.01,  # Percentage of weights to perturb (0.0-1.0)
            'min_probe_fraction': 0.01,  # Minimum probe fraction for initialization
            'max_probe_fraction': 0.25,  # Maximum probe fraction for initialization
            'min_probe_fraction_perturb': 0.05,  # Minimum probe fraction for perturbation
            'max_probe_fraction_perturb': 0.5,  # Maximum probe fraction for perturbation
            # Activation and normalization parameters
            'activation_function':'tanh',  # Activation function for encoding weights
            'decoder_activation': 'gelu',  # Activation function for decoder hidden layers ('relu', 'leaky_relu', 'gelu', 'swish', 'tanh')
            'sum_normalize_projection': 1,  # Whether to normalize projection by sum
            'bit_normalize_projection': 1,  # Whether to normalize projection by bit-wise statistics
        }

user_parameters['input'] = input_dir
# Define parameter variants - parameters to vary and their possible values
parameter_variants = {
    'best_model':[0,1],
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
    pd.DataFrame(current_params.values(), index=pd.Index(current_params.keys()), columns=pd.Index(['values'])).to_csv(fullfilepath)

print(f"Generated {len(combinations)} parameter files in {input_param_path}")