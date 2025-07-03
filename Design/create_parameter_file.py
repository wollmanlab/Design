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
import re

def find_next_run_number(base_dir):
    """Find the next available run number by looking at existing Run# directories."""
    if not os.path.exists(base_dir):
        return "Run0"
    
    # Get all directories in base_dir
    try:
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except (OSError, PermissionError):
        print(f"Warning: Could not read directory {base_dir}. Using Run0.")
        return "Run0"
    
    # Find directories that match the pattern "Run" followed by a number
    run_dirs = []
    for dir_name in existing_dirs:
        match = re.match(r'^Run(\d+)$', dir_name)
        if match:
            run_dirs.append(int(match.group(1)))
    
    if not run_dirs:
        return "Run0"
    
    # Find the highest run number and add 1
    next_run_num = max(run_dirs) + 1
    return f"Run{next_run_num}"

if len(sys.argv) > 1:
    run_dir = sys.argv[1]
    base_dir = os.path.join(base_dir, run_dir)
else:
    # Automatically infer the next run number
    run_dir = find_next_run_number(base_dir)
    base_dir = os.path.join(base_dir, run_dir)
    print(f"No run directory specified. Automatically using: {run_dir}")

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)


input_param_path = os.path.join(base_dir,'params_files_to_scan')
os.makedirs(input_param_path, exist_ok=True)
os.makedirs(os.path.join(base_dir, 'params_files_scanned'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'design_results'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'job_logs'), exist_ok=True)

user_parameters = {
            'n_cpu': 3,  # Number of CPU threads to use for PyTorch
            'n_bit': 24,  # Number of bits in the encoding (dimensionality of the projection)
            'n_iters': 250000,  # Total number of training iterations
            'batch_size': 500,  # Batch size for training (0 = use full dataset)
            'brightness': 4.5,  # Target brightness in log10 scale
            'n_probes': 30e4,  # Target total number of probes across all genes
            'probe_wt': 1,  # Weight for probe count loss term
            'gene_constraint_wt': 1,  # Weight for gene constraint violation penalty
            'brightness_wt':1,  # Weight for target brightness loss term
            'dynamic_wt': 0.0,  # Weight for dynamic range loss terms
            'dynamic_fold_s': 0.0,  # Initial target fold change for dynamic range
            'dynamic_fold_e': 4.0,  # Final target fold change for dynamic range
            'separation_wt': 1,  # Weight for cell type separation loss term
            'separation_fold_s': 0.0,  # Initial minimum fold change required between cell type pairs
            'separation_fold_e': 1.0,  # Final minimum fold change required between cell type pairs
            'gradient_clip': 1,  # Maximum gradient norm for clipping
            'lr_s': 0.01,  # Initial learning rate
            'lr_e': 0.01,  # Final learning rate (linear interpolation)
            'report_rt': 100,  # How often to report training progress
            'sparsity_s': 0.95,  # Initial target sparsity ratio (fraction of zeros)
            'sparsity_e': 0.95,  # Final target sparsity ratio (fraction of zeros)
            'sparsity_wt': 0,  # Weight for sparsity loss term
            'categorical_wt': 5,  # Weight for categorical classification loss
            'label_smoothing': 0.1,  # Label smoothing factor for cross-entropy loss
            'best_model': 0,  # Whether to save the best model during training
            'device': 'cpu',  # Device to run computations on ('cpu' or 'cuda')
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results',  # Output directory path
            'input': './',  # Input directory path
            'decoder_n_lyr': 0,  # Number of hidden layers in decoder
            'decoder_h_dim': 64,  # Hidden dimension size in decoder
            'top_n_genes': 0,  # Number of top genes to keep (0 = keep all genes)
            'constraints': 'constraints.csv',  # Path to gene constraints file
            'X_test': 'X_test.pt',  # Path to test features tensor
            'y_test': 'y_test.pt',  # Path to test labels tensor
            'X_train': 'X_train.pt',  # Path to training features tensor
            'y_train': 'y_train.pt',  # Path to training labels tensor
            'y_label_converter_path': 'categorical_converter.csv',  # Path to label mapping file
            'P_scaling': 5,  # Scaling factor for sum normalization (defaults to n_bit)
            # Gene-level noise parameters
            'X_drp_s': 0.05,  # Initial proportion of genes to drop out (randomly set to 0)
            'X_drp_e': 0.05,  # Final proportion of genes to drop out (randomly set to 0)
            'X_noise_s': 0,  # Initial gene expression noise level 0.5 -> 50% decrease to 200% increase (0-1)
            'X_noise_e': 0,  # Final gene expression noise level 0.5 -> 50% decrease to 200% increase (0-1)
            # Weight-level noise parameters
            'E_drp_s': 0.05,  # Initial proportion of encoding weights to drop out (randomly set to 0)
            'E_drp_e': 0.05,  # Final proportion of encoding weights to drop out (randomly set to 0)
            'E_noise_s': 0,  # Initial encoding weight noise level (percentage decrease with minimum bound 0-1)
            'E_noise_e': 0,  # Final encoding weight noise level (percentage decrease with minimum bound 0-1)
            # Projection-level noise parameters
            'P_drp_s': 0,  # Initial proportion of projection values to drop out (randomly set to 0)
            'P_drp_e': 0,  # Final proportion of projection values to drop out (randomly set to 0)
            'P_noise_s': 0,  # Initial projection measurement noise level (percentage accuracy error 0-1)
            'P_noise_e': 0,  # Final projection measurement noise level (percentage accuracy error 0-1)
            # Decoder-level noise parameters
            'D_drp_s': 0,  # Initial decoder dropout rate
            'D_drp_e': 0,  # Final decoder dropout rate
            # Constant noise parameters
            'P_add_s': 2.0,  # Initial constant noise level (log10 scale, added to projections)
            'P_add_e': 2.0,  # Final constant noise level (log10 scale, added to projections)
            # Weight perturbation parameters
            'E_perturb_rt': 0,  # How often to perturb weights (every N iterations)
            'E_perb_prct': 0.01,  # Percentage of weights to perturb (0.0-1.0)
            'E_init_min': 0.001,  # Minimum probe fraction for initialization
            'E_init_max': 0.01,  # Maximum probe fraction for initialization
            'E_perturb_min': 0.05,  # Minimum probe fraction for perturbation
            'E_perturb_max': 0.25,  # Maximum probe fraction for perturbation
            # Activation and normalization parameters
            'encoder_act':'tanh',  # Activation function for encoding weights
            'decoder_act': 'tanh',  # Activation function for decoder hidden layers ('relu', 'leaky_relu', 'gelu', 'swish', 'tanh')
            'sum_norm': 1,  # Whether to normalize projection by sum
            'bit_norm': 0,  # Whether to normalize projection by bit-wise statistics
            'continue_training': 0,  # Whether to continue training if model is loaded from file (0 = skip training, 1 = continue training)
            'use_noise': 1,  # Whether to apply noise/dropout during training (0 = no noise, 1 = use noise) 
        }

user_parameters['input'] = input_dir

# # Figure 2a Baseline  with vs without noise Run 2 (Run0 with no probe weights)
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],#[0,1,2,3]
#     'categorical_wt':[1],#[0.1,0.5,1,2.5,5,10]
#     'separation_wt':[0],#[0.1,0.5,1,2.5,5,10]
#     'n_bit':[24], #[12,24,48,96]
#     'dynamic_wt':[0], #[0.1,0.5,1,2.5,5,10]
#     'use_noise':[0,1]
#     },
# ]

# # Figure 2b Showing Tradeoffs Run 1
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],#[0,1,2,3]
#     'categorical_wt':[0.1,0.5,1,2.5,5,10],#[0.1,0.5,1,2.5,5,10]
#     'separation_wt':[0.0,0.1,0.5,1,2.5,5,10],#[0.1,0.5,1,2.5,5,10]
#     'brightness':[4.5],
#     'n_bit':[24], #[12,24,48,96]
#     'use_noise':[1]
#     },
# ]

# # Figure 2a Brightness Run 3
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],#[0,1,2,3]
#     'categorical_wt':[5],#[0.1,0.5,1,2.5,5,10]
#     'separation_wt':[1],#[0.1,0.5,1,2.5,5,10]
#     'brightness':[2.5,3,3.5,4,4.5,5,5.5,6],
#     'n_bit':[24], #[12,24,48,96]
#     'use_noise':[1]
#     },
# ]

# # N_bits with no experimental constraints Run 4
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],
#     'categorical_wt':[1],
#     'separation_wt':[0],
#     'brightness_wt':[0],
#     'gene_constraint_wt':[0],
#     'probe_wt':[0],
#     'dynamic_wt':[0],
#     'n_bit':[12,24,48,96], #[12,24,48,96]
#     'use_noise':[0,1]
#     },
# ]

# # N_bits with experimental constraints Run 5
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],
#     'categorical_wt':[1],
#     'separation_wt':[0],
#     'brightness_wt':[1],
#     'gene_constraint_wt':[1],
#     'probe_wt':[1],
#     'dynamic_wt':[0],
#     'n_bit':[12,24,48,96], #[12,24,48,96]
#     'use_noise':[1],
#     },
# ]

# # Separability screen  Run 6
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],
#     'categorical_wt':[1],
#     'separation_wt':[0.0,0.05,0.1,0.5,1,1.5,2,5],
#     'brightness_wt':[1],
#     'gene_constraint_wt':[1],
#     'probe_wt':[1],
#     'dynamic_wt':[0],
#     'n_bit':[24], # set 
#     'use_noise':[1],
#     },
# ]

# # Brightness screen  Run 7
# parameter_variant_list = [
#     {
#     'decoder_n_lyr':[0,1,2,3],
#     'categorical_wt':[1],
#     'separation_wt':[0.1], # set
#     'brightness_wt':[1],
#     'brightness':[2.5,3,3.5,4,4.5,5,5.5,6],
#     'n_probes':[100e3,200e3,300e3,400e3,500e3],
#     'gene_constraint_wt':[1],
#     'probe_wt':[1],
#     'dynamic_wt':[0],
#     'n_bit':[24], # set
#     'use_noise':[1],
#     },
# ]
parameter_variant_list = []
# Fig 2a Benchmark Encoding Capacity
parameter_variant_list.append(
    {
    'decoder_n_lyr':[0,1],
    'categorical_wt':[2.5],
    'separation_wt':[0.1],
    'brightness_wt':[1],
    'brightness':[4.5],
    'n_probes':[300e3],
    'gene_constraint_wt':[0],
    'probe_wt':[0],
    'dynamic_wt':[0],
    'n_bit':[6,12,24,48,96],
    'use_noise':[1],
    'fig':['fig2a']
    })

# Fig 2b In Situ Encoding Capacity
parameter_variant_list.append(
    {
    'decoder_n_lyr':[0,1],
    'categorical_wt':[2.5],
    'separation_wt':[0.1],
    'brightness_wt':[1],
    'brightness':[4.5],
    'n_probes':[300e3],
    'gene_constraint_wt':[1],
    'probe_wt':[0],
    'dynamic_wt':[0],
    'n_bit':[6,12,24,48,96],
    'use_noise':[1],
    'fig':['fig2b']
    })

# Fig 2c Separability Tradeoff
parameter_variant_list.append(
    {
    'decoder_n_lyr':[0,1],
    'categorical_wt':[2.5],
    'separation_wt':[0.01,0.05,0.1,0.5,1,2],
    'brightness_wt':[1],
    'brightness':[4.5],
    'n_probes':[300e3],
    'gene_constraint_wt':[1],
    'probe_wt':[0],
    'dynamic_wt':[0],
    'n_bit':[24],
    'use_noise':[1],
    'fig':['fig2c']
    })

# Fig 2d Brightness Tradeoff
parameter_variant_list.append(
    {
    'decoder_n_lyr':[0,1],
    'categorical_wt':[2.5],
    'separation_wt':[0.1],
    'brightness_wt':[1],
    'brightness':[3,3.5,4,4.5,5,5.5],
    'n_probes':[300e3],
    'gene_constraint_wt':[1],
    'probe_wt':[0],
    'dynamic_wt':[0],
    'n_bit':[24],
    'use_noise':[1],
    'fig':['fig2d']
    })

# Fig 3a Number of Encoding Probes Tradeoff
parameter_variant_list.append(
    {
    'decoder_n_lyr':[0,1],
    'categorical_wt':[2.5],
    'separation_wt':[0.1],
    'brightness_wt':[1],
    'brightness':[4.5],
    'n_probes':[250e3],
    'gene_constraint_wt':[1],
    'probe_wt':[1],
    'dynamic_wt':[0],
    'n_bit':[6,12,24,48,96],
    'use_noise':[1],
    'fig':['fig3a']
    })

# Fig 3b Number of Imaging Rounds Tradeoff
parameter_variant_list.append(
    {
    'decoder_n_lyr':[0,1],
    'categorical_wt':[2.5],
    'separation_wt':[0.1],
    'brightness_wt':[1],
    'brightness':[4.5],
    'n_probes':[10e3,50e3,100e3,250e3,500e3,1000e3],
    'gene_constraint_wt':[1],
    'probe_wt':[1],
    'dynamic_wt':[0],
    'n_bit':[24],
    'use_noise':[1],
    'fig':['fig3b']
    })

# add an option to have _s and _e be the same value
same_se = True

total_combinations = []
for parameter_variants in parameter_variant_list:
    # Generate all parameter combinations
    param_names = list(parameter_variants.keys())
    param_values = list(parameter_variants.values())
    for param_name in param_names:
        if not param_name in user_parameters.keys():
            if not param_name+'_s' in user_parameters.keys():
                raise ValueError(f"Parameter {param_name} not found in user_parameters")
    # Generate all combinations of parameter values
    combinations = list(itertools.product(*param_values))
    total_combinations.extend(combinations)
    for i, combination in enumerate(combinations):
        # Create a copy of default parameters
        current_params = user_parameters.copy()
        
        # Update with current combination
        param_desc_list = [] # Use a list to build description parts
        for j, param_name in enumerate(param_names):
            value_str = str(combination[j]).replace('.', 'p')
            if (same_se) & (param_name+'_s' in current_params.keys()):
                current_params[param_name+'_s'] = combination[j]
                current_params[param_name+'_e'] = combination[j]
                param_desc_list.append(f"{param_name}_{value_str}_se")
            else:
                current_params[param_name] = combination[j]
                param_desc_list.append(f"{param_name}_{value_str}")

        
        # Check if learning_rate_e is greater than learning_rate_s
        if current_params['lr_e'] > current_params['lr_s']:
            # If end is greater than start, make end equal to start
            current_params['lr_e'] = current_params['lr_s']
            # Add this adjustment to the parameter description
            param_desc_list.append(f"lr_e_adjusted")

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
print(f"Generated {len(total_combinations)} parameter files in {input_param_path}")

# Call the shell script to submit jobs
print(f"\nSubmitting jobs for run directory: {run_dir}")
import subprocess
import os
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
shell_script_path = os.path.join(script_dir, "sub_multi_param_file_optimization.sh")
# Make sure the shell script is executable
os.chmod(shell_script_path, 0o755)
# Call the shell script with the run directory as argument
try:
    result = subprocess.run([shell_script_path, run_dir], 
                          capture_output=True, 
                          text=True, 
                          check=True)
    print("Shell script executed successfully:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error executing shell script: {e}")
    print(f"Error output: {e.stderr}")
    sys.exit(1)

# Show current job status
print("\nCurrent job status:")
try:
    qstat_result = subprocess.run(['qstat', '-u', '$USER'], 
                                capture_output=True, 
                                text=True, 
                                check=True)
    print(qstat_result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error checking job status: {e}")
    print(f"Error output: {e.stderr}")
except FileNotFoundError:
    print("qstat command not found - may not be on a PBS/SGE system")