# %%
""" Set Parameters """
import pandas as pd
import os
import shutil
user_parameters = {
            'device': 'cpu',
            'Verbose': 1,
            'n_cpu': 1,
            'n_bit': 25,
            'n_iterations': 10000,
            'total_n_probes': 30e4,
            'probe_weight': 1,
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
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results',
            'input':'/u/project/rwollman/data/Allen_WMB_2024Mar06/Training_data/'
        }

if os.path.exists(user_parameters['output']):
    shutil.rmtree(user_parameters['output'])


input_param_path = '/u/home/r/rwollman/project-rwollman/atlas_design/opt_design_run_1/params_files_to_scan'
param_file_name = 'user_parameters_single_cpu_no_region_embedding'
fullfilepath = os.path.join(input_param_path, f"{param_file_name}.csv")
pd.DataFrame(user_parameters.values(),index=user_parameters.keys(),columns=['values']).to_csv(fullfilepath)


