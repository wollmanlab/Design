#!/usr/bin/env python
import pandas as pd
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import logging
import time
import datetime
import random
import math
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
import argparse
try:
    from IPython import get_ipython
    from IPython.display import Image, display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

# --- Helper function to sanitize filenames ---
def sanitize_filename(name):
    """Removes or replaces characters invalid for filenames."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace spaces and slashes with underscores
    name = re.sub(r'[\s/\\:]+', '_', name)
    # Remove characters that are generally problematic in filenames
    name = re.sub(r'[<>:"|?*]+', '', name)
    # Limit length if necessary (optional)
    # max_len = 100
    # name = name[:max_len]
    return name

# --- Standalone Visualization Function (Based on Prototype) ---
# NOTE: This function definition is from the original code provided by the user.
# It expects y_labels as strings, not indices + map.
# If errors occur here later, it might need adjustment based on how y_labels are handled elsewhere.
def plot_projection_space_density(P,y_labels,plot_path,sum_norm=True,log=True):
    logger = logging.getLogger("ProjectionPlotDensity") # Use a specific logger
    logger.info(f"Generating projection space density plot: {plot_path}")
    # Normalize P (add epsilon for numerical stability if sums can be zero)
    if sum_norm:
        P = P * (np.clip(P.sum(1),1,None).mean() / (np.clip(P.sum(1),1,None)[:, None]))
    labels = np.array([f"Bit {str(bit)}" for bit in range(P.shape[1])])
    unique_cell_types = np.unique(y_labels)
    num_measurements = labels.shape[0]
    num_plot_pairs = math.ceil(num_measurements / 2)
    total_rows = num_plot_pairs
    total_cols = 2 # Strict 2 columns

    # Handle case with zero measurements -> zero rows/plots
    if num_plot_pairs == 0:
        print("No measurement pairs to plot.")
        # Optionally create an empty plot or just return
        fig, ax = plt.subplots(1,1, figsize=(6,1))
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center')
        ax.axis('off')
        try:
            plt.savefig(plot_path, dpi=100)
        except Exception as e:
            logger.error(f"Failed to save empty plot {plot_path}: {e}")
        finally:
            plt.close(fig)
        return


    fig, axes = plt.subplots(total_rows, total_cols,
                                 figsize=(12, 5 * total_rows), # e.g., 14 inches wide, 7 per row height
                                 squeeze=False) # Ensure axes is always 2D

    # --- 4. Initialize Variables for Plot 2 Coloring --- (Keep Original)
    color_mapper = {}
    used_colors_list = [np.array([0., 0., 0.]), np.array([1., 1., 1.])]

    # --- 5. Loop Through Measurement Pairs and Plot ---
    plot_pair_idx = 0 # Index for the *pair* of plots, corresponds to ROW index
    for i_pair_start in range(num_measurements):
        # Process pairs of measurements (Measurement i vs Measurement i+1)
        if i_pair_start % 2 == 0 and plot_pair_idx < num_plot_pairs: # Check ensures we don't exceed rows
            feature_idx1 = i_pair_start
            # Handle the case of an odd number of measurements (Original Logic)
            if feature_idx1 + 1 >= num_measurements:
                feature_idx2 = feature_idx1 - 1 # Pair last with second-to-last
                if feature_idx2 < 0: continue # Skip if only one measurement
            else:
                feature_idx2 = feature_idx1 + 1

            feature_name1 = labels[feature_idx1]
            feature_name2 = labels[feature_idx2]

            # --- 5a. Get Data for the Pair --- (Keep Original)
            x = np.array(P[:, feature_idx1]).ravel()
            y = np.array(P[:, feature_idx2]).ravel()

            # --- 5b. Common Preprocessing (Clipping, Log Scale, Bins) --- (Keep Original)
            # Clip based on percentiles of *positive* values to avoid issues with zeros
            x_pos = x[x > 0]
            y_pos = y[y > 0]

            # Original percentile logic with checks for few points
            if len(x_pos) > 1:
                vmin_x, vmax_x = np.percentile(x_pos, [0.1, 99.9])
            elif len(x_pos) == 1: vmin_x, vmax_x = x_pos[0], x_pos[0] # Handle single positive point
            else: vmin_x, vmax_x = 0, 0 # Fallback if no positive points

            if len(y_pos) > 1:
                vmin_y, vmax_y = np.percentile(y_pos, [0.1, 99.9])
            elif len(y_pos) == 1: vmin_y, vmax_y = y_pos[0], y_pos[0] # Handle single positive point
            else: vmin_y, vmax_y = 0, 0 # Fallback

            # Ensure max >= min (original didn't explicitly check, but good practice)
            vmax_x = max(vmax_x, vmin_x)
            vmax_y = max(vmax_y, vmin_y)

            x = np.clip(x, vmin_x, vmax_x)
            if log:
                x = np.log10(x + 1) # Add 1 before log
            # Define bins robustly even if data range is small (Original Logic)
            x_min, x_max = x.min(), x.max()
            x_bins = np.linspace(x_min, x_max if x_max > x_min else x_max + 1, 100) # Original +1 if min=max

            y = np.clip(y, vmin_y, vmax_y)
            if log:
                y = np.log10(y + 1) # Add 1 before log
            y_min, y_max = y.min(), y.max()
            y_bins = np.linspace(y_min, y_max if y_max > y_min else y_max + 1, 100) # Original +1 if min=max

            # --- 5c. Determine Axis Indices (Modified for 2 Columns) ---
            current_row_idx = plot_pair_idx
            # ax1 uses column 0, ax2 uses column 1

            # --- 5d. Plot 1 Logic (Density - Column 0) --- (Keep Original Internals)
            ax1 = axes[current_row_idx, 0] # Assign to first column

            # Calculate 2D histogram (Original Logic)
            img, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
            img = np.log10(img + 1) # Log transform counts

            # Determine color limits for the histogram image (Original Logic)
            img_pos = img[img > 0]
            if len(img_pos) > 0:
                vmin_img, vmax_img = np.percentile(img_pos, [0.1, 99]) # Original percentiles
                if vmin_img == vmax_img: vmax_img += 1e-6 # Avoid identical limits
            else: vmin_img, vmax_img = 0, 1 # Default if empty

            # Use original imshow parameters (cmap='bwr')
            # Use extent to map bins to axes correctly
            im1 = ax1.imshow(img.T, vmin=vmin_img, vmax=vmax_img, cmap='bwr', origin='lower', aspect='auto', interpolation='nearest',
                           extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])

            # Set ticks and labels for Plot 1 (Original Logic, but use extent)
            num_ticks = 5
            # Use linspace based on extent for tick positions
            x_tick_labels_val = np.linspace(x_bins[0], x_bins[-1], num=num_ticks)
            y_tick_labels_val = np.linspace(y_bins[0], y_bins[-1], num=num_ticks)
            ax1.set_xticks(x_tick_labels_val)
            ax1.set_yticks(y_tick_labels_val)

            # Use original rounding for labels
            ax1.set_xticklabels(np.round(x_tick_labels_val, 1))
            ax1.set_yticklabels(np.round(y_tick_labels_val, 1))

            if log:
                ax1.set_xlabel(f"Bit {feature_name1} (log10)")
                ax1.set_ylabel(f"Bit {feature_name2} (log10)")
            else:
                ax1.set_xlabel(f"Bit {feature_name1}")
                ax1.set_ylabel(f"Bit {feature_name2}")
            ax1.grid(False)
            # fig.colorbar(im1, ax=ax1) # Add colorbar

            # --- 5e. Plot 2 Logic (Cell Types - Column 1) --- (Keep Original Internals)
            ax2 = axes[current_row_idx, 1] # Assign to second column

            # Initialize composite image for this subplot (Original Logic)
            composite_img = np.zeros((len(y_bins)-1, len(x_bins)-1, 3))
            legend_handles = [] # Initialize legend handles for this plot

            for ct in unique_cell_types:
                mask = y_labels == ct
                if np.sum(mask) < 2:
                    continue

                img_ct, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_bins, y_bins])
                img_ct = np.log10(img_ct + 1)

                # Normalize based on percentiles *of positive values within the cell type* (Original Logic)
                img_ct_pos = img_ct[img_ct > 0]
                if len(img_ct_pos) >= 2: # Need points for percentile
                    vmin_ct, vmax_ct = np.percentile(img_ct_pos, [0.1, 99]) # Original percentiles
                    vmin_ct = 0 # Force min to 0 (Original Logic)
                    if vmax_ct <= vmin_ct: vmax_ct = vmin_ct + 1e-6 # Avoid division issues
                    # Check if vmax_ct is zero before division
                    if vmax_ct > 1e-9:
                        img_ct_norm = (img_ct - vmin_ct) / vmax_ct
                    else:
                        img_ct_norm = np.zeros_like(img_ct)
                elif len(img_ct_pos) == 1: # Handle single non-zero point case
                    img_ct_norm = (img_ct > 0).astype(float) # Set the single point bin to 1
                else: # Handle all zero case
                    img_ct_norm = np.zeros_like(img_ct) # All zero image

                img_ct_norm = np.clip(img_ct_norm, 0, 1).T # Transpose here (Original Logic)

                # Get or generate color for this cell type (Original Logic)
                if ct not in color_mapper:
                    attempts = 0
                    max_attempts = 200
                    min_dist_sq = 0.1 # Original value
                    min_sum = 0.5 # Original value

                    while attempts < max_attempts:
                        color = np.random.rand(3)
                        # Optional: Zero out one channel like original code (Keep commented out if preferred)
                        # color[np.random.randint(0, 3)] = 0
                        color_sum = np.sum(color)
                        distances_sq = [np.sum((color - existing_color)**2) for existing_color in used_colors_list]
                        min_d2 = min(distances_sq) if distances_sq else 1.0

                        if min_d2 > min_dist_sq and color_sum > min_sum:
                            color_mapper[ct] = color
                            used_colors_list.append(color)
                            break
                        attempts += 1
                    if ct not in color_mapper: # Fallback (Original Logic)
                        color_mapper[ct] = np.random.rand(3) * 0.8
                        used_colors_list.append(color_mapper[ct])

                # Create colored layer and add to composite image (Original Logic)
                ct_layer = np.dstack([img_ct_norm] * 3) * color_mapper[ct]
                composite_img += ct_layer

            # Normalize and clip the final composite image (Original Logic)
            if composite_img.max() > 0:
                # vmax_composite = np.percentile(composite_img[composite_img > 1e-9], 99.9) # Original percentile line was commented out
                vmax_composite = 1.0 # Use original fixed normalization factor
                composite_img = composite_img / max(vmax_composite, 1e-9) # Avoid division by zero

            composite_img = np.clip(composite_img, 0, 1)

            ax2.imshow(composite_img, origin='lower', aspect='auto', interpolation='nearest',
                       extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]) # Use extent

            # Set ticks and labels for Plot 2 (reuse from Plot 1, Original Logic)
            ax2.set_xticks(x_tick_labels_val)
            ax2.set_xticklabels(np.round(x_tick_labels_val, 1)) # Original rounding
            ax2.set_yticks(y_tick_labels_val)
            ax2.set_yticklabels(np.round(y_tick_labels_val, 1)) # Original rounding
            if log:
                ax2.set_xlabel(f"Bit {feature_name1} (log10)")
                ax2.set_ylabel(f"Bit {feature_name2} (log10)")
            else:
                ax2.set_xlabel(f"Bit {feature_name1}")
                ax2.set_ylabel(f"Bit {feature_name2}")
            ax2.grid(False)
            # Add legend for cell types
            handles = [Patch(color=color_mapper[ct], label=ct) for ct in unique_cell_types if ct in color_mapper]
            # ax2.legend(handles=handles, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')


            plot_pair_idx += 1 # Increment the row index

    # plt.tight_layout(rect=[0, 0, 0.9, 1]) # Original rect

    try:
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Saved projection space density plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")
    finally:
        plt.close(fig)

class EncodingDesigner(nn.Module):
    def __init__(self, user_parameters_path=None):
        """
        Initializes the EncodingDesigner model.

        Args:
            user_parameters_path (str, optional): Path to a CSV file containing user parameters.
                                                 The CSV should have parameter names as the index
                                                 and a single column named 'values'.
                                                 Defaults to None, in which case default parameters are used.
        """
        super().__init__() # Call super constructor first
        self.loss_component_names = [
            'probe_weight',
            'categorical',
            'gene_constraint',
            'p_std', # Corresponds to pnorm_std_weight, aims to maximize min CV
            'type_correlation_max', # Assuming this is the primary correlation loss you want to balance
            # 'type_correlation_mean', # Add if you want to balance this separately
            'hierarchical_scatter', # This component's weight is handled separately if active
            'intra_type_variance', # NEW: For intra-type projection variance
            'bit_iqr_variance', # NEW: For bit IQR variance
            'type_entropy', # NEW: For type projection entropy
        ]

        # --- Defaults ---
        # These are the base defaults. They will be potentially overwritten by the loaded file.
        self.user_parameters = {
            'device': 'cpu',
            'Verbose': 1,
            'n_cpu': 30,
            'n_bit': 25,
            'n_iterations': 5000,
            'total_n_probes': 30e4,
            'probe_weight': 1,
            'probe_under_weight_factor': 0.05, # Penalty factor for when total_n_probes is below target but we still want to push it down
            'weight_dropout_proportion': 0.1,
            'projection_dropout_proportion': 0.1,
            'gene_constraint_weight': 1,
            'target_brightness_log': 4.5,
            'learning_rate': 0.05, # Now acts as default start LR if specific ones aren't set
            'learning_rate_start': 0.05, # Starting learning rate
            'learning_rate_end': 0.005, # Ending learning rate
            'report_freq': 250,
            'type_correlation_mean_weight': 0,
            'type_correlation_max_weight': 1,
            'noise_level': 3,
            'categorical_weight': 1,
            'batch_size': 1000,
            'pnorm_std_weight': 0.1, # Weight for Pnormalized STD loss
            'use_region_info': 1, # Flag to use region-specific decoders (1=yes, 0=no)
            'region_embedding_dim': 4, # Dimension for region embeddings
            'correlation_thresh': 0.75,
            'output': './',
            'input': './', # Added input directory parameter
            'constraints': 'constraints.csv', # Default filename
            'X_test': 'X_test.pt',           # Default filename
            'y_test': 'y_test.pt',           # Default filename
            'r_test': 'r_test.pt',           # Default filename
            'X_train': 'X_train.pt',         # Default filename
            'y_train': 'y_train.pt',         # Default filename
            'r_train': 'r_train.pt',         # Default filename
            'y_label_converter_path': 'categorical_converter.csv', # Default filename
            'r_label_converter_path': 'region_categorical_converter.csv', # Default filename
            'hierarchical_scatter_weight': 0.0,  # Weight for the new hierarchical scatter loss
            'y_hierarchy_file_path': 'child_parent_relationships.csv',      # Path to the file defining cell type hierarchy (e.g., 'cell_type_hierarchy.csv')
            'intra_type_variance_weight': 0.0, # NEW: Weight for intra-type projection variance
            'bit_iqr_variance_weight': 0.0, # NEW: Weight for bit IQR variance
            'type_entropy_weight': 0.0, # NEW: Weight for type projection entropy
            'tanh_slope_factor': 1.0, # NEW: Slope factor for tanh activation
        }

        # --- Logging Setup (Must happen before parameter loading/conversion uses the logger) ---
        # Determine output dir early for logging setup. Use default first, may be updated by loaded params.
        temp_output_dir = self.user_parameters['output']
        # If a path was provided, try to get the output dir from there for logging setup
        loaded_params_temp = {}
        if user_parameters_path is not None:
            try:
                # Load temporarily just to check for 'output' dir for logging
                df_temp = pd.read_csv(user_parameters_path, index_col=0, low_memory=False)
                if 'values' not in df_temp.columns:
                    # Log using a basic config temporarily if logger not fully set up
                    logging.basicConfig(level=logging.ERROR)
                    logging.error(f"Parameter file {user_parameters_path} missing 'values' column.")
                    # Continue without loaded params, defaults will be used.
                else:
                    loaded_params_temp = dict(zip(df_temp.index, df_temp['values']))
                    temp_output_dir = loaded_params_temp.get('output', temp_output_dir) # Update if present
            except FileNotFoundError:
                logging.basicConfig(level=logging.ERROR)
                logging.error(f"Parameter file not found at: {user_parameters_path}. Using default parameters.")
            except Exception as e:
                logging.basicConfig(level=logging.ERROR)
                logging.error(f"Error loading parameter file {user_parameters_path}: {e}. Using default parameters.")


        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO) # Reset basic config
        
        # Extract filename from user_parameters_path and remove .csv extension
        input_filename = os.path.basename(user_parameters_path) if user_parameters_path else "default"
        input_filename = os.path.splitext(input_filename)[0]  # Remove .csv extension
        self.log_file = os.path.join(temp_output_dir, f'log_{input_filename}.log')
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        logging.basicConfig(
            filename=self.log_file, filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%Y %B %d %H:%M:%S', level=logging.INFO, force=True)
        self.log = logging.getLogger("Designer")
        self.results = {}

        # --- Best Model Tracking ---
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1

        # --- Load parameters from file if path is provided ---
        loaded_user_parameters = {}
        if user_parameters_path is not None:
            self.log.info(f"Loading user parameters from: {user_parameters_path}")
            try:
                user_parameters_df = pd.read_csv(user_parameters_path, index_col=0, low_memory=False)
                if 'values' not in user_parameters_df.columns:
                    self.log.error(f"Parameter file {user_parameters_path} missing 'values' column. Sticking to defaults.")
                else:
                    # Convert DataFrame to dictionary
                    loaded_user_parameters = dict(zip(user_parameters_df.index, user_parameters_df['values']))
                    self.log.info(f"Successfully loaded {len(loaded_user_parameters)} parameters from file.")

                    # Attempt to convert values to float
                    for key, val in loaded_user_parameters.items():
                        try:
                            # Convert to float first, as int() might fail on "3.0"
                            float_val = float(val)
                            # Check if it's actually an integer
                            if float_val.is_integer():
                                loaded_user_parameters[key] = int(float_val)
                            else:
                                loaded_user_parameters[key] = float_val
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails (e.g., paths, 'cpu', 'True')
                            self.log.debug(f"Could not convert parameter '{key}' value '{val}' to float. Keeping as string.")
                            continue # Keep original string value
                    self.log.info("Attempted conversion of loaded parameters to numeric types.")

            except FileNotFoundError:
                self.log.error(f"Parameter file not found at: {user_parameters_path}. Using default parameters.")
                loaded_user_parameters = {} # Reset to ensure defaults are used
            except Exception as e:
                self.log.error(f"Error processing parameter file {user_parameters_path}: {e}. Using default parameters.")
                loaded_user_parameters = {} # Reset to ensure defaults are used
        else:
            self.log.info("No user parameter file provided. Using default parameters.")


        # --- Update defaults with loaded parameters ---
        # Iterate through the loaded parameters and update the defaults
        for key, val in loaded_user_parameters.items():
            if key in self.user_parameters:
                self.user_parameters[key] = val # Update default with loaded value
            else:
                self.log.warning(f"Parameter '{key}' from file is not a default parameter. Adding it.")
                self.user_parameters[key] = val # Add new parameter

        # Set learning_rate_start default if not provided
        if 'learning_rate_start' not in loaded_user_parameters:
            self.user_parameters['learning_rate_start'] = self.user_parameters['learning_rate']
            self.log.info(f"Using default learning_rate ({self.user_parameters['learning_rate']}) as learning_rate_start.")

        # --- Construct full paths using the 'input' directory ---
        # This needs to happen *after* merging defaults with loaded params,
        # in case 'input' dir itself was loaded from the file.
        input_dir = self.user_parameters['input']
        file_params_to_prefix = [
            'constraints', 'X_test', 'y_test', 'r_test',
            'X_train', 'y_train', 'r_train',
            'y_label_converter_path', 'r_label_converter_path'
        ]
        for param_key in file_params_to_prefix:
            current_path = self.user_parameters[param_key]
            # Check if it's likely just a filename (no directory component)
            # Also check if it's not already an absolute path
            if current_path and not os.path.dirname(current_path) and not os.path.isabs(current_path):
                self.user_parameters[param_key] = os.path.join(input_dir, current_path)
                self.log.info(f"Constructed path for '{param_key}': {self.user_parameters[param_key]}")


        # --- Force integer types using helper function ---
        # This should happen *after* loading and float conversion,
        # to correctly handle integer values loaded as floats (e.g., "5000.0")
        params_to_int = ['n_bit', 'n_iterations', 'report_freq', 'batch_size', 'n_cpu',
                         'use_region_info', 'region_embedding_dim'] # Added use_region_info, region_embedding_dim
        for param_key in params_to_int:
            self._convert_param_to_int(param_key) # This will raise error if conversion fails


        self.log.info(f"Final Parameters (after path construction & type conversion):")
        for key, val in self.user_parameters.items():
            self.log.info(f"{key}: {val} (type: {type(val).__name__})") # Log type as well

        # --- Thread limiting (Original) ---
        self.log.info(f"Limiting Torch to {self.user_parameters['n_cpu']} threads")
        torch.set_num_threads(self.user_parameters['n_cpu'])

        # --- Save parameters (Original) ---
        # Ensure output directory exists before saving parameters
        output_dir = self.user_parameters['output'] # Get final output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.log.info(f"Created output directory: {output_dir}")
        pd.DataFrame(self.user_parameters.values(), index=self.user_parameters.keys(), columns=['values']).to_csv(os.path.join(output_dir, 'used_user_parameters.csv'))

        # --- *** START: Create Symlinks for Input Files *** ---
        self.log.info("Creating symlinks for input files in output directory...")
        input_param_keys = [
            'constraints', 'X_test', 'y_test', 'r_test',
            'X_train', 'y_train', 'r_train',
            'y_label_converter_path', 'r_label_converter_path'
        ]
        input_files_to_link = []
        # Add files specified in parameters
        for key in input_param_keys:
            path = self.user_parameters.get(key)
            if isinstance(path, str) and path: # Ensure it's a non-empty string
                input_files_to_link.append(path)

        # Add the user parameters file itself, if provided
        if user_parameters_path is not None and isinstance(user_parameters_path, str):
            input_files_to_link.append(user_parameters_path)
            # Also add the 'used_user_parameters.csv' we just saved
            input_files_to_link.append(os.path.join(output_dir, 'used_user_parameters.csv'))


        linked_count = 0
        skipped_count = 0
        error_count = 0
        for input_path in set(input_files_to_link): # Use set to avoid duplicates
            try:
                # Ensure the input path is absolute for robust linking
                abs_input_path = os.path.abspath(input_path)

                if not os.path.exists(abs_input_path):
                    self.log.warning(f"Input file not found, cannot create symlink: {abs_input_path}")
                    skipped_count += 1
                    continue

                filename = os.path.basename(abs_input_path)
                symlink_path = os.path.join(output_dir, filename)

                # Check if target path exists and handle appropriately
                if os.path.lexists(symlink_path): # Use lexists to check symlink itself
                    if os.path.islink(symlink_path):
                        # It's a symlink, remove it first to ensure it points correctly
                        self.log.debug(f"Removing existing symlink: {symlink_path}")
                        os.remove(symlink_path)
                    else:
                        # It's a regular file or directory, log a warning and skip
                        self.log.warning(f"Target path exists but is not a symlink, skipping: {symlink_path}")
                        skipped_count += 1
                        continue

                # Create the symlink (target, link_name)
                os.symlink(abs_input_path, symlink_path)
                self.log.info(f"Created symlink: {symlink_path} -> {abs_input_path}")
                linked_count += 1

            except OSError as e:
                # Catch specific OS errors during symlink creation
                self.log.error(f"Failed to create symlink for {input_path} -> {symlink_path}: {e}")
                error_count += 1
            except Exception as e:
                # Catch any other unexpected errors
                self.log.error(f"An unexpected error occurred while trying to symlink {input_path}: {e}")
                error_count += 1

        self.log.info(f"Symlinking complete. Created: {linked_count}, Skipped: {skipped_count}, Errors: {error_count}")
        # --- *** END: Create Symlinks for Input Files *** ---


        # --- Initialize attributes ---
        self.E_scaling_constant = None
        self.E = None
        self.P = None
        self.Pnormalized = None
        self.genes = None
        self.constraints = None
        self.encoder = None
        # *** REVERTED: Back to ModuleDict ***
        self.decoders = None # This was from an older version, ensure it's decoder for single decoder logic
        self.decoder = None  # Explicitly define for single decoder model
        self.region_embedder = None # Explicitly define for single decoder model

        # *** REMOVED: self.decoder and self.region_embedder (they are defined above now) ***
        self.optimizer_gen = None
        self.learning_stats = {}
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.is_initialized_from_file = False # Flag to track if model state was loaded
        self.type_cooccurrence_mask = None 

        # Data attributes
        self.X_train, self.y_train, self.r_train = None, None, None
        self.X_test, self.y_test, self.r_test = None, None, None
        self.n_genes = None # Will be set in initialize
        self.n_categories = None
        self.n_regions = None
        self.y_label_map = None # Forward map (original label -> internal index 0..N-1)
        self.r_label_map = None # Forward map (original label -> internal index 0..N-1)
        self.y_reverse_label_map = None # Reverse map (internal index 0..N-1 -> string label)
        self.r_reverse_label_map = None # Reverse map (internal index 0..N-1 -> string label)
        self.y_unique_labels = None # Stores original y labels
        self.r_unique_labels = None # Stores original r labels
        self.mapped_region_indices = None # Internal region indices [0..N-1]
        self.y_parent_child_map = None # Stores {internal_parent_idx: [internal_child1_idx, ...]}
        self.y_child_to_parent_map = None # Optional: Stores {internal_child_idx: internal_parent_idx}


    def _convert_param_to_int(self, param_key):
        """
        Attempts to convert a specific parameter in self.user_parameters to int.
        Logs and raises error on failure. Handles values that might be float first.
        """
        try:
            original_value = self.user_parameters[param_key]
            # Convert to float first to handle strings like "5000.0"
            float_value = float(original_value)
            # Check if it's essentially an integer before converting
            if float_value.is_integer():
                self.user_parameters[param_key] = int(float_value)
            else:
                # If it has a decimal part, raise an error as we expect an integer
                raise ValueError(f"Value '{original_value}' is not a whole number.")
        except KeyError:
            self.log.error(f"Parameter '{param_key}' not found for integer conversion.")
            raise KeyError(f"Required parameter '{param_key}' is missing.")
        except (ValueError, TypeError) as e:
            self.log.error(f"Error converting parameter '{param_key}' to int. Value was '{original_value}' (type: {type(original_value).__name__}). Error: {e}")
            raise ValueError(f"Could not convert parameter '{param_key}' to integer. Invalid value: '{original_value}'.")

    def initialize(self):
        """
        Loads input data, constraints, label converters, processes labels,
        and attempts to load a pre-trained model state if available.
        Sets up necessary instance attributes. Uses original label mapping logic.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        self.log.info("--- Starting Initialization (Original Logic) ---")
        current_device = self.user_parameters['device']
        output_dir = self.user_parameters['output']
        model_state_path = os.path.join(output_dir, 'final_model_state.pt')
        # *** REMOVED use_region_info handling here, using original structure ***

        try:
            # --- Load Constraints ---
            self.log.info("Loading Gene Constraints")
            constraints_path = self.user_parameters['constraints']
            constraints_df = pd.read_csv(constraints_path, index_col=0)
            self.genes = np.array(constraints_df.index)
            if 'constraints' not in constraints_df.columns:
                raise KeyError(f"Column 'constraints' not found in {constraints_path}")
            self.constraints = torch.tensor(constraints_df['constraints'].values, dtype=torch.float32, device=current_device)
            self.n_genes = len(self.genes) # Set n_genes attribute
            self.log.info(f"Loaded {self.n_genes} genes from constraints.")

            # --- Load Data (X, y, r) ---
            def load_tensor(path, dtype, device):
                self.log.info(f"Loading {os.path.basename(path)} from {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data file not found: {path}")
                loaded_data = torch.load(path)
                if not isinstance(loaded_data, torch.Tensor):
                    tensor = torch.tensor(loaded_data, dtype=dtype, device=device)
                else:
                    tensor = loaded_data.to(dtype=dtype, device=device)
                return tensor
            
            self.X_train = load_tensor(self.user_parameters['X_train'], torch.float32, current_device)
            self.X_test = load_tensor(self.user_parameters['X_test'], torch.float32, current_device)

            self.y_train = load_tensor(self.user_parameters['y_train'], torch.long, current_device)
            self.y_test = load_tensor(self.user_parameters['y_test'], torch.long, current_device)
            all_y_labels = torch.cat((self.y_train, self.y_test))
            self.updated_y_label_map = {label.item(): i for i, label in enumerate(torch.unique(all_y_labels))}
            self.y_train = torch.tensor([self.updated_y_label_map[y.item()] for y in self.y_train], dtype=torch.long, device=current_device)
            self.y_test = torch.tensor([self.updated_y_label_map[y.item()] for y in self.y_test], dtype=torch.long, device=current_device)
            y_converter_path = self.user_parameters['y_label_converter_path']
            self.log.info(f"Loading y label converter from: {y_converter_path}")
            y_converter = pd.read_csv(y_converter_path, index_col=0)
            y_converter = dict(zip(y_converter.index,y_converter['label']))
            self.y_label_map = {k:self.updated_y_label_map[j] for k,j in y_converter.items()}
            self.y_reverse_label_map = {j:k for k,j in self.y_label_map.items()}
            all_y_labels = torch.cat((self.y_train, self.y_test))
            unique_y_labels_tensor = torch.unique(all_y_labels)
            self.n_categories = len(unique_y_labels_tensor)
            self.mapped_category_indices = list(range(self.n_categories)) # Internal indices 0..N-1

            self.r_train = load_tensor(self.user_parameters['r_train'], torch.long, current_device)
            self.r_test = load_tensor(self.user_parameters['r_test'], torch.long, current_device)
            if self.user_parameters['use_region_info']:
                all_r_labels = torch.cat((self.r_train, self.r_test))
                self.updated_r_label_map = {label.item(): i for i, label in enumerate(torch.unique(all_r_labels))}
                self.r_train = torch.tensor([self.updated_r_label_map[r.item()] for r in self.r_train], dtype=torch.long, device=current_device)
                self.r_test = torch.tensor([self.updated_r_label_map[r.item()] for r in self.r_test], dtype=torch.long, device=current_device)
                r_converter_path = self.user_parameters['r_label_converter_path']
                self.log.info(f"Loading y label converter from: {r_converter_path}")
                r_converter = pd.read_csv(r_converter_path, index_col=0)
                r_converter = dict(zip(r_converter.index,r_converter['label']))
                self.r_label_map = {k:self.updated_r_label_map[j] for k,j in r_converter.items()}
                self.r_reverse_label_map = {j:k for k,j in self.r_label_map.items()}
            else:
                self.r_train = self.r_train*0
                self.r_test = self.r_test*0
                self.r_label_map = {0:'Global'}
                self.r_reverse_label_map = {j:k for k,j in self.r_label_map.items()}
            all_r_labels = torch.cat((self.r_train, self.r_test))
            unique_r_labels_tensor = torch.unique(all_r_labels)
            self.n_regions = len(unique_r_labels_tensor)
            self.mapped_region_indices = list(range(self.n_regions)) # Internal indices 0..N-1
            if self.user_parameters['region_embedding_dim'] == -1:
                self.user_parameters['region_embedding_dim'] = self.n_regions

            self.log.info(f"Number of regions being used: {self.n_regions}")
            self.log.info(f"Region indices being used: {self.mapped_region_indices}")
            self.log.info(f"Region reverse map: {self.r_reverse_label_map}")

            # --- Validate Data Shapes ---
            if self.X_train.shape[1] != self.n_genes:
                raise ValueError(f"X_train gene dimension mismatch")
            if self.X_test.shape[1] != self.n_genes:
                raise ValueError(f"X_test gene dimension mismatch")
            if not (self.X_train.shape[0] == self.y_train.shape[0] == self.r_train.shape[0]):
                raise ValueError(f"Training data shape mismatch")
            if not (self.X_test.shape[0] == self.y_test.shape[0] == self.r_test.shape[0]):
                raise ValueError(f"Testing data shape mismatch")
            self.log.info("Data loaded and shapes validated.")
            self.log.info(f"Inferred {self.n_categories} cell type categories.")


            # --- Initialize Model Components (Encoder, Region Embedder, Decoder) ---
            # *** RESTORED: Single decoder structure initialization ***
            self.encoder = nn.Embedding(self.n_genes, self.user_parameters['n_bit']).to(current_device)
            self.region_embedder = nn.Embedding(self.n_regions, self.user_parameters['region_embedding_dim']).to(current_device)
            self.decoder = nn.Linear(self.user_parameters['n_bit'] + self.user_parameters['region_embedding_dim'], self.n_categories).to(current_device)
            self.log.info(f"Initialized encoder, region embedder (dim={self.user_parameters['region_embedding_dim']}), and single linear decoder.")
            # *** END RESTORED ***

            # --- Pre-calculate Type Co-occurrence Mask ---
            self.log.info("Calculating type co-occurrence mask based on training data regions...")
            self.type_cooccurrence_mask = torch.zeros((self.n_categories, self.n_categories), dtype=torch.bool, device=current_device)
            if self.user_parameters['use_region_info']: # Only calculate if using regions
                for r_idx in self.mapped_region_indices:
                    region_mask = (self.r_train == r_idx) # Use internal region indices
                    if region_mask.sum() > 0:
                        y_in_region = torch.unique(self.y_train[region_mask]) # Use internal y indices
                        for i in range(len(y_in_region)):
                            for j in range(i+1, len(y_in_region)): # Only consider distinct pairs
                                type_i = y_in_region[i].item()
                                type_j = y_in_region[j].item()
                                if 0 <= type_i < self.n_categories and 0 <= type_j < self.n_categories:
                                    self.type_cooccurrence_mask[type_i, type_j] = True
                                    self.type_cooccurrence_mask[type_j, type_i] = True
                num_cooccurring_pairs = self.type_cooccurrence_mask.sum().item() // 2
                self.log.info(f"Found {num_cooccurring_pairs} co-occurring type pairs across regions.")
            else:
                self.type_cooccurrence_mask = ~torch.eye(self.n_categories, dtype=torch.bool, device=current_device)
                self.log.info("Not using region info, all type pairs considered co-occurring for correlation loss.")

            # --- Load Cell Type Hierarchy (New Section) ---
            hierarchy_file_path = self.user_parameters.get('y_hierarchy_file_path', None)
            if isinstance(hierarchy_file_path, str) and os.path.exists(hierarchy_file_path) and \
            self.user_parameters.get('hierarchical_scatter_weight', 0) != 0:
                self.log.info(f"Loading cell type hierarchy from: {hierarchy_file_path}")
                try:
                    hierarchy_df = pd.read_csv(hierarchy_file_path)
                    if 'child_label' not in hierarchy_df.columns or 'parent_label' not in hierarchy_df.columns:
                        self.log.error("Hierarchy file must contain 'child_label' and 'parent_label' columns. Hierarchical loss disabled.")
                        self.y_parent_child_map = None
                    else:
                        self.y_parent_child_map = {}
                        self.y_child_to_parent_map = {} # Initialize this too if needed elsewhere

                        for _, row in hierarchy_df.iterrows():
                            child_str = str(row['child_label'])
                            parent_str = str(row['parent_label'])

                            if child_str in self.y_label_map and parent_str in self.y_label_map:
                                child_idx = self.y_label_map[child_str]
                                parent_idx = self.y_label_map[parent_str]

                                if parent_idx not in self.y_parent_child_map:
                                    self.y_parent_child_map[parent_idx] = []
                                if child_idx not in self.y_parent_child_map[parent_idx]: # Avoid duplicates
                                    self.y_parent_child_map[parent_idx].append(child_idx)
                                
                                # Populate child_to_parent map (can be useful for validation or other logic)
                                if child_idx in self.y_child_to_parent_map and self.y_child_to_parent_map[child_idx] != parent_idx:
                                    self.log.warning(f"Child type '{child_str}' (idx {child_idx}) mapped to multiple parents. Using last one: '{parent_str}' (idx {parent_idx}).")
                                self.y_child_to_parent_map[child_idx] = parent_idx
                            else:
                                self.log.warning(f"Skipping hierarchy entry: child '{child_str}' or parent '{parent_str}'. Label(s) not found in y_label_map.")
                        
                        if self.y_parent_child_map:
                            self.log.info(f"Successfully processed hierarchy: {len(self.y_parent_child_map)} parent groups mapped.")
                            # For debugging, you can log the structure:
                            # for p_idx, c_indices in self.y_parent_child_map.items():
                            #     p_name = self.y_reverse_label_map.get(p_idx, f"ParentIdx_{p_idx}")
                            #     c_names = [self.y_reverse_label_map.get(c_idx, f"ChildIdx_{c_idx}") for c_idx in c_indices]
                            #     self.log.debug(f"Hierarchy: {p_name} -> {c_names}")
                        else:
                            self.log.warning("Hierarchy map is empty after processing the file. Hierarchical loss might not be effective.")

                except Exception as e:
                    self.log.error(f"Failed to load or process hierarchy file {hierarchy_file_path}: {e}. Hierarchical loss disabled.")
                    self.y_parent_child_map = None
            elif self.user_parameters.get('hierarchical_scatter_weight', 0) != 0:
                self.log.warning("hierarchical_scatter_weight > 0 but 'y_hierarchy_file_path' is not provided, file not found, or path is not a string. Hierarchical loss will not be active.")
                self.y_parent_child_map = None
            # --- End of New Section for Hierarchy ---

            # --- Attempt to Load Pre-trained Model State ---
            if os.path.exists(model_state_path):
                self.log.info(f"Found existing model state file: {model_state_path}. Attempting to load.")
                try:
                    loaded_state_dict = torch.load(model_state_path, map_location=current_device)
                    # *** Use strict=False because the decoder structure changed ***
                    missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=False)
                    if missing_keys: self.log.warning(f"Missing keys when loading state_dict: {missing_keys}")
                    if unexpected_keys: self.log.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")

                    self.to(current_device) # Ensure model is on correct device
                    self.is_initialized_from_file = True
                    self.log.info("Successfully loaded model state from file (strict=False).")

                    # Calculate and store constrained E based on loaded model
                    self.eval() # Set to eval mode for get_encoding_weights
                    with torch.no_grad():
                        final_E = self.get_encoding_weights().detach().clone()
                        self.log.info("Enforcing constraints on loaded E matrix...")
                        if self.constraints is None:
                            self.log.error("Cannot enforce constraints: self.constraints is None.")
                            self.E = final_E # Store unconstrained E if constraints missing
                        else:
                            E_final_constrained = torch.clip(final_E.round(), 0, None)
                            T = self.constraints.clone().detach()
                            m = E_final_constrained.sum(1) > T
                            if m.any():
                                scaling_factors = (T[m] / E_final_constrained.sum(1)[m].clamp(min=1e-8)).unsqueeze(1)
                                E_final_constrained[m, :] = (E_final_constrained[m, :] * scaling_factors).floor()
                                E_final_constrained = E_final_constrained.clamp(min=0)
                            self.E = E_final_constrained.clone().detach() # Store final constrained E
                            self.log.info(f"Stored constrained E matrix from loaded model. Probe count: {self.E.sum().item():.2f}")


                except Exception as e:
                    self.log.error(f"Failed to load model state from {model_state_path}: {e}. Model will be trained from scratch.")
                    self.is_initialized_from_file = False
                    # Reset potentially partially initialized model components
                    # *** Ensure correct components are reset ***
                    self.encoder = None
                    self.region_embedder = None
                    self.decoder = None
                    self.E = None
            else:
                self.log.info(f"No existing model state file found at {model_state_path}. Model will be trained from scratch.")
                self.is_initialized_from_file = False
                # *** Ensure components are initialized if not loading ***
                if self.encoder is None:
                    self.encoder = nn.Embedding(self.n_genes, self.user_parameters['n_bit']).to(current_device)
                if self.region_embedder is None:
                    self.region_embedder = nn.Embedding(self.n_regions, self.user_parameters['region_embedding_dim']).to(current_device)
                if self.decoder is None:
                    self.decoder = nn.Linear(self.user_parameters['n_bit'] + self.user_parameters['region_embedding_dim'], self.n_categories).to(current_device)
                self.log.info("Initialized new model components.")


            self.log.info("--- Initialization Complete ---")
            return True

        except FileNotFoundError as e:
            self.log.error(f"Initialization failed: Input file not found. {e}")
            return False
        except KeyError as e:
            self.log.error(f"Initialization failed: Missing expected column or key. {e}")
            return False
        except ValueError as e:
            self.log.error(f"Initialization failed: Data validation error. {e}")
            return False
        except Exception as e:
            self.log.exception(f"An unexpected error occurred during initialization: {e}")
            return False

    def get_encoding_weights(self):
        """ Get encoding weights from the shared encoder. """
        if self.encoder is None:
            raise RuntimeError("Encoder not initialized. Call initialize() or fit() first.")
        E = F.softplus(self.encoder.weight)
        if self.E_scaling_constant is None:
            self.E_scaling_constant = (self.user_parameters['total_n_probes'] / E.sum().clamp(min=1e-8)).detach()
        E = E * self.E_scaling_constant
        if self.training and self.user_parameters['weight_dropout_proportion'] > 0:
            dropout_mask_E = (torch.rand_like(E) > self.user_parameters['weight_dropout_proportion']).float()
            E = E * dropout_mask_E
        return E

    def project(self, X, E):
        """ Project the input data using the encoding weights. """
        P = X.mm(E)
        if self.user_parameters['noise_level'] != 0:
            noise = (2 * torch.rand_like(P) - 1) * (10 ** self.user_parameters['noise_level'])
            P = torch.clip(P + noise, min=1.0)
        P_sum = P.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P_mean_sum = P_sum.mean().clamp(min=1e-8)
        P = P * (P_mean_sum / P_sum)
        input_to_tanh = (P.clamp(min=1).log10() - self.user_parameters['target_brightness_log'])
        Pnormalized = (self.user_parameters['tanh_slope_factor'] * input_to_tanh).tanh() # Modified line
        if self.training and self.user_parameters['projection_dropout_proportion'] > 0:
            dropout_mask_P = (torch.rand_like(Pnormalized) > self.user_parameters['projection_dropout_proportion']).float()
            Pnormalized_dropout = Pnormalized * dropout_mask_P
        else:
            Pnormalized_dropout = Pnormalized
        return P, Pnormalized, Pnormalized_dropout

    def decode(self, Pnormalized_dropout, r_labels, y):
        """
        Decode the projected data using the single decoder, incorporating region embeddings.

        Args:
            Pnormalized_dropout (Tensor): The dropout-applied normalized projection (batch_size x n_bits).
            r_labels (Tensor): Region labels for the batch (batch_size).
            y (Tensor): True cell type labels for the batch (batch_size).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predicted labels, accuracy, categorical loss.
        """
        if self.decoder is None or self.region_embedder is None:
            raise RuntimeError("Decoder or region_embedder not initialized.")
        if not isinstance(self.decoder, nn.Module):
            raise ValueError("Invalid decoder module.")
        if not isinstance(self.region_embedder, nn.Module):
            raise ValueError("Invalid region_embedder module.")

        # Get region embeddings for the batch
        if r_labels.min() < 0 or r_labels.max() >= self.n_regions:
            self.log.error(f"Region labels out of bounds ({r_labels.min()}-{r_labels.max()}) for region_embedder (expected 0-{self.n_regions-1}).")
            r_labels = torch.clamp(r_labels, 0, self.n_regions - 1)

        region_embeds = self.region_embedder(r_labels) # Shape: [batch_size, self.user_parameters['region_embedding_dim']]

        # Concatenate projection and region embedding
        decoder_input = torch.cat([Pnormalized_dropout, region_embeds], dim=1)

        # Pass through the linear decoder
        R = self.decoder(decoder_input) # Shape: [batch_size, n_categories]

        # Calculate predictions and accuracy
        y_predict = R.max(1)[1]
        accuracy = (y_predict == y).float().mean()

        # Calculate categorical loss
        if self.user_parameters['categorical_weight'] != 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            if y.min() < 0 or y.max() >= self.n_categories:
                self.log.error(f"Target labels y out of bounds ({y.min()}-{y.max()}) for CrossEntropyLoss (expected 0-{self.n_categories-1}).")
                categorical_loss = torch.tensor(0.0, device=R.device, requires_grad=True) # Fallback
            else:
                # Raw categorical loss value
                categorical_loss = loss_fn(R, y)
        else:
            categorical_loss = torch.tensor(0.0, device=R.device, requires_grad=True)

        return y_predict, accuracy, categorical_loss

    def calculate_loss(self, X, y, r_labels, iteration, suffix=''):
        """
        Calculate loss for the EncodingDesigner model for a batch of data.
        Uses the single decoder structure and co-occurrence mask for type correlation.
        Returns raw loss components for GradNorm.

        Args:
            X (Tensor): Input data batch.
            y (Tensor): Cell type labels batch.
            r_labels (Tensor): Region labels batch.
            iteration (int): Current training iteration.
            suffix (str): Suffix for logging stats ('_train' or '_test').

        Returns:
            Tuple[Dict, Dict]: Dictionary of raw loss tensors, dictionary of stats.
        """
        # --- Shared Encoder Forward Pass ---
        E = self.get_encoding_weights()
        P_original, Pnormalized, Pnormalized_dropout = self.project(X, E) # Pnormalized is the tanh output

        # --- Create P_rescaled for scale-sensitive loss terms ---
        P_rescaled = P_original / (10**self.user_parameters['target_brightness_log']) 

        # --- Decode using the single decoder (incorporates region embedding) ---
        y_predict, accuracy, raw_categorical_loss_component = self.decode(Pnormalized_dropout, r_labels, y)

        # --- Initialize Dictionaries ---
        raw_losses = {}
        current_stats = {}

        # --- Calculate Stats (on the full batch) ---
        current_stats['accuracy' + suffix] = accuracy.item()
        current_stats['median brightness' + suffix] = P_original.median().item()

        # -- Probe Weight Loss (Global) --
        probe_count = E.sum()
        current_stats['total_n_probes' + suffix] = probe_count.item()

        raw_probe_weight_loss = torch.tensor(0.0, device=probe_count.device, dtype=probe_count.dtype)
        probe_weight_for_over = self.user_parameters['probe_weight']
        push_down_weight_for_under = self.user_parameters['probe_under_weight_factor']

        if probe_weight_for_over != 0.0 or push_down_weight_for_under != 0.0:
            total_n_probes_target = float(self.user_parameters['total_n_probes'])
            penalty_over = torch.tensor(0.0, device=probe_count.device, dtype=probe_count.dtype)
            if probe_weight_for_over != 0.0:
                diff_over = probe_count - total_n_probes_target
                penalty_over = probe_weight_for_over * (F.relu(diff_over) + 1).log10()

            penalty_push_down = torch.tensor(0.0, device=probe_count.device, dtype=probe_count.dtype)
            if push_down_weight_for_under != 0.0:
                if probe_count < total_n_probes_target:
                    safe_total_n_probes_target = max(total_n_probes_target, 1e-8)
                    normalized_under_log_argument = (probe_count / safe_total_n_probes_target) + 1
                    penalty_push_down = push_down_weight_for_under * normalized_under_log_argument.log10()
            raw_probe_weight_loss = penalty_over + penalty_push_down
        
        raw_losses['probe_weight'] = raw_probe_weight_loss
        # Log raw value for probe_weight_loss, effective weight applied in fit
        # For consistency in logging, if gradnorm_original_static_weights is available, use it
        static_probe_weight_for_logging = self.user_parameters['probe_weight']
        current_stats['probe_weight_loss' + suffix] = raw_probe_weight_loss.item() * static_probe_weight_for_logging


        # -- Categorical Loss --
        if self.user_parameters['categorical_weight'] != 0: # Still check if static weight is zero to disable calculation
            raw_losses['categorical'] = raw_categorical_loss_component
            # Log weighted by original static weight for comparison to pre-GradNorm runs
            current_stats['categorical_loss' + suffix] = raw_categorical_loss_component.item() * self.user_parameters['categorical_weight']
        else:
            raw_losses['categorical'] = torch.tensor(0.0, device=self.user_parameters['device'])
            current_stats['categorical_loss' + suffix] = 0.0

        # -- Gene Constraint Loss (Global) --
        raw_gene_constraint_loss = torch.tensor(0.0, device=self.user_parameters['device'])
        if self.user_parameters['gene_constraint_weight'] != 0:
            if self.constraints is None: raise RuntimeError("Constraints not loaded. Run initialize() first.")
            constraint_violation = F.relu(E.sum(dim=1) - self.constraints)
            raw_gene_constraint_loss = torch.sqrt(constraint_violation.mean().clamp(min=1e-8))
            current_stats['gene_constraint_loss' + suffix] = raw_gene_constraint_loss.item() * self.user_parameters['gene_constraint_weight']
        else:
            current_stats['gene_constraint_loss' + suffix] = 0.0
        raw_losses['gene_constraint'] = raw_gene_constraint_loss
        

        # -- P Standard Deviation Loss (Minimizing -CV, so maximizing CV) --
        raw_p_std_loss = torch.tensor(0.0, device=self.user_parameters['device'])
        min_p_cv_val = np.nan 

        if self.user_parameters['pnorm_std_weight'] != 0 and P_rescaled.shape[0] > 1 and P_rescaled.shape[1] > 0: 
            mean_per_bit = P_rescaled.mean(dim=0)
            std_per_bit = P_rescaled.std(dim=0) 
            epsilon = 1e-8 # Avoid division by zero if mean is zero or very small
            cv_per_bit = std_per_bit / (mean_per_bit + epsilon)
            min_p_cv_tensor = cv_per_bit.min()
            min_p_cv_val = min_p_cv_tensor.item()
            raw_p_std_loss = -min_p_cv_tensor # GradNorm minimizes this (thus maximizes min_p_cv_tensor)
            current_stats['p_std_loss' + suffix] = raw_p_std_loss.item() * self.user_parameters['pnorm_std_weight']
        else:
            current_stats['p_std_loss' + suffix] = 0.0
        raw_losses['p_std'] = raw_p_std_loss
        current_stats['p_std_min' + suffix] = min_p_cv_val # Log the actual min CV value
        

        # -- Type Correlation Loss (using co-occurrence mask) --
        # This loss uses original P_original to calculate P_type_batch for correlations
        P_type_batch = torch.zeros((self.n_categories, P_original.shape[1]), device=P_original.device)
        unique_y_batch, y_batch_indices = torch.unique(y, return_inverse=True) 
        valid_types_in_batch_mask = torch.zeros(self.n_categories, dtype=torch.bool, device=P_original.device)

        for i, type_idx in enumerate(unique_y_batch):
            if 0 <= type_idx.item() < self.n_categories:
                mask = (y == type_idx)
                P_type_batch[type_idx] = P_original[mask].mean(dim=0) # Use P_original here
                valid_types_in_batch_mask[type_idx] = True

        P_corr = P_type_batch[valid_types_in_batch_mask] 
        batch_type_indices = torch.where(valid_types_in_batch_mask)[0] 
        n_types_batch = P_corr.shape[0]

        # Create P_corr_rescaled for losses that need it
        # P_corr is based on P_original, so we need to derive a rescaled version for relevant losses.
        # This can be done by calculating means from P_rescaled for the valid types.
        P_corr_rescaled = torch.zeros((n_types_batch, P_rescaled.shape[1]), device=P_rescaled.device)
        if n_types_batch > 0:
            temp_idx_map = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(batch_type_indices)}
            for i_orig_y, type_idx_tensor in enumerate(unique_y_batch): # Iterate through types present in batch y
                type_idx_item = type_idx_tensor.item()
                if type_idx_item in temp_idx_map: # If this type is part of P_corr
                    mask = (y == type_idx_tensor) # Mask from original y to get correct cells
                    new_idx_for_P_corr_rescaled = temp_idx_map[type_idx_item]
                    P_corr_rescaled[new_idx_for_P_corr_rescaled] = P_rescaled[mask].mean(dim=0)

        raw_type_correlation_max_loss = torch.tensor(0.0, device=self.user_parameters['device'])

        if (n_types_batch > 1) and (self.user_parameters['type_correlation_max_weight'] != 0): # Only calculate if weight is non-zero
            n_bits = P_corr.shape[1]
            P_type_centered_types = P_corr - P_corr.mean(dim=1, keepdim=True)
            P_type_std_types = P_type_centered_types.std(dim=1, keepdim=True).clamp(min=1e-6)
            P_type_norm_types = P_type_centered_types / P_type_std_types 
            correlation_matrix_types = P_type_norm_types @ P_type_norm_types.T / n_bits 

            batch_off_diag_mask = torch.eye(n_types_batch, device=P_corr.device) == 0
            batch_cooccurrence_mask = self.type_cooccurrence_mask[batch_type_indices][:, batch_type_indices]
            final_corr_mask = batch_off_diag_mask & batch_cooccurrence_mask
            relevant_corrs = correlation_matrix_types[final_corr_mask]

            if relevant_corrs.numel() > 0:
                current_stats['type_correlation_max' + suffix] = relevant_corrs.abs().max().item()
                current_stats['type_correlation_min' + suffix] = relevant_corrs.min().item()
                current_stats['type_correlation_mean' + suffix] = relevant_corrs.mean().item()
                correlation_thresh = self.user_parameters['correlation_thresh']
                off_diag_corr_types_loss = F.relu((relevant_corrs.abs() - correlation_thresh) / (correlation_thresh + 1e-8))
                
                raw_type_correlation_max_loss = off_diag_corr_types_loss.max()
                current_stats['type_correlation_max_loss' + suffix] = raw_type_correlation_max_loss.item() * self.user_parameters['type_correlation_max_weight']
                # Note: type_correlation_mean_loss is not part of GradNorm tasks by default per instructions
                if self.user_parameters['type_correlation_mean_weight'] != 0: 
                    type_correlation_mean_loss_val = self.user_parameters['type_correlation_mean_weight'] * off_diag_corr_types_loss.mean()
                    current_stats['type_correlation_mean_loss' + suffix] = type_correlation_mean_loss_val.item()
                else:
                    current_stats['type_correlation_mean_loss' + suffix] = 0.0
            else: 
                current_stats['type_correlation_max' + suffix] = np.nan
                current_stats['type_correlation_min' + suffix] = np.nan
                current_stats['type_correlation_mean' + suffix] = np.nan
                current_stats['type_correlation_max_loss' + suffix] = 0.0
                current_stats['type_correlation_mean_loss' + suffix] = 0.0
        else: 
            current_stats['type_correlation_max' + suffix] = np.nan
            current_stats['type_correlation_min' + suffix] = np.nan
            current_stats['type_correlation_mean' + suffix] = np.nan
            current_stats['type_correlation_max_loss' + suffix] = 0.0
            current_stats['type_correlation_mean_loss' + suffix] = 0.0
        raw_losses['type_correlation_max'] = raw_type_correlation_max_loss

        # --- Hierarchical Scatter Loss (Minimizing -Value, so maximizing Value) ---
        hierarchical_scatter_value_stat = 0.0 
        raw_hierarchical_scatter_loss = torch.tensor(0.0, device=P_original.device) 

        P_for_scatter = P_original 
        hierarchical_scatter_weight = self.user_parameters.get('hierarchical_scatter_weight', 0)
        if hierarchical_scatter_weight != 0 and \
           hasattr(self, 'y_parent_child_map') and self.y_parent_child_map and \
           P_for_scatter.shape[0] > 1: 

            unique_y_in_batch, y_counts_in_batch = torch.unique(y, return_counts=True)
            map_batch_labels_to_counts = {label.item(): count.item() for label, count in zip(unique_y_in_batch, y_counts_in_batch)}
            accumulated_weighted_squared_distances = []

            for parent_idx, child_idx_list in self.y_parent_child_map.items():
                if parent_idx in map_batch_labels_to_counts and map_batch_labels_to_counts[parent_idx] >= 1: 
                    parent_mask = (y == parent_idx)
                    parent_centroid = P_for_scatter[parent_mask].mean(dim=0)

                    for child_idx in child_idx_list:
                        if child_idx in map_batch_labels_to_counts and map_batch_labels_to_counts[child_idx] >= 1: 
                            child_mask = (y == child_idx)
                            child_centroid = P_for_scatter[child_mask].mean(dim=0)
                            n_child_in_batch = map_batch_labels_to_counts[child_idx]
                            squared_distance = ((child_centroid - parent_centroid)**2).sum()
                            weighted_squared_distance = n_child_in_batch * squared_distance
                            accumulated_weighted_squared_distances.append(weighted_squared_distance)

            if accumulated_weighted_squared_distances:
                hierarchical_scatter_value = torch.stack(accumulated_weighted_squared_distances).sum()
                raw_hierarchical_scatter_loss = -hierarchical_scatter_value # GradNorm minimizes this
                hierarchical_scatter_value_stat = hierarchical_scatter_value.item()
                current_stats['hierarchical_scatter_loss' + suffix] = raw_hierarchical_scatter_loss.item() * hierarchical_scatter_weight
            else:
                current_stats['hierarchical_scatter_loss' + suffix] = 0.0
        else:
            current_stats['hierarchical_scatter_loss' + suffix] = 0.0
        
        raw_losses['hierarchical_scatter'] = raw_hierarchical_scatter_loss # Add raw loss for fit() to use
        current_stats['hierarchical_scatter_value' + suffix] = hierarchical_scatter_value_stat

        # --- Intra-Type Variance Loss (Maximizing variance of each type's projection across bits) ---
        # Uses P_corr_rescaled
        raw_intra_type_variance_loss = torch.tensor(0.0, device=self.user_parameters['device'])
        if self.user_parameters['intra_type_variance_weight'] != 0:
            if n_types_batch > 0 and P_corr_rescaled.shape[1] > 0: # Need at least 1 type and some bits
                # P_corr_rescaled has shape [n_types_batch, n_bits]
                variances_intra_type = torch.var(P_corr_rescaled, dim=1) # Variance across bits for each type
                raw_intra_type_variance_loss = -torch.mean(variances_intra_type) # Negative mean to maximize
                current_stats['intra_type_variance_loss' + suffix] = raw_intra_type_variance_loss.item() * self.user_parameters['intra_type_variance_weight']
            else:
                current_stats['intra_type_variance_loss' + suffix] = 0.0
        else:
            current_stats['intra_type_variance_loss' + suffix] = 0.0
        raw_losses['intra_type_variance'] = raw_intra_type_variance_loss

        # --- Bit IQR Variance Loss (Maximizing IQR of each bit's projection across types) ---
        # Uses P_corr_rescaled
        raw_bit_iqr_variance_loss = torch.tensor(0.0, device=self.user_parameters['device'])
        if self.user_parameters['bit_iqr_variance_weight'] != 0:
            if n_types_batch > 1 and P_corr_rescaled.shape[1] > 0: # Need at least 2 types and some bits
                # P_corr_rescaled has shape [n_types_batch, n_bits]
                # Calculate 25th and 75th percentiles for each bit (column-wise)
                q1 = torch.quantile(P_corr_rescaled, 0.25, dim=0)
                q3 = torch.quantile(P_corr_rescaled, 0.75, dim=0)
                iqr_per_bit = q3 - q1
                raw_bit_iqr_variance_loss = -torch.mean(iqr_per_bit) # Negative mean to maximize IQR
                current_stats['bit_iqr_variance_loss' + suffix] = raw_bit_iqr_variance_loss.item() * self.user_parameters['bit_iqr_variance_weight']
            else:
                current_stats['bit_iqr_variance_loss' + suffix] = 0.0
        else:
            current_stats['bit_iqr_variance_loss' + suffix] = 0.0
        raw_losses['bit_iqr_variance'] = raw_bit_iqr_variance_loss
        
        # --- Type Normalized Entropy Loss (Minimizing entropy of each type's normalized projection) ---
        # Uses original P_corr (derived from P_original)
        raw_type_entropy_loss = torch.tensor(0.0, device=self.user_parameters['device'])
        if self.user_parameters['type_entropy_weight'] != 0:
            if n_types_batch > 0 and P_corr.shape[1] > 0: # Need types and bits
                # P_corr is P_type_batch_valid, shape [n_types_batch, n_bits]
                # Ensure positivity for log, P_corr should be positive from project() then mean(). Add epsilon for safety.
                P_corr_positive = P_corr + 1e-12
                # Normalize each type's projection across bits to sum to 1
                P_type_norm_rows = P_corr_positive / (P_corr_positive.sum(dim=1, keepdim=True))
                
                # Calculate entropy for each row
                # Add epsilon inside log for numerical stability
                entropies_per_type = -torch.sum(P_type_norm_rows * torch.log(P_type_norm_rows + 1e-12), dim=1)
                raw_type_entropy_loss = torch.mean(entropies_per_type) # Minimize mean entropy
                current_stats['type_entropy_loss' + suffix] = raw_type_entropy_loss.item() * self.user_parameters['type_entropy_weight']
            else:
                current_stats['type_entropy_loss' + suffix] = 0.0
        else:
            current_stats['type_entropy_loss' + suffix] = 0.0
        raw_losses['type_entropy'] = raw_type_entropy_loss
        
        return raw_losses, current_stats

    def fit(self):
        """ Fits the EncodingDesigner model using a single decoder and vectorized batches. """
        # --- Check if initialized ---
        if self.X_train is None or self.y_train is None or self.r_train is None or self.constraints is None or self.decoder is None or self.region_embedder is None:
            self.log.error("Model is not initialized. Call initialize() before fit().")
            raise RuntimeError("Model is not initialized. Call initialize() before fit().")

        # --- Initializations ---
        self.learning_stats = {} # Reset learning stats for this fit run
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1
        start_time = time.time()
        current_device = self.user_parameters['device']
        self.E_scaling_constant = None # Reset scaling constant for training
        n_categories = self.n_categories # Get from initialize()

        # Get LR parameters
        lr_start = self.user_parameters['learning_rate_start']
        lr_end = self.user_parameters['learning_rate_end']

        # --- Timing and Loop Variables ---
        last_report_time = start_time
        last_report_iteration = 0
        n_iterations = self.user_parameters['n_iterations']
        report_freq = self.user_parameters['report_freq']
        batch_size = self.user_parameters['batch_size']
        n_train_samples = self.X_train.shape[0]

        # --- Training Loop ---
        try:
            for iteration in range(n_iterations):
                self.learning_stats[iteration] = {}

                # --- Initialization (Iteration 0, only if not loaded from file) ---
                if iteration == 0:
                    if not self.is_initialized_from_file:
                        self.log.info("Model not initialized from file, using randomly initialized weights.")
                    else:
                        self.log.info("Using model loaded during initialization.")
                    self.to(current_device)

                    # Optimizer uses STARTING LR - include all trainable parameters
                    optimizer_gen = torch.optim.Adam([
                        {'params': self.encoder.parameters(), 'lr': lr_start},
                        {'params': self.region_embedder.parameters(), 'lr': lr_start},
                        {'params': self.decoder.parameters(), 'lr': lr_start}
                    ])
                    self.optimizer_gen = optimizer_gen

                # Calculate and Set Current Learning Rate
                if n_iterations <= 1: current_lr = lr_start
                else: current_lr = lr_start + (lr_end - lr_start) * (iteration / (n_iterations - 1))
                for param_group in self.optimizer_gen.param_groups: param_group['lr'] = current_lr

                is_report_iter = (iteration % report_freq == 0) or (iteration == n_iterations - 1) # Report last iter

                # --- Training Step (Single Batch from Full Dataset) ---
                self.train() # Set model to training mode

                if (batch_size > 0) and (batch_size < n_train_samples):
                    idxs = np.random.choice(n_train_samples, batch_size, replace=False)
                    X_batch = self.X_train[idxs]
                    y_batch = self.y_train[idxs]
                    r_batch = self.r_train[idxs] 
                else: 
                    X_batch = self.X_train
                    y_batch = self.y_train
                    r_batch = self.r_train
                    if batch_size > 0: 
                        self.log.debug(f"Batch size {batch_size} >= dataset size {n_train_samples}. Using full dataset for iteration {iteration}.")

                # --- A. Get Raw Losses ---
                self.optimizer_gen.zero_grad() # Zero gradients for model parameters first
                
                raw_losses_batch, batch_stats = self.calculate_loss(
                    X_batch, y_batch, r_batch, iteration, suffix='_train'
                )
                self.learning_stats[iteration].update(batch_stats) # Store initial stats from calculate_loss

                # --- B. Determine Effective Loss Weights (Using Static Weights) ---
                # Define mapping from raw loss keys (from loss_component_names) to user_parameter keys for weights
                key_to_weight_param_map = {
                    'probe_weight': 'probe_weight',
                    'categorical': 'categorical_weight',
                    'gene_constraint': 'gene_constraint_weight',
                    'p_std': 'pnorm_std_weight',
                    'type_correlation_max': 'type_correlation_max_weight',
                    'hierarchical_scatter': 'hierarchical_scatter_weight',
                    'intra_type_variance': 'intra_type_variance_weight', # NEW
                    'bit_iqr_variance': 'bit_iqr_variance_weight', # NEW
                    'type_entropy': 'type_entropy_weight', # NEW
                }
                
                effective_weights_for_logging_and_use = {}
                for task_name_from_list in self.loss_component_names:
                    weight_param_key = key_to_weight_param_map.get(task_name_from_list)
                    if weight_param_key:
                        weight_value = self.user_parameters.get(weight_param_key, 0.0) # Default to 0.0 if key missing
                        effective_weights_for_logging_and_use[task_name_from_list] = weight_value
                        self.learning_stats[iteration][f'weight_{task_name_from_list}'] = weight_value
                    else:
                        # This case should ideally not happen if loss_component_names and map are consistent
                        self.log.warning(f"Task '{task_name_from_list}' from loss_component_names not found in key_to_weight_param_map. Weight will be 0.")
                        effective_weights_for_logging_and_use[task_name_from_list] = 0.0
                        self.learning_stats[iteration][f'weight_{task_name_from_list}'] = 0.0


                # --- C. Calculate Weighted Total Loss (total_loss_weighted) ---
                total_loss_weighted = torch.tensor(0.0, device=current_device)
  
                for task_name, raw_loss_component in raw_losses_batch.items():
                    if task_name in self.loss_component_names: # Only consider tasks in our defined list
                        current_weight = effective_weights_for_logging_and_use.get(task_name, 0.0)
                        if raw_loss_component is not None and current_weight > 0:
                            total_loss_weighted = total_loss_weighted + current_weight * raw_loss_component
                        # Logging of weight already done above when populating effective_weights_for_logging_and_use
  
                # Update the logged total_loss_train stat (this is the primary loss for best model tracking)
                self.learning_stats[iteration]['total_loss_train'] = total_loss_weighted.item()

                                # --- D. Main Model Parameter Update ---
                # Gradients for model parameters were already zeroed
                total_loss_weighted.backward() # No need for retain_graph=True

                nan_detected = False
                for name, param in self.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        nan_detected = True
                        self.log.warning(f"NaNs or Infs detected in gradients of model parameter '{name}' at iteration {iteration}. Skipping step and attempting revert.")
                        self.optimizer_gen.zero_grad() # Zero the corrupted model gradients
                        break
                
                if not nan_detected:
                    # Now, step the main model optimizer
                    self.optimizer_gen.step() 

                    # --- F. Update Stats and Best Model ---
                    current_loss_item = total_loss_weighted.item() 
                    # self.learning_stats[iteration]['total_loss_train'] was already updated

                    if not np.isnan(current_loss_item) and current_loss_item < self.best_loss:
                        self.best_loss = current_loss_item
                        self.best_model_state_dict = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
                        self.best_iteration = iteration
                        self.log.info(f"*** New best model found at iteration {iteration} (Train Loss: {self.best_loss:.4f}) ***")

                    if is_report_iter or iteration == self.best_iteration: 
                        self.saved_models[iteration] = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
                        self.saved_optimizer_states[iteration] = self.optimizer_gen.state_dict()
                else: # NaN detected in model parameter gradients
                    # --- Revert Logic ---
                    valid_iters = [k for k in self.saved_models if k < iteration]
                    if valid_iters:
                        revert_iter = max(valid_iters)
                        self.log.warning(f"Reverting model and optimizer to state from iteration {revert_iter}")
                        try:
                            self.load_state_dict(self.saved_models[revert_iter])
                            self.to(current_device)
                            optimizer_gen = torch.optim.Adam([
                                {'params': self.encoder.parameters(), 'lr': lr_start},
                                {'params': self.region_embedder.parameters(), 'lr': lr_start},
                                {'params': self.decoder.parameters(), 'lr': lr_start}
                            ])
                            optimizer_gen.load_state_dict(self.saved_optimizer_states[revert_iter])
                            self.optimizer_gen = optimizer_gen
                            for state in self.optimizer_gen.state.values():
                                for k, v in state.items():
                                    if isinstance(v, torch.Tensor):
                                        state[k] = v.to(current_device)
                        except Exception as e:
                            self.log.error(f"Failed to load state from iter {revert_iter}: {e}. Optimizer state might be reset.")
                            self.optimizer_gen = torch.optim.Adam([
                                {'params': self.encoder.parameters(), 'lr': lr_start},
                                {'params': self.region_embedder.parameters(), 'lr': lr_start},
                                {'params': self.decoder.parameters(), 'lr': lr_start}
                            ])
                        # Clear current iteration's stats as it's invalid due to revert
                        self.learning_stats[iteration] = {} # Reset dict for this iter
                        self.learning_stats[iteration]['status'] = f'Reverted from NaN at {iteration}'
                    else:
                        self.log.error(f"NaNs/Infs detected in gradients at iter {iteration}, but no previous state found. Stopping.")
                        raise ValueError("NaNs/Infs encountered and cannot revert.")
                    # --- End Revert ---

                # --- Evaluation Step (Periodically) ---
                if is_report_iter:
                    self.eval()
                    all_test_stats_list = [] # Renamed to avoid conflict
                    total_test_loss_items = []

                    with torch.no_grad():
                        # For evaluation, calculate loss with current effective_weights (could be static or gradnorm)
                        # This means calculate_loss will return raw_test_losses, and we weight them here.
                        raw_test_losses, test_stats_from_calc = self.calculate_loss(
                            self.X_test, self.y_test, self.r_test, iteration, suffix='_test'
                        )
                        
                        # Determine effective weights for test evaluation (consistent with training static weights)
                        # Re-use the key_to_weight_param_map defined earlier for consistency
                        eval_effective_weights = {}
                        for task_name_from_list in self.loss_component_names:
                            weight_param_key = key_to_weight_param_map.get(task_name_from_list)
                            if weight_param_key:
                                weight_value = self.user_parameters.get(weight_param_key, 0.0)
                                eval_effective_weights[task_name_from_list] = weight_value
                            else:
                                eval_effective_weights[task_name_from_list] = 0.0


                        # Calculate weighted test loss
                        weighted_test_loss_val = 0.0
                        for task_name_test, raw_loss_test_comp in raw_test_losses.items():
                            if task_name_test in self.loss_component_names: # Consider only relevant tasks
                                current_weight = eval_effective_weights.get(task_name_test, 0.0)
                                if raw_loss_test_comp is not None and current_weight > 0:
                                    weighted_test_loss_val += current_weight * raw_loss_test_comp.item()
                        
                        test_stats_from_calc['total_loss_test'] = weighted_test_loss_val # Add weighted total test loss to stats
                        all_test_stats_list.append(test_stats_from_calc)
                        total_test_loss_items.append(weighted_test_loss_val)

                    avg_test_stats = {}
                    if all_test_stats_list:
                        stat_keys = all_test_stats_list[0].keys()
                        for key in stat_keys:
                            values = [stats.get(key, np.nan) for stats in all_test_stats_list]
                            valid_values = [v for v in values if not np.isnan(v)]
                            avg_test_stats[key] = np.mean(valid_values) if valid_values else np.nan
                    else:
                        avg_test_stats = {}

                    self.learning_stats[iteration].update(avg_test_stats)
                    avg_test_loss_item = np.mean(total_test_loss_items) if total_test_loss_items else np.nan
                    self.learning_stats[iteration]['total_loss_test_avg'] = avg_test_loss_item # This is the primary test loss metric

                    current_time = time.time()
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elapsed_time = current_time - last_report_time
                    iterations_since_last = iteration - last_report_iteration + 1 # Avoid div by zero
                    avg_iter_time = elapsed_time / iterations_since_last if iterations_since_last > 0 else 0
                    self.log.info(f"Avg time/iter since last report: {avg_iter_time:.4f} seconds")
                    last_report_time = current_time
                    last_report_iteration = iteration
                    red_start = "\033[91m"; reset_color = "\033[0m"
                    # Report global evaluation
                    log_msg_header = f"--- Iteration: {iteration}/{n_iterations} Eval (Global Test Set) ---"
                    self.log.info(log_msg_header)
                    if self.user_parameters['Verbose'] == 1: print(f"{red_start}{log_msg_header}{reset_color}")

                    # Log current learning rate
                    log_msg_lr = f"Current LR: {current_lr:.6f}"
                    self.log.info(log_msg_lr)
                    if self.user_parameters['Verbose'] == 1: print(log_msg_lr)

                    # Print selected stats (batch train loss and global test stats)
                    train_loss_key = 'total_loss_train' # Use the batch train loss stat
                    if train_loss_key in self.learning_stats[iteration]:
                        log_msg = f'{train_loss_key}: {round(self.learning_stats[iteration][train_loss_key], 4)}'
                        self.log.info(log_msg)
                        if self.user_parameters['Verbose'] == 1: print(log_msg)

                    for name, item in avg_test_stats.items():
                        # Include the reinstated correlation stats and pnorm_std_min
                        if 'loss' in name or 'accuracy' in name or 'correlation_' in name or \
                           'total_n_probes' in name or 'pnorm_std' in name or 'median brightness' in name: # Added median brightness
                            log_msg = f'{name}: {round(item, 4) if isinstance(item, (float, int)) and not np.isnan(item) else item}'
                            self.log.info(log_msg)
                            if self.user_parameters['Verbose'] == 1: print(log_msg)
                    self.log.info('------------------')


                # --- Cleanup Old Models ---
                if iteration > 20:
                    keys_to_delete = sorted([k for k in self.saved_models if k < iteration - 20 and k != 0 and k != self.best_iteration])
                    for key_to_del in keys_to_delete:
                        self.saved_models.pop(key_to_del, None)
                        self.saved_optimizer_states.pop(key_to_del, None)

        except Exception as e:
            self.log.exception(f"Error during training loop at iteration {iteration}: {e}")

        finally:
            # --- Load Best Model State (if found during training) ---
            if self.best_model_state_dict is not None:
                self.log.info(f"Loading best model state from iteration {self.best_iteration} (Train Loss: {self.best_loss:.4f}) before final save.")
                try:
                    # Use strict=False when loading best state due to potential structural changes if parameters changed between runs
                    missing_keys, unexpected_keys = self.load_state_dict(self.best_model_state_dict, strict=False)
                    if missing_keys: self.log.warning(f"Missing keys when loading best state_dict: {missing_keys}")
                    if unexpected_keys: self.log.warning(f"Unexpected keys when loading best state_dict: {unexpected_keys}")
                    self.to(current_device)
                    self.log.info(f"Successfully loaded best model state for final saving.")
                except Exception as e:
                    self.log.error(f"Failed to load best model state before saving: {e}. Saving the final iteration state instead.")
            else:
                self.log.warning("No best model state was saved during training. Saving the final iteration state.")

            # --- Save Final Model State (Best or Last) ---
            output_dir = self.user_parameters['output']
            final_model_path = os.path.join(output_dir, 'final_model_state.pt')
            try:
                torch.save(self.state_dict(), final_model_path)
                self.log.info(f"Final model state dictionary saved to {final_model_path}")
            except Exception as e:
                self.log.error(f"Failed to save final model state: {e}")


            # --- Final Evaluation (using the loaded best or final state) ---
            self.eval()
            final_iter_key = 'Final'
            self.learning_stats[final_iter_key] = {}
            # Calculate final stats globally
            with torch.no_grad():
                raw_final_losses_dict, final_stats_dict = self.calculate_loss(
                    self.X_test, self.y_test, self.r_test, iteration=final_iter_key, suffix='_test'
                )

                # This ensures consistency with how the model was last trained or would be used
                # Re-use key_to_weight_param_map for final evaluation weights
                final_eval_effective_weights = {}
                # key_to_weight_param_map should be defined if fit() ran, but as a fallback:
                current_key_to_weight_param_map = {
                    'probe_weight': 'probe_weight',
                    'categorical': 'categorical_weight',
                    'gene_constraint': 'gene_constraint_weight',
                    'p_std': 'pnorm_std_weight',
                    'type_correlation_max': 'type_correlation_max_weight',
                    'hierarchical_scatter': 'hierarchical_scatter_weight',
                    'intra_type_variance': 'intra_type_variance_weight', # NEW
                    'bit_iqr_variance': 'bit_iqr_variance_weight', # NEW
                    'type_entropy': 'type_entropy_weight', # NEW
                }
                for task_name_from_list in self.loss_component_names: # self.loss_component_names should exist
                    weight_param_key = current_key_to_weight_param_map.get(task_name_from_list)
                    if weight_param_key:
                        weight_value = self.user_parameters.get(weight_param_key, 0.0)
                        final_eval_effective_weights[task_name_from_list] = weight_value
                    else:
                        final_eval_effective_weights[task_name_from_list] = 0.0
                
                final_weighted_loss_value = 0.0
                for task_name, raw_loss_comp in raw_final_losses_dict.items():
                    if task_name in self.loss_component_names: # Consider only relevant tasks
                        current_weight = final_eval_effective_weights.get(task_name, 0.0)
                        if raw_loss_comp is not None and current_weight > 0:
                            loss_item = raw_loss_comp.item() if isinstance(raw_loss_comp, torch.Tensor) else raw_loss_comp
                            final_weighted_loss_value += current_weight * loss_item
                
                self.learning_stats[final_iter_key].update(final_stats_dict)
                self.learning_stats[final_iter_key]['total_loss_test_avg'] = final_weighted_loss_value


            # --- Final Reporting ---
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            red_start = "\033[91m"; reset_color = "\033[0m"
            log_prefix = f"--- Final Eval Stats (Global Test Set) at {now_str} ---"
            self.log.info(log_prefix)
            if self.user_parameters['Verbose'] == 1: print(f"{red_start}{log_prefix}{reset_color}")
            for name, item in self.learning_stats[final_iter_key].items():
                # Include the reinstated correlation stats and pnorm_std_min
                if 'loss' in name or 'accuracy' in name or 'correlation_' in name or \
                    'total_n_probes' in name or 'pnorm_std' in name or 'median brightness' in name: # Added median brightness
                    log_msg = f'{name}: {round(item, 4) if isinstance(item, (float, int)) and not np.isnan(item) else item}'
                    self.log.info(log_msg)
                    if self.user_parameters['Verbose'] == 1: print(log_msg)
            self.log.info('------------------')


            self.log.info('Total time taken: {:.2f} seconds'.format(time.time() - start_time))

            # --- Enforce Constraints on Final E (based on the state used for eval: best or last) ---
            # Need to recalculate E based on the potentially reloaded best state
            self.eval() # Ensure eval mode
            with torch.no_grad():
                final_E = self.get_encoding_weights().detach().clone() # Get E from current model state

            self.log.info("Enforcing constraints on the final E matrix...")
            if self.constraints is None:
                self.log.error("Cannot enforce constraints: self.constraints is None.")
                self.E = final_E # Store unconstrained E if constraints missing
            else:
                E_final_constrained = torch.clip(final_E.round(), 0, None)
                T = self.constraints.clone().detach()
                m = E_final_constrained.sum(1) > T
                if m.any():
                    scaling_factors = (T[m] / E_final_constrained.sum(1)[m].clamp(min=1e-8)).unsqueeze(1)
                    E_final_constrained[m, :] = (E_final_constrained[m, :] * scaling_factors).floor()
                    E_final_constrained = E_final_constrained.clamp(min=0)
                self.E = E_final_constrained.clone().detach() # Store final constrained E

                # --- Save Final Constrained E ---
                # output_dir already defined
                e_csv_path = os.path.join(output_dir, 'E_constrained.csv')
                e_pt_path = os.path.join(output_dir, 'E_constrained.pt')
                if self.genes is None:
                    self.log.warning("Gene names not available. Saving E_constrained.csv without index.")
                    pd.DataFrame(self.E.cpu().numpy()).to_csv(e_csv_path)
                else:
                    pd.DataFrame(self.E.cpu().numpy(), index=self.genes).to_csv(e_csv_path)
                torch.save(self.E.cpu(), e_pt_path)
                self.log.info(f"Final constrained E matrix saved to {e_csv_path} and {e_pt_path}")
                self.log.info(f"Final constrained probe count: {self.E.sum().item():.2f}")


            # --- Save Learning Curve ---
            try:
                learning_df = pd.DataFrame.from_dict(self.learning_stats, orient='index')
                learning_curve_path = os.path.join(output_dir, 'learning_curve.csv')
                learning_df.to_csv(learning_curve_path)
                self.log.info(f"Learning curve data saved to {learning_curve_path}")
            except Exception as e:
                self.log.error(f"Failed to save learning curve: {e}")

    def simulate_noise(self, poisson_noise_scale=0, max_cell_type_gene_shifts=0, max_background_scale=0):
        """ Simulates noise on test data and projects train/test data using final E. """
        # Add check for initialization
        if self.X_train is None or self.X_test is None or self.y_test is None or self.E is None:
            self.log.error("Model not initialized. Run initialize() first.")
            raise RuntimeError("Model not initialized. Run initialize() first.")

        X_train = self.X_train.detach()
        X_test = self.X_test.detach()
        y_test = self.y_test.detach()

        X_test_noisy = X_test.clone()
        if poisson_noise_scale > 0:
            X_test_noisy = torch.poisson((X_test_noisy * poisson_noise_scale).clamp(min=0)) / poisson_noise_scale
        if max_cell_type_gene_shifts > 0:
            X_test_shifts = torch.zeros_like(X_test_noisy)
            unique_test_labels = torch.unique(y_test)
            for cell_type_idx in unique_test_labels:
                m = (y_test == cell_type_idx)
                if m.sum() > 0:
                    shift = (1 - 2 * torch.rand_like(X_test_shifts[0, :])) * max_cell_type_gene_shifts
                    X_test_shifts[m, :] = X_test_noisy[m, :] * shift
            X_test_noisy = (X_test_noisy + X_test_shifts).clamp(min=0)

        final_E_device = self.E.to(self.user_parameters['device'])
        P_train = X_train.mm(final_E_device)
        P_test = X_test_noisy.mm(final_E_device)

        P_sum_train = P_train.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P_mean_sum_train = P_sum_train.mean().clamp(min=1e-8)
        P_train = P_train * (P_mean_sum_train / P_sum_train)

        P_sum_test = P_test.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P_mean_sum_test = P_sum_test.mean().clamp(min=1e-8)
        P_test = P_test * (P_mean_sum_test / P_sum_test)

        if max_background_scale > 0:
            background = (torch.rand_like(P_test)) * (10 ** max_background_scale)
            P_test = (P_test + background).clamp(min=0)

        return P_test, P_train

    def evaluate(self):
        """
        Evaluates the trained model using the single decoder structure.
        Calculates accuracy under different noise conditions, potentially
        averaged over regions if use_region_info was True during training.
        """
        # Add check for initialization
        if self.E is None or self.decoder is None or self.region_embedder is None or \
           self.X_train is None or self.X_test is None or self.y_train is None or \
           self.y_test is None or self.r_test is None:
            self.log.error("Cannot evaluate: Model not initialized or trained. Run initialize() and fit() first.")
            return

        self.results = {}
        current_device = self.user_parameters['device']
        use_region_info = bool(self.user_parameters['use_region_info'])

        # --- Basic Stats using Final Constrained E ---
        final_E_cpu = self.E.cpu().detach()
        self.results['Number of Probes (Constrained)'] = final_E_cpu.sum().item()

        # Calculate average P_type across all regions/or globally
        all_P_type = [] # Store P_type calculated on the relevant data subset
        X_train_eval = self.X_train # Use full training set for global P_type if not using regions
        y_train_eval = self.y_train
        r_train_eval = self.r_train # Needed for region embeddings

        # If using region info, calculate P_type per region and average
        # If not, calculate P_type globally once
        eval_indices = self.mapped_region_indices if use_region_info else [0] # Loop once if global

        for r_idx in eval_indices:
            if use_region_info:
                mask = (self.r_train == r_idx)
                X_region = self.X_train[mask]
                y_region = self.y_train[mask]
            else: # Global calculation
                X_region = self.X_train
                y_region = self.y_train

            if X_region.shape[0] > 0:
                with torch.no_grad():
                    P_region, _, _ = self.project(X_region, self.E) # Project relevant data
                    P_region_cpu = P_region.cpu()
                    P_type_current = torch.zeros((self.n_categories, P_region_cpu.shape[1]), device='cpu')
                    unique_y_region = torch.unique(y_region)
                    for type_idx_tensor in unique_y_region:
                        type_idx = type_idx_tensor.item()
                        type_mask = (y_region == type_idx_tensor)
                        if type_mask.sum() > 0 and 0 <= type_idx < self.n_categories:
                            P_type_current[type_idx] = P_region_cpu[type_mask].mean(dim=0)
                    all_P_type.append(P_type_current)

        if all_P_type:
            avg_P_type = torch.stack(all_P_type).mean(dim=0) # Average over regions if multiple, else just use the global one
            self.results['Minimum Signal (Avg P_type)'] = avg_P_type.min().item()
            self.results['Average Signal (Avg P_type)'] = avg_P_type.mean().item()
            self.results['Maximum Signal (Avg P_type)'] = avg_P_type.max().item()
            for bit in range(avg_P_type.shape[1]):
                self.results[f"Number of Probes Bit {bit}"] = final_E_cpu[:, bit].sum().item()
                self.results[f"Minimum Signal Bit {bit} (Avg P_type)"] = avg_P_type[:, bit].min().item()
                self.results[f"Average Signal Bit {bit} (Avg P_type)"] = avg_P_type[:, bit].mean().item()
                self.results[f"Maximum Signal Bit {bit} (Avg P_type)"] = avg_P_type[:, bit].max().item()
        else:
            self.log.warning("Could not calculate average P_type for evaluation stats.")

        # Log basic stats
        self.log.info("--- Basic Evaluation Stats ---")
        for key, val in self.results.items():
            log_msg = f" {key}: {round(val, 4) if isinstance(val, (float, int)) else val}"
            self.log.info(log_msg)
            if self.user_parameters['Verbose'] == 1: print(log_msg)
        self.log.info("-----------------------------")

        # --- Accuracy Evaluation with Noise ---
        noise_levels = {
            "No Noise":    {'poisson': 0, 'shifts': 0,    'background': 0},
            "Low Noise":   {'poisson': 1, 'shifts': 0.25, 'background': 2.5},
            "Medium Noise":{'poisson': 1, 'shifts': 0.5,  'background': 3.0},
            "High Noise":  {'poisson': 1, 'shifts': 1.0,  'background': 3.5}
        }
        self.eval()
        for level_name, params in noise_levels.items():
            self.log.info(f"Calculating {level_name} Accuracy (Averaged over Regions if applicable)")
            total_accuracy = 0.0
            samples_counted = 0 # Count total samples evaluated
            try:
                P_test_noisy, _ = self.simulate_noise(
                    poisson_noise_scale=params['poisson'],
                    max_cell_type_gene_shifts=params['shifts'],
                    max_background_scale=params['background']
                )
                P_test_noisy = P_test_noisy.to(current_device)

                # Evaluate on the full test set (or could batch if needed)
                with torch.no_grad():
                    # Re-apply normalization/transform to the noisy projections
                    P_sum_test = P_test_noisy.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    P_mean_sum_test = P_sum_test.mean().clamp(min=1e-8)
                    P_norm_test = P_test_noisy * (P_mean_sum_test / P_sum_test)
                    # Pnorm_transformed_test = ((P_norm_test.clamp(min=1).log10() - self.user_parameters['target_brightness_log'])).tanh() # Old line
                    input_to_tanh_test = (P_norm_test.clamp(min=1).log10() - self.user_parameters['target_brightness_log'])
                    Pnorm_transformed_test = (self.user_parameters['tanh_slope_factor'] * input_to_tanh_test).tanh() # Modified line

                    # Decode using the single decoder, passing the test region labels
                    # r_test contains the appropriate labels (all 0s if not using regions)
                    y_pred_test, accuracy_test, _ = self.decode(Pnorm_transformed_test, self.r_test, self.y_test)
                    # Accuracy is calculated over the whole test set now
                    avg_accuracy = accuracy_test.item()
                    self.log.info(f" {level_name} Accuracy: {round(avg_accuracy, 4)}")
                    self.results[f'{level_name} Accuracy'] = avg_accuracy # Store global accuracy

            except Exception as e:
                self.log.error(f"Error during {level_name} accuracy calculation: {e}")
                self.results[f'{level_name} Accuracy'] = np.nan

        # --- Save Final Results ---
        results_df = pd.DataFrame(self.results.values(), index=self.results.keys(), columns=['values'])
        results_path = os.path.join(self.user_parameters['output'], 'Results.csv') # Changed filename
        results_df.to_csv(results_path)
        self.log.info(f"Evaluation results saved to {results_path}")

        # Final printout if Verbose
        if self.user_parameters['Verbose'] == 1:
            print("--- Evaluation Summary ---")
            for key, val in self.results.items():
                print(f" {key}: {round(val, 4) if isinstance(val, (float, int)) and not np.isnan(val) else val}")
            print("-------------------------------------------------")

    def visualize(self, show_plots=False): 
        """
        Generates and saves type-by-type correlation heatmaps, P_type heatmaps,
        and projection space density plots for each region based on the
        projections of training data. Optionally displays saved plots if in
        an IPython environment.

        Args:
            show_plots (bool): If True and running in IPython, display plots inline.
                               Defaults to False.
        """
        self.log.info("Starting visualization generation...")
        # Add check for initialization
        # *** WARNING: This original function might be incompatible with the vectorized model structure ***
        # *** It expects self.decoders to be a ModuleDict and loops through regions ***
        # *** Check if self.decoder exists instead of self.decoders ***
        if self.E is None or self.decoder is None or self.region_embedder is None or \
           self.X_train is None or self.y_train is None or self.r_train is None or \
           self.y_reverse_label_map is None or self.r_reverse_label_map is None:
            self.log.error("Cannot visualize: Model not initialized or incompatible structure. Run initialize() and fit() first.")
            return

        current_device = self.user_parameters['device']
        output_dir = self.user_parameters['output']
        saved_plot_paths = [] # Keep track of saved plots
        use_region_info = bool(self.user_parameters['use_region_info']) # Needed for logic below

        # Use the final constrained E matrix on the correct device
        final_E_device = self.E.to(current_device)
        self.eval() # Ensure model is in evaluation mode for projection

        # Loop through each region (or just 'Global' if use_region_info is False)
        for r_idx in self.mapped_region_indices:
            # Get string name and sanitize it for filename
            region_name_str = self.r_reverse_label_map.get(r_idx, f"Region_{r_idx}")
            region_fname_safe = sanitize_filename(region_name_str)
            self.log.info(f"Generating visualization for {region_name_str} (Index: {r_idx}, Filename safe: {region_fname_safe})...")

            # Filter training data for the current region/or use all if global
            region_mask = (self.r_train == r_idx)
            X_region = self.X_train[region_mask]
            y_region = self.y_train[region_mask] # These are mapped indices (0..N-1)

            if X_region.shape[0] == 0:
                self.log.warning(f"Skipping visualization for {region_name_str}: No training data found.")
                continue

            # Calculate P_region and P_type_region for this subset
            with torch.no_grad():
                # Project region data
                P_region_tensor, _, _ = self.project(X_region, final_E_device)
                # P_region_tensor = P_region_tensor.clamp(min=1).log10()
                P_region_np = P_region_tensor.cpu().numpy() # Keep numpy version for plotting

                n_bits = P_region_tensor.shape[1]
                P_type_region = torch.zeros((self.n_categories, n_bits), device=current_device)
                unique_y_indices_in_region = torch.unique(y_region)

                # Keep track of valid type indices and their corresponding string labels for this region
                valid_type_indices = []
                valid_type_labels = []

                for type_idx_tensor in unique_y_indices_in_region:
                    type_idx = type_idx_tensor.item() # Convert tensor to int
                    mask = (y_region == type_idx_tensor)
                    if mask.sum() > 0:
                        if 0 <= type_idx < self.n_categories:
                            P_type_region[type_idx] = P_region_tensor[mask].mean(dim=0) # Use tensor P_region
                            valid_type_indices.append(type_idx)
                            # Get string label using the reverse map
                            valid_type_labels.append(self.y_reverse_label_map.get(type_idx, f"Type_{type_idx}"))
                        else:
                            self.log.warning(f"Skipping type index {type_idx} in region {region_name_str} during P_type calculation (out of bounds).")

                # Filter P_type_region to only include types present in this region
                if not valid_type_indices:
                    self.log.warning(f"Skipping visualization for {region_name_str}: No valid cell types found after projection.")
                    continue

                P_type_region_present = P_type_region[valid_type_indices].cpu() # Move to CPU for plotting/pandas
                n_types_present = P_type_region_present.shape[0]

                # --- Plot 1: Type-Type Correlation Heatmap ---
                if n_types_present > 1:
                    # Calculate type-type correlation matrix
                    P_type_centered = P_type_region_present - P_type_region_present.mean(dim=1, keepdim=True)
                    P_type_std = P_type_centered.std(dim=1, keepdim=True).clamp(min=1e-6)
                    P_type_norm = P_type_centered / P_type_std
                    correlation_matrix = (P_type_norm @ P_type_norm.T / n_bits).numpy() # Already on CPU

                    # Create DataFrame for seaborn using STRING LABELS
                    corr_df = pd.DataFrame(correlation_matrix, index=valid_type_labels, columns=valid_type_labels)

                    # Plotting
                    fig_corr = None # Initialize figure variable
                    try:
                        # Calculate figure size, capped at 25x25
                        fig_width = min(max(8, n_types_present / 1.5), 25)
                        fig_height = min(max(6, n_types_present / 2), 25)
                        fig_corr = plt.figure(figsize=(fig_width, fig_height))
                        ax_corr = fig_corr.add_subplot(111) # Add axes to the figure
                        # Set annot=False, add cbar=True (default, but explicit)
                        sns.heatmap(corr_df, annot=False, cmap='vlag', fmt=".2f", vmin=-1, vmax=1, center=0, linewidths=.5, ax=ax_corr, cbar=True) # Keep vlag for correlation
                        ax_corr.set_title(f"Type Correlation Matrix - {region_name_str}") # Use string name in title
                        plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right')
                        plt.setp(ax_corr.get_yticklabels(), rotation=0)
                        fig_corr.tight_layout()

                        # Save the figure using SANITIZED region name
                        plot_filename = f"type_correlation_heatmap_{region_fname_safe}.png"
                        plot_path = os.path.join(output_dir, plot_filename)
                        fig_corr.savefig(plot_path, dpi=300, bbox_inches='tight')
                        saved_plot_paths.append(plot_path) # Add path to list
                        self.log.info(f"Saved Type Correlation plot for {region_name_str} to {plot_path}")

                    except Exception as e:
                        self.log.error(f"Error generating Type Correlation heatmap for {region_name_str}: {e}")
                    finally:
                        if fig_corr is not None:
                            plt.close(fig_corr) # Close the figure

                else:
                    self.log.warning(f"Skipping correlation plot for {region_name_str}: Only {n_types_present} cell type(s) present.")

                # --- Plot 2: P_type (Average Projection) Clustermap ---
                if n_types_present > 0:
                    # Create DataFrame for P_type using STRING LABELS
                    p_type_df = pd.DataFrame(P_type_region_present.clamp(min=1).log10().numpy(), # Already on CPU
                                             index=valid_type_labels,
                                             columns=[f"Bit_{b}" for b in range(n_bits)])

                    # Plotting using clustermap
                    cluster_fig = None # Initialize figure variable
                    try:
                        # Calculate figure size, capped at 25x25
                        fig_width = min(max(6, n_bits / 1.5), 25)
                        fig_height = min(max(6, n_types_present / 2), 25)
                        # clustermap handles figure creation
                        # Remove cbar_pos=None to show the colorbar
                        cluster_fig = sns.clustermap(p_type_df,
                                                     cmap="inferno", # Use inferno colormap
                                                     figsize=(fig_width, fig_height),
                                                     linewidths=0.1,
                                                     dendrogram_ratio=(.2, .2) # Adjust dendrogram sizes
                                                     # cbar_pos=None # Removed to show colorbar
                                                     )
                        cluster_fig.fig.suptitle(f"Average Projection (P_type) - {region_name_str}", y=1.02) # Use string name in title
                        cluster_fig.ax_heatmap.set_xlabel("Projection Bit")
                        cluster_fig.ax_heatmap.set_ylabel("Cell Type (Clustered)")
                        plt.setp(cluster_fig.ax_heatmap.get_xticklabels(), rotation=90) # Rotate x-axis labels if needed
                        plt.setp(cluster_fig.ax_heatmap.get_yticklabels(), rotation=0) # Ensure y-axis labels are horizontal

                        # Save the figure using SANITIZED region name
                        plot_filename = f"P_type_clustermap_{region_fname_safe}.png"
                        plot_path = os.path.join(output_dir, plot_filename)
                        cluster_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        saved_plot_paths.append(plot_path) # Add path to list
                        self.log.info(f"Saved P_type clustermap for {region_name_str} to {plot_path}")

                    except Exception as e:
                        self.log.error(f"Error generating P_type clustermap for {region_name_str}: {e}")
                    finally:
                        # Clustermap creates its own figure, close it
                        if cluster_fig is not None:
                            plt.close(cluster_fig.fig)

                else:
                    # This case should have been caught earlier, but added for safety
                    self.log.warning(f"Skipping P_type plot for {region_name_str}: No cell types present.")

                # --- Plot 3: Call the standalone projection space density plot ---
                if n_types_present > 0 and n_bits >= 2:
                    # Use SANITIZED region name for filename
                    plot_filename = f"projection_density_plot_{region_fname_safe}.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    try:
                        # *** Pass y_region_str_labels to the original plotting function ***
                        y_region_str_labels = np.array([self.y_reverse_label_map.get(idx.item(), f"Type_{idx.item()}") for idx in y_region])
                        plot_projection_space_density(
                            # Use unnormalized P for density plot
                            X_region.cpu().numpy() @ self.E.cpu().numpy(), # Re-project on CPU if needed
                            y_region_str_labels, # Pass string labels
                            plot_path,
                            sum_norm=True, # Let the function handle normalization
                            log=True
                            )
                        saved_plot_paths.append(plot_path) # Add path to list
                    except Exception as e:
                        # Use str(e) for explicit conversion in log message
                        self.log.error(f"Error generating projection space density plot for {region_name_str}: {str(e)}")
                elif n_bits < 2:
                    self.log.warning(f"Skipping projection space density plot for {region_name_str}: Requires at least 2 bits (found {n_bits}).")
                # else: (n_types_present == 0) - Already handled above

                # --- Plot 4: Confusion Matrix for Test Set ---
                self.log.info(f"Generating confusion matrix for test data in {region_name_str}...")
                # Filter test data for the current region
                test_region_mask = (self.r_test == r_idx)
                X_test_region = self.X_test[test_region_mask]
                y_test_region = self.y_test[test_region_mask] # True internal labels
                r_test_region = self.r_test[test_region_mask] # Region labels for decode

                if X_test_region.shape[0] > 0:
                    fig_cm = None # Initialize figure variable
                    try:
                        with torch.no_grad():
                            # Project test data
                            P_test_tensor, Pnorm_test_tensor, Pnorm_dropout_test_tensor = self.project(X_test_region, final_E_device)
                            # Decode test data using the single decoder
                            y_pred_test, _, _ = self.decode(Pnorm_dropout_test_tensor, r_test_region, y_test_region)

                        # Move labels to CPU for sklearn/plotting
                        y_true_np = y_test_region.cpu().numpy()
                        y_pred_np = y_pred_test.cpu().numpy()

                        # *** MODIFIED: Get labels present in this region's test set ***
                        present_labels_idx = np.unique(np.concatenate((y_true_np, y_pred_np)))
                        present_labels_str = [self.y_reverse_label_map.get(i, f"Type_{i}") for i in present_labels_idx]

                        if len(present_labels_idx) > 0: # Only plot if there are labels
                            # Generate confusion matrix using only present labels
                            cm = confusion_matrix(y_true_np, y_pred_np, labels=present_labels_idx) # Use present internal indices

                            # Normalize row-wise
                            cm_sum = cm.sum(axis=1, keepdims=True)
                            cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) # Avoid division by zero

                            # Create DataFrame with string labels of present types
                            cm_df = pd.DataFrame(cm_norm, index=present_labels_str, columns=present_labels_str)

                            # Plotting
                            fig_width = min(max(10, len(present_labels_str) / 1.5), 25)
                            fig_height = min(max(8, len(present_labels_str) / 2), 25)
                            fig_cm = plt.figure(figsize=(fig_width, fig_height))
                            ax_cm = fig_cm.add_subplot(111)
                            # *** EDITED: Set linewidths=0 ***
                            sns.heatmap(cm_df, annot=False, cmap='jet', linewidths=0.1, ax=ax_cm, vmin=0, vmax=1) # Normalize range 0-1
                            ax_cm.set_xlabel('Predicted Label')
                            ax_cm.set_ylabel('True Label')
                            ax_cm.set_title(f'Confusion Matrix (Test Set) - {region_name_str}')
                            plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right')
                            plt.setp(ax_cm.get_yticklabels(), rotation=0)
                            fig_cm.tight_layout()

                            # Save the figure
                            plot_filename = f"confusion_matrix_test_{region_fname_safe}.png"
                            plot_path = os.path.join(output_dir, plot_filename)
                            fig_cm.savefig(plot_path, dpi=300, bbox_inches='tight')
                            saved_plot_paths.append(plot_path)
                            self.log.info(f"Saved Test Confusion Matrix for {region_name_str} to {plot_path}")
                        else:
                            self.log.warning(f"Skipping confusion matrix for {region_name_str}: No labels found in true or predicted test data.")

                    except Exception as e:
                        self.log.error(f"Error generating Test Confusion Matrix for {region_name_str}: {e}")
                    finally:
                        if fig_cm is not None:
                            plt.close(fig_cm) # Close the figure
                else:
                    self.log.warning(f"Skipping confusion matrix for {region_name_str}: No test data found.")
                # --- *** END ADDED PLOT 4 *** ---

        self.log.info("Visualization generation finished.")

        # --- Optional: Display saved plots if requested and in IPython ---
        if show_plots:
            if _IPYTHON_AVAILABLE and get_ipython() is not None:
                self.log.info("Displaying saved plots inline (IPython detected)...")
                if not saved_plot_paths:
                    self.log.info("No plots were generated or saved.")
                for plot_path in saved_plot_paths:
                    if os.path.exists(plot_path):
                        self.log.info(f"Displaying: {os.path.basename(plot_path)}")
                        display(Image(filename=plot_path))
                    else:
                        self.log.error(f"Plot file not found: {plot_path}")
            else:
                self.log.warning("\nPlots saved to output directory. Run in an IPython environment (like Jupyter) and set show_plots=True to display inline.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("user_parameters_path", type=str, help="Path to csv containing parameters for model")
    args = parser.parse_args()
    user_parameters_path = args.user_parameters_path
    
    model = EncodingDesigner(user_parameters_path=user_parameters_path)
    if not model.initialize():
        print("Initialization failed. Check the log file for details.")
        exit(1)
    model.fit()
    model.evaluate()
    model.visualize(show_plots=False)