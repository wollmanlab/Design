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


class EncodingDesigner(nn.Module):
    def __init__(self, user_parameters_path=None):
        super().__init__() 

        self.user_parameters = {
            'n_cpu': 12,
            'n_bit': 24,
            'n_iterations': 10000,
            'batch_size': 1000,
            'target_brightness_log': 4.5,
            'total_n_probes': 30e4,
            'probe_weight': 1.0,
            'probe_under_weight_factor': 0.1,
            'gene_constraint_weight': 1.0,
            'target_brightness_weight':1.0,
            'gradient_clip_max_norm': 1.0, # Added for gradient clipping
            'learning_rate_start': 0.05, 
            'learning_rate_end': 0.005, 
            'report_freq': 250,
            'sparsity_target': 0.8, # Target sparsity ratio (80% zeros)
            'sparsity_weight': 0.0, # Weight for sparsity loss (increased from 0.1)
            'categorical_weight': 1.0,
            'weight_dropout_proportion_start': 0.0,
            'weight_dropout_proportion_end': 0.1,
            'projection_dropout_proportion_start': 0.0,
            'projection_dropout_proportion_end': 0.1,
            'gene_dropout_proportion_start': 0.0,
            'gene_dropout_proportion_end': 0.1,
            'decoder_dropout_rate_start': 0.1,
            'decoder_dropout_rate_end': 0.0,
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

        temp_output_dir = self.user_parameters['output']
        loaded_params_temp = {}
        if user_parameters_path is not None:
            try:
                df_temp = pd.read_csv(user_parameters_path, index_col=0, low_memory=False)
                if 'values' not in df_temp.columns:
                    logging.basicConfig(level=logging.ERROR)
                    logging.error(f"Parameter file {user_parameters_path} missing 'values' column.")
                else:
                    loaded_params_temp = dict(zip(df_temp.index, df_temp['values']))
                    temp_output_dir = loaded_params_temp.get('output', temp_output_dir) 
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
        logging.basicConfig(level=logging.INFO) 
        
        input_filename = os.path.basename(user_parameters_path) if user_parameters_path else "default"
        input_filename = os.path.splitext(input_filename)[0]  
        self.log_file = os.path.join(temp_output_dir, f'log_{input_filename}.log')
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        logging.basicConfig(
            filename=self.log_file, filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%Y %B %d %H:%M:%S', level=logging.INFO, force=True)
        self.log = logging.getLogger("Designer")
        self.results = {}

        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1

        loaded_user_parameters = {}
        if user_parameters_path is not None:
            self.log.info(f"Loading user parameters from: {user_parameters_path}")
            try:
                user_parameters_df = pd.read_csv(user_parameters_path, index_col=0, low_memory=False)
                if 'values' not in user_parameters_df.columns:
                    self.log.error(f"Parameter file {user_parameters_path} missing 'values' column. Sticking to defaults.")
                else:
                    loaded_user_parameters = dict(zip(user_parameters_df.index, user_parameters_df['values']))
                    self.log.info(f"Successfully loaded {len(loaded_user_parameters)} parameters from file.")

                    for key, val in loaded_user_parameters.items():
                        try:
                            float_val = float(val)
                            if float_val.is_integer():
                                loaded_user_parameters[key] = int(float_val)
                            else:
                                loaded_user_parameters[key] = float_val
                        except (ValueError, TypeError):
                            self.log.debug(f"Could not convert parameter '{key}' value '{val}' to float. Keeping as string.")
                            continue 
                    self.log.info("Attempted conversion of loaded parameters to numeric types.")

            except FileNotFoundError:
                self.log.error(f"Parameter file not found at: {user_parameters_path}. Using default parameters.")
                loaded_user_parameters = {} 
            except Exception as e:
                self.log.error(f"Error processing parameter file {user_parameters_path}: {e}. Using default parameters.")
                loaded_user_parameters = {} 
        else:
            self.log.info("No user parameter file provided. Using default parameters.")

        for key, val in loaded_user_parameters.items():
            if key in self.user_parameters:
                self.user_parameters[key] = val 
            else:
                self.log.warning(f"Parameter '{key}' from file is not a default parameter. Adding it.")
                self.user_parameters[key] = val 
            if '_start' in key:
                new_key = key.replace('_start', '')
                if new_key not in self.user_parameters:
                    self.user_parameters[new_key] = val
                    self.log.info(f"Created parameter '{new_key}' from '{key}' with value {val}")
                else:
                    self.log.info(f"Parameter '{new_key}' already exists. Skipping creation.")

        input_dir = self.user_parameters['input']
        file_params_to_prefix = [
            'constraints', 'X_test', 'y_test', 
            'X_train', 'y_train', 
            'y_label_converter_path' 
        ]
        for param_key in file_params_to_prefix:
            current_path = self.user_parameters[param_key]
            if current_path and not os.path.dirname(current_path) and not os.path.isabs(current_path):
                self.user_parameters[param_key] = os.path.join(input_dir, current_path)
                self.log.info(f"Constructed path for '{param_key}': {self.user_parameters[param_key]}")

        params_to_int = ['n_bit', 'n_iterations', 'report_freq', 'batch_size', 'n_cpu',
                         'decoder_hidden_layers', 'decoder_hidden_dim'] 
        for param_key in params_to_int:
            self._convert_param_to_int(param_key) 

        self.log.info(f"Final Parameters (after path construction & type conversion):")
        for key, val in self.user_parameters.items():
            self.log.info(f"{key}: {val} (type: {type(val).__name__})") 

        self.log.info(f"Limiting Torch to {self.user_parameters['n_cpu']} threads")
        torch.set_num_threads(self.user_parameters['n_cpu'])

        output_dir = self.user_parameters['output'] 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.log.info(f"Created output directory: {output_dir}")
        pd.DataFrame(self.user_parameters.values(), index=self.user_parameters.keys(), columns=['values']).to_csv(os.path.join(output_dir, 'used_user_parameters.csv'))

        self.log.info("Creating symlinks for input files in output directory...")
        input_param_keys = [
            'constraints', 'X_test', 'y_test', 
            'X_train', 'y_train', 
            'y_label_converter_path'
        ]
        input_files_to_link = []
        for key in input_param_keys:
            path = self.user_parameters.get(key)
            if isinstance(path, str) and path: 
                input_files_to_link.append(path)

        if user_parameters_path is not None and isinstance(user_parameters_path, str):
            input_files_to_link.append(user_parameters_path)
            input_files_to_link.append(os.path.join(output_dir, 'used_user_parameters.csv'))

        linked_count = 0
        skipped_count = 0
        error_count = 0
        for input_path in set(input_files_to_link): 
            try:
                abs_input_path = os.path.abspath(input_path)
                if not os.path.exists(abs_input_path):
                    self.log.warning(f"Input file not found, cannot create symlink: {abs_input_path}")
                    skipped_count += 1
                    continue
                filename = os.path.basename(abs_input_path)
                symlink_path = os.path.join(output_dir, filename)
                if os.path.lexists(symlink_path): 
                    if os.path.islink(symlink_path):
                        self.log.debug(f"Removing existing symlink: {symlink_path}")
                        os.remove(symlink_path)
                    else:
                        self.log.warning(f"Target path exists but is not a symlink, skipping: {symlink_path}")
                        skipped_count += 1
                        continue
                os.symlink(abs_input_path, symlink_path)
                self.log.info(f"Created symlink: {symlink_path} -> {abs_input_path}")
                linked_count += 1
            except OSError as e:
                self.log.error(f"Failed to create symlink for {input_path} -> {symlink_path}: {e}")
                error_count += 1
            except Exception as e:
                self.log.error(f"An unexpected error occurred while trying to symlink {input_path}: {e}")
                error_count += 1
        self.log.info(f"Symlinking complete. Created: {linked_count}, Skipped: {skipped_count}, Errors: {error_count}")

        self.E = None
        self.P = None
        self.Pnormalized = None
        self.genes = None
        self.constraints = None
        self.encoder = None
        self.decoder = None  
        self.optimizer_gen = None
        self.learning_stats = {}
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.is_initialized_from_file = False 
        self.type_cooccurrence_mask = None  
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.n_genes = None 
        self.n_categories = None
        self.y_label_map = None
        self.y_reverse_label_map = None
        self.y_unique_labels = None

    def _convert_param_to_int(self, param_key):
        try:
            original_value = self.user_parameters[param_key]
            float_value = float(original_value)
            if float_value.is_integer():
                self.user_parameters[param_key] = int(float_value)
            else:
                raise ValueError(f"Value '{original_value}' is not a whole number.")
        except KeyError:
            self.log.error(f"Parameter '{param_key}' not found for integer conversion.")
            raise KeyError(f"Required parameter '{param_key}' is missing.")
        except (ValueError, TypeError) as e:
            self.log.error(f"Error converting parameter '{param_key}' to int. Value was '{original_value}' (type: {type(original_value).__name__}). Error: {e}")
            raise ValueError(f"Could not convert parameter '{param_key}' to integer. Invalid value: '{original_value}'.")

    def initialize(self):
        self.log.info("--- Starting Initialization ---")
        current_device = self.user_parameters['device']
        output_dir = self.user_parameters['output']
        model_state_path = os.path.join(output_dir, 'final_model_state.pt')

        try:
            self.log.info("Loading Gene Constraints")
            constraints_path = self.user_parameters['constraints']
            constraints_df = pd.read_csv(constraints_path, index_col=0)
            self.genes = np.array(constraints_df.index)
            if 'constraints' not in constraints_df.columns:
                raise KeyError(f"Column 'constraints' not found in {constraints_path}")
            self.constraints = torch.tensor(constraints_df['constraints'].values, dtype=torch.float32, device=current_device)
            self.n_genes = len(self.genes) 
            self.log.info(f"Loaded {self.n_genes} genes from constraints.")

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
            y_converter_df = pd.read_csv(y_converter_path, index_col=0) 
            y_converter_dict = dict(zip(y_converter_df.index,y_converter_df['label'])) 
            self.y_label_map = {k:self.updated_y_label_map[j] for k,j in y_converter_dict.items()}
            self.y_reverse_label_map = {j:k for k,j in self.y_label_map.items()}
            
            all_y_labels_for_n_categories = torch.cat((self.y_train, self.y_test)) 
            unique_y_labels_tensor = torch.unique(all_y_labels_for_n_categories)
            self.n_categories = len(unique_y_labels_tensor)
            self.mapped_category_indices = list(range(self.n_categories)) 

            if self.X_train.shape[1] != self.n_genes:
                raise ValueError(f"X_train gene dimension mismatch")
            if self.X_test.shape[1] != self.n_genes:
                raise ValueError(f"X_test gene dimension mismatch")
            if not (self.X_train.shape[0] == self.y_train.shape[0]):
                raise ValueError(f"Training data shape mismatch (X_train, y_train)")
            if not (self.X_test.shape[0] == self.y_test.shape[0]): 
                raise ValueError(f"Testing data shape mismatch (X_test, y_test)")
            self.log.info("Data loaded and shapes validated.")
            self.log.info(f"Inferred {self.n_categories} cell type categories.")

            # --- Initialize Model Components Structurally (ONCE) ---
            # Initialize encoder with proper scaling
            self.encoder = nn.Embedding(self.n_genes, self.user_parameters['n_bit']).to(current_device)
            
            # Initialize encoder weights for sigmoid-based fractions
            with torch.no_grad():
                # Initialize to larger random values for sigmoid
                # This gives sigmoid values roughly in [0.1, 0.9] range
                self.encoder.weight.data = torch.randn_like(self.encoder.weight.data) * 2.0
                
                self.log.info(f"Initialized encoder weights for sigmoid-based fractions")
                self.log.info(f"Initial sigmoid range: {torch.sigmoid(self.encoder.weight).min().item():.3f} to {torch.sigmoid(self.encoder.weight).max().item():.3f}")


            n_hidden_layers_decoder = self.user_parameters['decoder_hidden_layers']
            hidden_dim_decoder = self.user_parameters['decoder_hidden_dim']
            dropout_rate_decoder = self.user_parameters['decoder_dropout_rate']

            decoder_modules = []
            current_decoder_layer_input_dim = self.user_parameters['n_bit']

            if n_hidden_layers_decoder == 0:
                decoder_modules.append(nn.Linear(current_decoder_layer_input_dim, self.n_categories))
                log_msg_decoder_structure = "Initialized single linear decoder."
            else:
                for i in range(n_hidden_layers_decoder):
                    decoder_modules.append(nn.Linear(current_decoder_layer_input_dim, hidden_dim_decoder))
                    decoder_modules.append(nn.BatchNorm1d(hidden_dim_decoder)) 
                    decoder_modules.append(nn.ReLU())
                    decoder_modules.append(nn.Dropout(p=dropout_rate_decoder))
                    current_decoder_layer_input_dim = hidden_dim_decoder 
                decoder_modules.append(nn.Linear(current_decoder_layer_input_dim, self.n_categories))
                log_msg_decoder_structure = f"Initialized decoder with {n_hidden_layers_decoder} hidden layer(s) (dim={hidden_dim_decoder}, dropout={dropout_rate_decoder}) and output layer."
            
            self.decoder = nn.Sequential(*decoder_modules).to(current_device)
            self.log.info(f"Initialized encoder.") 
            self.log.info(log_msg_decoder_structure)

            self.log.info("Calculating type co-occurrence mask...")
            self.type_cooccurrence_mask = ~torch.eye(self.n_categories, dtype=torch.bool, device=current_device)
            self.log.info("All type pairs (excluding self-correlation) considered co-occurring for correlation loss.")
            
            if os.path.exists(model_state_path):
                self.log.info(f"Found existing model state file: {model_state_path}. Attempting to load.")
                try:
                    loaded_state_dict = torch.load(model_state_path, map_location=current_device)
                    missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=False)
                    if missing_keys: self.log.warning(f"Missing keys when loading state_dict: {missing_keys}")
                    if unexpected_keys: self.log.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
                    self.to(current_device) 
                    self.is_initialized_from_file = True
                    self.log.info("Successfully loaded model state from file (strict=False).")
                    self.eval() 
                    with torch.no_grad():
                        final_E = self.get_encoding_weights().detach().clone()
                        self.log.info("Enforcing constraints on loaded E matrix...")
                        if self.constraints is None:
                            self.log.error("Cannot enforce constraints: self.constraints is None.")
                            self.E = final_E 
                        else:
                            E_final_constrained = torch.clip(final_E.round(), 0, None)
                            T = self.constraints.clone().detach()
                            m = E_final_constrained.sum(1) > T
                            if m.any():
                                scaling_factors = (T[m] / E_final_constrained.sum(1)[m].clamp(min=1e-8)).unsqueeze(1)
                                E_final_constrained[m, :] = (E_final_constrained[m, :] * scaling_factors).floor()
                                E_final_constrained = E_final_constrained.clamp(min=0)
                            self.E = E_final_constrained.clone().detach() 
                            self.log.info(f"Stored constrained E matrix from loaded model. Probe count: {self.E.sum().item():.2f}")
                except Exception as e:
                    self.log.error(f"Failed to load model state from {model_state_path}: {e}. Model will use fresh initial weights.")
                    self.is_initialized_from_file = False
                    self.E = None 
            else:
                self.log.info(f"No existing model state file found at {model_state_path}. Model will use fresh initial weights.")
                self.is_initialized_from_file = False
            
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
        if self.encoder is None:
            raise RuntimeError("Encoder not initialized. Call initialize() or fit() first.")
        
        # Get fractions (0 to 1) using sigmoid
        fractions = torch.sigmoid(self.encoder.weight)
        
        # Convert to actual probe counts using gene constraints
        E = fractions * self.constraints.unsqueeze(1)
        
        # Apply dropout if training - set percentage of E values to 0
        if self.training and self.user_parameters['weight_dropout_proportion'] > 0:
            dropout_mask_E = (torch.rand_like(E) > self.user_parameters['weight_dropout_proportion']).float()
            E = E * dropout_mask_E
        
        return E

    def perturb_weights(self):
        """
        Randomly perturb encoder weights to random fractions (0 to 1).
        """
        with torch.no_grad():
            # Get current weights
            current_weights = self.encoder.weight.data.clone()
            
            # Calculate number of weights to perturb
            total_weights = current_weights.numel()
            num_to_perturb = int(total_weights * self.user_parameters['perturbation_percentage'])
            
            # Randomly select weights to perturb
            flat_indices = torch.randperm(total_weights)[:num_to_perturb]
            row_indices = flat_indices // current_weights.shape[1]
            col_indices = flat_indices % current_weights.shape[1]
            
            # Generate random fractions (0 to 1) and convert to sigmoid input values
            random_fractions = torch.rand(num_to_perturb)
            # Use inverse sigmoid (logit) to get the raw weight values
            perturbation_values = torch.logit(random_fractions.clamp(min=1e-7, max=1-1e-7))
            
            # Apply the perturbation
            current_weights[row_indices, col_indices] = perturbation_values
            self.encoder.weight.data = current_weights
            
            if self.user_parameters.get('Verbose', 0) == 1:
                print(f"Perturbed {num_to_perturb} weights ({self.user_parameters['perturbation_percentage']*100:.1f}%) "
                      f"to random fractions (range: {random_fractions.min().item():.3f} to {random_fractions.max().item():.3f})")
            
            self.log.info(f"Perturbed {num_to_perturb} weights ({self.user_parameters['perturbation_percentage']*100:.1f}%) "
                         f"to random fractions (range: {random_fractions.min().item():.3f} to {random_fractions.max().item():.3f})")

    def project(self, X, E):
        if self.user_parameters['gene_fold_noise'] != 0:
            fold = self.user_parameters['gene_fold_noise']
            gene_noise = torch.exp(torch.rand_like(X) * 2 * torch.log(torch.tensor(fold)) - torch.log(torch.tensor(fold)))
            X = X * gene_noise

        if self.user_parameters['gene_dropout_proportion'] != 0:
            dropout_mask_X = (torch.rand_like(X) > self.user_parameters['gene_dropout_proportion']).float()
            X = X * dropout_mask_X
            
        P = X.mm(E)
        if self.user_parameters['constant_noise'] != 0:
            noise = torch.rand_like(P) * (10 ** self.user_parameters['constant_noise'])
            P = torch.clip(P + noise, min=1.0)
        Pnormalized = P.clamp(min=1).log10() - self.user_parameters['target_brightness_log']
        if self.training and self.user_parameters['projection_dropout_proportion'] > 0:
            dropout_mask_P = (torch.rand_like(Pnormalized) > self.user_parameters['projection_dropout_proportion']).float()
            Pnormalized_dropout = Pnormalized * dropout_mask_P
        else:
            Pnormalized_dropout = Pnormalized
        return P, Pnormalized, Pnormalized_dropout

    def decode(self, Pnormalized_dropout, y):
        if self.decoder is None :
            raise RuntimeError("Decoder not initialized.")
        if not isinstance(self.decoder, nn.Module):
            raise ValueError("Invalid decoder module.")

        decoder_input = Pnormalized_dropout 
        R = self.decoder(decoder_input) 

        y_predict = R.max(1)[1]
        accuracy = (y_predict == y).float().mean()

        if self.user_parameters['categorical_weight'] != 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            categorical_loss = loss_fn(R, y)
        else:
            categorical_loss = torch.tensor(0.0, device=R.device, requires_grad=True)
        return y_predict, accuracy, categorical_loss

    def calculate_loss(self, X, y, iteration, suffix=''):
        E = self.get_encoding_weights()
        P_original, Pnormalized, Pnormalized_dropout = self.project(X, E) 
        y_predict, accuracy, raw_categorical_loss_component = self.decode(Pnormalized_dropout, y)
        raw_losses = {}
        current_stats = {}
        current_stats['accuracy' + suffix] = accuracy.item()
        current_stats['median brightness' + suffix] = P_original.median().item()
        
        # Calculate dynamic range utilization and median brightness for each bit
        bit_dynamic_ranges = []
        bit_percentiles = []
        bit_medians = []
        for bit_idx in range(P_original.shape[1]):
            bit_values = P_original[:, bit_idx]
            p10, p50, p90 = torch.quantile(bit_values, torch.tensor([0.1, 0.5, 0.9]))
            fold_change = p90 / p10.clamp(min=1e-8)  # Avoid division by zero
            bit_dynamic_ranges.append(fold_change.item())
            bit_percentiles.append((p10.item(), p50.item(), p90.item()))
            bit_medians.append(p50.item())
        current_stats['lowest bit median brightness' + suffix] = f"{np.log10(min(bit_medians)):.2f}"
        current_stats['highest bit median brightness' + suffix] = f"{np.log10(max(bit_medians)):.2f}"
        min_range_idx = bit_dynamic_ranges.index(min(bit_dynamic_ranges))
        max_range_idx = bit_dynamic_ranges.index(max(bit_dynamic_ranges))
        min_p10, min_p50, min_p90 = bit_percentiles[min_range_idx]
        max_p10, max_p50, max_p90 = bit_percentiles[max_range_idx]
        min_fold_change = bit_dynamic_ranges[min_range_idx]
        max_fold_change = bit_dynamic_ranges[max_range_idx]
        current_stats['lowest_dynamic_range_bit' + suffix] = f"p10:{np.log10(min_p10):.2f}, p50:{np.log10(min_p50):.2f}, p90:{np.log10(min_p90):.2f}, fold:{min_fold_change:.2f}"
        current_stats['highest_dynamic_range_bit' + suffix] = f"p10:{np.log10(max_p10):.2f}, p50:{np.log10(max_p50):.2f}, p90:{np.log10(max_p90):.2f}, fold:{max_fold_change:.2f}"

        # The model should not use more probes than self.user_parameters['total_n_probes']
        probe_count = E.sum()
        current_stats['total_n_probes' + suffix] = probe_count.item()
        current_stats['total_n_genes' + suffix] = (E > 1).any(1).sum().item()
        current_stats['median_probe_weight' + suffix] = E[E > 1].median().item() if (E > 1).any() else 0
        if self.user_parameters['probe_weight']!=0:
            fold = (probe_count/self.user_parameters['total_n_probes']) - 1
            probe_weight_loss = self.user_parameters['probe_weight'] * (F.relu(fold) + self.user_parameters['probe_under_weight_factor'] * F.relu(-fold))
            raw_losses['probe_weight'] = probe_weight_loss
            current_stats['probe_weight_loss' + suffix] = probe_weight_loss.item()

        # The model should accurately decode cell type labels
        if self.user_parameters['categorical_weight'] != 0:
            categorical_loss_component = self.user_parameters['categorical_weight'] * raw_categorical_loss_component
            raw_losses['categorical'] = categorical_loss_component
            current_stats['categorical_loss' + suffix] = categorical_loss_component.item()

        # The model should not use more probes than a gene can supply
        if self.user_parameters['gene_constraint_weight'] != 0:
            fold = (E.sum(dim=1) / self.constraints) - 1
            gene_constraint_loss = self.user_parameters['gene_constraint_weight']*F.relu(fold).mean()
            raw_losses['gene_constraint_loss'] = gene_constraint_loss
            current_stats['gene_constraint_loss' + suffix] = gene_constraint_loss.item()

        # The model should have a median brightness atleast to the target brightness
        if self.user_parameters['target_brightness_weight'] != 0:
            fold = self.user_parameters['target_brightness_log']/P_original.mean(0).min().clamp(min=1).log10()
            brightness_loss = self.user_parameters['target_brightness_weight']* F.relu(fold)
            raw_losses['brightness_loss'] = brightness_loss
            current_stats['brightness_loss' + suffix] = brightness_loss.item()

        # Target sparsity loss to encourage specific percentage of zeros
        if self.user_parameters['sparsity_weight'] != 0:
            sparsity_threshold = 0.01
            sparsity_ratio = (E < sparsity_threshold).float().mean()
            target_sparsity = self.user_parameters['sparsity_target']
            difference = F.relu(target_sparsity - sparsity_ratio)
            sparsity_loss = self.user_parameters['sparsity_weight'] * difference
            raw_losses['sparsity'] = sparsity_loss
            current_stats['sparsity_loss' + suffix] = sparsity_loss.item()
            current_stats['current_sparsity_ratio' + suffix] = sparsity_ratio.item()

        total_loss = sum(raw_losses.values())
        
        return total_loss, current_stats

    def fit(self):
        if self.X_train is None or self.y_train is None or self.constraints is None or self.decoder is None : 
            self.log.error("Model is not initialized. Call initialize() before fit().")
            raise RuntimeError("Model is not initialized. Call initialize() before fit().")

        self.learning_stats = {} 
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1
        start_time = time.time()
        current_device = self.user_parameters['device']

        last_report_time = start_time
        last_report_iteration = 0
        n_iterations = self.user_parameters['n_iterations']
        report_freq = self.user_parameters['report_freq']
        batch_size = self.user_parameters['batch_size']
        n_train_samples = self.X_train.shape[0]

        try:
            for iteration in range(n_iterations):
                progress = iteration / (n_iterations - 1) if n_iterations > 1 else 0
                parameters_to_update = [i.replace('_start', '') for i in self.user_parameters if '_start' in i]
                for param in parameters_to_update:
                    self.user_parameters[param] = (self.user_parameters[f'{param}_start'] + (self.user_parameters[f'{param}_end'] - self.user_parameters[f'{param}_start']) * progress)
                self.learning_stats[iteration] = {}

                if iteration == 0:
                    if not self.is_initialized_from_file:
                        self.log.info("Model not initialized from file, using randomly initialized weights.")
                    else:
                        self.log.info("Using model loaded during initialization.")
                    self.to(current_device)

                    optimizer_gen = torch.optim.Adam([
                        {'params': self.encoder.parameters(), 'lr': self.user_parameters['learning_rate']},
                        {'params': self.decoder.parameters(), 'lr': self.user_parameters['learning_rate']}
                    ])
                    self.optimizer_gen = optimizer_gen

                for param_group in self.optimizer_gen.param_groups: param_group['lr'] = self.user_parameters['learning_rate']

                is_report_iter = (iteration % report_freq == 0) or (iteration == n_iterations - 1) 

                self.train() 

                if (batch_size > 0) and (batch_size < n_train_samples):
                    idxs = np.random.choice(n_train_samples, batch_size, replace=False)
                    X_batch = self.X_train[idxs]
                    y_batch = self.y_train[idxs]
                else:  
                    X_batch = self.X_train
                    y_batch = self.y_train
                    if batch_size > 0:  
                        self.log.debug(f"Batch size {batch_size} >= dataset size {n_train_samples}. Using full dataset for iteration {iteration}.")

                self.optimizer_gen.zero_grad() 
                
                total_loss, batch_stats = self.calculate_loss(
                    X_batch, y_batch, iteration, suffix='_train'
                )
                self.learning_stats[iteration].update(batch_stats)
                self.learning_stats[iteration]['total_loss_train'] = total_loss.item()
                total_loss.backward() 

                nan_detected = False
                for name, param in self.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        nan_detected = True
                        self.log.warning(f"NaNs or Infs detected in gradients of model parameter '{name}' at iteration {iteration}. Skipping step and attempting revert.")
                        self.optimizer_gen.zero_grad() 
                        break
                
                if not nan_detected:
                    # --- GRADIENT CLIPPING ---
                    max_norm_value = self.user_parameters.get('gradient_clip_max_norm', 1.0) 
                    if max_norm_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm_value)
                    # --- END GRADIENT CLIPPING ---
                    self.optimizer_gen.step()  
                    
                    # --- WEIGHT PERTURBATION ---
                    if self.user_parameters['perturbation_frequency'] > 0:
                        if iteration % self.user_parameters['perturbation_frequency'] == 0:
                            self.perturb_weights()
                    # --- END WEIGHT PERTURBATION ---
                    
                    current_loss_item = total_loss.item()
                    
                    if not np.isnan(current_loss_item) and current_loss_item < self.best_loss:
                        self.best_loss = current_loss_item
                        self.best_model_state_dict = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
                        self.best_iteration = iteration
                        self.log.info(f"*** New best model found at iteration {iteration} (Train Loss: {self.best_loss:.4f}) ***")
                    if is_report_iter or iteration == self.best_iteration:  
                        self.saved_models[iteration] = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
                        self.saved_optimizer_states[iteration] = self.optimizer_gen.state_dict()
                else: 
                    valid_iters = [k for k in self.saved_models if k < iteration]
                    if valid_iters:
                        revert_iter = max(valid_iters)
                        self.log.warning(f"Reverting model and optimizer to state from iteration {revert_iter}")
                        try:
                            self.load_state_dict(self.saved_models[revert_iter])
                            self.to(current_device)
                            optimizer_gen = torch.optim.Adam([ # Re-init optimizer for the reverted state
                                {'params': self.encoder.parameters(), 'lr': self.user_parameters['learning_rate']},
                                {'params': self.decoder.parameters(), 'lr': self.user_parameters['learning_rate']}
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
                                {'params': self.encoder.parameters(), 'lr': self.user_parameters['learning_rate']},
                                {'params': self.decoder.parameters(), 'lr': self.user_parameters['learning_rate']}
                            ])
                        self.learning_stats[iteration] = {} 
                        self.learning_stats[iteration]['status'] = f'Reverted from NaN at {iteration}'
                    else:
                        self.log.error(f"NaNs/Infs detected in gradients at iter {iteration}, but no previous state found. Stopping.")
                        raise ValueError("NaNs/Infs encountered and cannot revert.")

                if is_report_iter:
                    self.eval()
                    all_test_stats_list = [] 
                    total_test_loss_items = []
                    with torch.no_grad():
                        total_test_loss, test_stats_from_calc = self.calculate_loss(
                            self.X_test, self.y_test, iteration, suffix='_test'
                        )
                        test_stats_from_calc['total_loss_test'] = total_test_loss.item()
                        all_test_stats_list.append(test_stats_from_calc)
                        total_test_loss_items.append(total_test_loss.item())
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
                    self.learning_stats[iteration]['total_loss_test_avg'] = avg_test_loss_item 

                    current_time = time.time()
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elapsed_time = current_time - last_report_time
                    iterations_since_last = iteration - last_report_iteration + 1 
                    avg_iter_time = elapsed_time / iterations_since_last if iterations_since_last > 0 else 0
                    self.log.info(f"Avg time/iter since last report: {avg_iter_time:.4f} seconds")
                    last_report_time = current_time
                    last_report_iteration = iteration
                    red_start = "\033[91m"; reset_color = "\033[0m"
                    log_msg_header = f"--- Iteration: {iteration}/{n_iterations} Eval (Global Test Set) ---"
                    self.log.info(log_msg_header)
                    if self.user_parameters['Verbose'] == 1: print(f"{red_start}{log_msg_header}{reset_color}")
                    log_msg_lr = f"Current LR: {self.user_parameters['learning_rate']:.6f}"
                    self.log.info(log_msg_lr)
                    if self.user_parameters['Verbose'] == 1: print(log_msg_lr)
                    train_loss_key = 'total_loss_train' 
                    if train_loss_key in self.learning_stats[iteration]:
                        log_msg = f'{train_loss_key}: {round(self.learning_stats[iteration][train_loss_key], 4)}'
                        self.log.info(log_msg)
                        if self.user_parameters['Verbose'] == 1: print(log_msg)
                    for name, item in avg_test_stats.items():
                        log_msg = f'{name}: {round(item, 4) if isinstance(item, (float, int)) and not np.isnan(item) else item}'
                        self.log.info(log_msg)
                        if self.user_parameters['Verbose'] == 1: print(log_msg)
                    self.log.info('------------------')

                if iteration > 20:
                    keys_to_delete = sorted([k for k in self.saved_models if k < iteration - 20 and k != 0 and k != self.best_iteration])
                    for key_to_del in keys_to_delete:
                        self.saved_models.pop(key_to_del, None)
                        self.saved_optimizer_states.pop(key_to_del, None)
        except Exception as e:
            self.log.exception(f"Error during training loop at iteration {iteration}: {e}")
        finally:
            if self.best_model_state_dict is not None:
                self.log.info(f"Loading best model state from iteration {self.best_iteration} (Train Loss: {self.best_loss:.4f}) before final save.")
                try:
                    missing_keys, unexpected_keys = self.load_state_dict(self.best_model_state_dict, strict=False)
                    if missing_keys: self.log.warning(f"Missing keys when loading best state_dict: {missing_keys}")
                    if unexpected_keys: self.log.warning(f"Unexpected keys when loading best state_dict: {unexpected_keys}")
                    self.to(current_device)
                    self.log.info(f"Successfully loaded best model state for final saving.")
                except Exception as e:
                    self.log.error(f"Failed to load best model state before saving: {e}. Saving the final iteration state instead.")
            else:
                self.log.warning("No best model state was saved during training. Saving the final iteration state.")
            output_dir = self.user_parameters['output']
            final_model_path = os.path.join(output_dir, 'final_model_state.pt')
            try:
                torch.save(self.state_dict(), final_model_path)
                self.log.info(f"Final model state dictionary saved to {final_model_path}")
            except Exception as e:
                self.log.error(f"Failed to save final model state: {e}")

            self.eval()
            final_iter_key = 'Final'
            self.learning_stats[final_iter_key] = {}
            with torch.no_grad():
                total_final_loss, final_stats_dict = self.calculate_loss(
                    self.X_test, self.y_test, iteration="Final", suffix='_test'
                )
                self.learning_stats[final_iter_key].update(final_stats_dict)
                self.learning_stats[final_iter_key]['total_loss_test_avg'] = total_final_loss.item()

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            red_start = "\033[91m"; reset_color = "\033[0m"
            log_prefix = f"--- Final Eval Stats (Global Test Set) at {now_str} ---"
            self.log.info(log_prefix)
            if self.user_parameters['Verbose'] == 1: print(f"{red_start}{log_prefix}{reset_color}")
            for name, item in self.learning_stats[final_iter_key].items():
                log_msg = f'{name}: {round(item, 4) if isinstance(item, (float, int)) and not np.isnan(item) else item}'
                self.log.info(log_msg)
                if self.user_parameters['Verbose'] == 1: print(log_msg)
            self.log.info('------------------')

            self.log.info('Total time taken: {:.2f} seconds'.format(time.time() - start_time))
            self.eval() 
            with torch.no_grad():
                final_E = self.get_encoding_weights().detach().clone() 
            self.log.info("Enforcing constraints on the final E matrix...")
            if self.constraints is None:
                self.log.error("Cannot enforce constraints: self.constraints is None.")
                self.E = final_E 
            else:
                E_final_constrained = torch.clip(final_E.round(), 0, None)
                T = self.constraints.clone().detach()
                m = E_final_constrained.sum(1) > T
                if m.any():
                    scaling_factors = (T[m] / E_final_constrained.sum(1)[m].clamp(min=1e-8)).unsqueeze(1)
                    E_final_constrained[m, :] = (E_final_constrained[m, :] * scaling_factors).floor()
                    E_final_constrained = E_final_constrained.clamp(min=0)
                self.E = E_final_constrained.clone().detach() 
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
            try:
                learning_df = pd.DataFrame.from_dict(self.learning_stats, orient='index')
                learning_curve_path = os.path.join(output_dir, 'learning_curve.csv')
                learning_df.to_csv(learning_curve_path)
                self.log.info(f"Learning curve data saved to {learning_curve_path}")
            except Exception as e:
                self.log.error(f"Failed to save learning curve: {e}")

    def evaluate(self):
        if self.encoder is None or self.decoder is None or \
           self.X_train is None or self.X_test is None or self.y_train is None or \
           self.y_test is None : 
            self.log.error("Cannot evaluate: Model not initialized or trained. Run initialize() and fit() first.")
            return

        self.results = {}
        current_device = self.user_parameters['device']

        # Use get_encoding_weights() to get proper encoding weights with sigmoid and constraints
        E_weights = self.get_encoding_weights()
        final_E_cpu = E_weights.cpu().detach()
        self.results['Number of Probes (Constrained)'] = final_E_cpu.sum().item()

        all_P_type = []
        X_global_train = self.X_train 
        y_global_train = self.y_train

        if X_global_train.shape[0] > 0:
            with torch.no_grad():
                # Use the E_weights we already calculated above
                P_global, _, _ = self.project(X_global_train, E_weights) 
                P_global_cpu = P_global.cpu()
                P_type_global = torch.zeros((self.n_categories, P_global_cpu.shape[1]), device='cpu')
                unique_y_global = torch.unique(y_global_train)
                for type_idx_tensor in unique_y_global:
                    type_idx = type_idx_tensor.item()
                    type_mask = (y_global_train == type_idx_tensor)
                    if type_mask.sum() > 0 and 0 <= type_idx < self.n_categories:
                        P_type_global[type_idx] = P_global_cpu[type_mask].mean(dim=0)
                all_P_type.append(P_type_global)

        if all_P_type:
            avg_P_type = torch.stack(all_P_type).mean(dim=0) 
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

        self.log.info("--- Basic Evaluation Stats ---")
        for key, val in self.results.items():
            log_msg = f" {key}: {round(val, 4) if isinstance(val, (float, int)) else val}"
            self.log.info(log_msg)
            if self.user_parameters['Verbose'] == 1: print(log_msg)
        self.log.info("-----------------------------")

        # Test model robustness under different noise conditions
        # We use the existing noise parameters in the model to simulate different noise levels
        # This ensures complete compatibility with the training pipeline
        noise_levels = {
            "No Noise": {
                'constant_noise': 0.0,
                'gene_fold_noise': 0.0,
                'gene_dropout_proportion': 0.0,
                'projection_dropout_proportion': 0.0,
                'weight_dropout_proportion': 0.0
            },
            "Low Noise": {
                'constant_noise': 2.0,
                'gene_fold_noise': 0.1,
                'gene_dropout_proportion': 0.02,
                'projection_dropout_proportion': 0.02,
                'weight_dropout_proportion': 0.02
            },
            "Medium Noise": {
                'constant_noise': 2.5,
                'gene_fold_noise': 0.5,
                'gene_dropout_proportion': 0.05,
                'projection_dropout_proportion': 0.05,
                'weight_dropout_proportion': 0.05
            },
            "High Noise": {
                'constant_noise': 3.0,
                'gene_fold_noise': 1.0,
                'gene_dropout_proportion': 0.1,
                'projection_dropout_proportion': 0.1,
                'weight_dropout_proportion': 0.1
            }
        }
        self.eval()
        for level_name, params in noise_levels.items():
            self.log.info(f"Calculating {level_name} Accuracy (Global)")
            try:
                # Store original parameters
                original_params = {}
                for param_name in params.keys():
                    original_params[param_name] = self.user_parameters[param_name]
                
                # Update parameters for this noise level
                for param_name, param_value in params.items():
                    self.user_parameters[param_name] = param_value
                
                with torch.no_grad():
                    # Use the existing pipeline: get_encoding_weights -> project -> decode
                    E_weights = self.get_encoding_weights()
                    P_test, Pnormalized_test, Pnormalized_dropout_test = self.project(self.X_test, E_weights)
                    y_pred_test, accuracy_test, _ = self.decode(Pnormalized_dropout_test, self.y_test)
                    avg_accuracy = accuracy_test.item()
                    self.log.info(f" {level_name} Accuracy: {round(avg_accuracy, 4)}")
                    self.results[f'{level_name} Accuracy'] = avg_accuracy 
                
                # Restore original parameters
                for param_name, original_value in original_params.items():
                    self.user_parameters[param_name] = original_value
                
            except Exception as e:
                self.log.error(f"Error during {level_name} accuracy calculation: {e}")
                self.results[f'{level_name} Accuracy'] = np.nan

        results_df = pd.DataFrame(self.results.values(), index=self.results.keys(), columns=['values'])
        results_path = os.path.join(self.user_parameters['output'], 'Results.csv') 
        results_df.to_csv(results_path)
        self.log.info(f"Evaluation results saved to {results_path}")

        if self.user_parameters['Verbose'] == 1:
            print("--- Evaluation Summary ---")
            for key, val in self.results.items():
                print(f" {key}: {round(val, 4) if isinstance(val, (float, int)) and not np.isnan(val) else val}")
            print("-------------------------------------------------")

    def visualize(self, show_plots=False): 
        self.log.info("Starting visualization generation...")
        if self.encoder is None or self.decoder is None or \
           self.X_train is None or self.y_train is None or \
           self.y_reverse_label_map is None : 
            self.log.error("Cannot visualize: Model not initialized. Run initialize() and fit() first.")
            return

        current_device = self.user_parameters['device']
        output_dir = self.user_parameters['output']
        saved_plot_paths = []

        # Use get_encoding_weights() to get proper encoding weights with sigmoid and constraints
        E_weights = self.get_encoding_weights()
        final_E_device = E_weights.to(current_device)
        self.eval()

        # Visualizations are now global
        global_name_str = "Global"
        global_fname_safe = sanitize_filename(global_name_str)
        self.log.info(f"Generating visualization for {global_name_str}...")

        X_data_vis = self.X_train # Use full training data
        y_data_vis = self.y_train # Mapped internal labels

        if X_data_vis.shape[0] == 0:
            self.log.warning(f"Skipping visualization for {global_name_str}: No training data found.")
            return

        with torch.no_grad():
            P_tensor_vis, _, _ = self.project(X_data_vis, final_E_device)
            P_np_vis = P_tensor_vis.cpu().numpy() 

            n_bits = P_tensor_vis.shape[1]
            P_type_global = torch.zeros((self.n_categories, n_bits), device=current_device)
            unique_y_indices_global = torch.unique(y_data_vis)

            valid_type_indices = []
            valid_type_labels = []

            for type_idx_tensor in unique_y_indices_global:
                type_idx = type_idx_tensor.item() 
                mask = (y_data_vis == type_idx_tensor)
                if mask.sum() > 0:
                    if 0 <= type_idx < self.n_categories:
                        P_type_global[type_idx] = P_tensor_vis[mask].mean(dim=0) 
                        valid_type_indices.append(type_idx)
                        valid_type_labels.append(self.y_reverse_label_map.get(type_idx, f"Type_{type_idx}"))
                    else:
                        self.log.warning(f"Skipping type index {type_idx} during P_type calculation (out of bounds).")
            
            if not valid_type_indices:
                self.log.warning(f"Skipping visualization for {global_name_str}: No valid cell types found after projection.")
                return

            P_type_global_present = P_type_global[valid_type_indices].cpu() 
            n_types_present = P_type_global_present.shape[0]

            if n_types_present > 1:
                P_type_centered = P_type_global_present - P_type_global_present.mean(dim=1, keepdim=True)
                P_type_std = P_type_centered.std(dim=1, keepdim=True).clamp(min=1e-6)
                P_type_norm = P_type_centered / P_type_std
                correlation_matrix = (P_type_norm @ P_type_norm.T / n_bits).numpy() 
                corr_df = pd.DataFrame(correlation_matrix, index=valid_type_labels, columns=valid_type_labels)
                fig_corr = None 
                try:
                    fig_width = min(max(8, n_types_present / 1.5), 25)
                    fig_height = min(max(6, n_types_present / 2), 25)
                    fig_corr = plt.figure(figsize=(fig_width, fig_height))
                    ax_corr = fig_corr.add_subplot(111) 
                    sns.heatmap(corr_df, annot=False, cmap='vlag', fmt=".2f", vmin=-1, vmax=1, center=0, linewidths=.5, ax=ax_corr, cbar=True) 
                    ax_corr.set_title(f"Type Correlation Matrix - {global_name_str}") 
                    plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right')
                    plt.setp(ax_corr.get_yticklabels(), rotation=0)
                    fig_corr.tight_layout()
                    plot_filename = f"type_correlation_heatmap_{global_fname_safe}.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    fig_corr.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plot_paths.append(plot_path) 
                    self.log.info(f"Saved Type Correlation plot for {global_name_str} to {plot_path}")
                except Exception as e:
                    self.log.error(f"Error generating Type Correlation heatmap for {global_name_str}: {e}")
                finally:
                    if fig_corr is not None:
                        plt.close(fig_corr) 
            else:
                self.log.warning(f"Skipping correlation plot for {global_name_str}: Only {n_types_present} cell type(s) present.")

            if n_types_present > 0:
                p_type_df = pd.DataFrame(P_type_global_present.clamp(min=1).log10().numpy(), 
                                         index=valid_type_labels,
                                         columns=[f"Bit_{b}" for b in range(n_bits)])
                cluster_fig = None 
                try:
                    fig_width = min(max(6, n_bits / 1.5), 25)
                    fig_height = min(max(6, n_types_present / 2), 25)
                    cluster_fig = sns.clustermap(p_type_df,
                                                 cmap="inferno", 
                                                 figsize=(fig_width, fig_height),
                                                 linewidths=0.1,
                                                 dendrogram_ratio=(.2, .2) 
                                                 )
                    cluster_fig.fig.suptitle(f"Average Projection (P_type) - {global_name_str}", y=1.02) 
                    cluster_fig.ax_heatmap.set_xlabel("Projection Bit")
                    cluster_fig.ax_heatmap.set_ylabel("Cell Type (Clustered)")
                    plt.setp(cluster_fig.ax_heatmap.get_xticklabels(), rotation=90) 
                    plt.setp(cluster_fig.ax_heatmap.get_yticklabels(), rotation=0) 
                    plot_filename = f"P_type_clustermap_{global_fname_safe}.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    cluster_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plot_paths.append(plot_path) 
                    self.log.info(f"Saved P_type clustermap for {global_name_str} to {plot_path}")
                except Exception as e:
                    self.log.error(f"Error generating P_type clustermap for {global_name_str}: {e}")
                finally:
                    if cluster_fig is not None:
                        plt.close(cluster_fig.fig)
            else:
                self.log.warning(f"Skipping P_type plot for {global_name_str}: No cell types present.")

            if n_types_present > 0 and n_bits >= 2:
                plot_filename = f"projection_density_plot_{global_fname_safe}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                try:
                    y_vis_str_labels = np.array([self.y_reverse_label_map.get(idx.item(), f"Type_{idx.item()}") for idx in y_data_vis])
                    # Use the proper encoding weights for the projection density plot
                    E_weights_cpu = E_weights.cpu().numpy()
                    plot_projection_space_density(
                        X_data_vis.cpu().numpy() @ E_weights_cpu, 
                        y_vis_str_labels, 
                        plot_path,
                        sum_norm=False, 
                        log=True
                        )
                    saved_plot_paths.append(plot_path)
                except Exception as e:
                    self.log.error(f"Error generating projection space density plot for {global_name_str}: {str(e)}")
            elif n_bits < 2:
                self.log.warning(f"Skipping projection space density plot for {global_name_str}: Requires at least 2 bits (found {n_bits}).")

            self.log.info(f"Generating confusion matrix for test data (Global)...")
            X_test_global = self.X_test
            y_test_global = self.y_test # True internal labels

            if X_test_global.shape[0] > 0:
                fig_cm = None 
                try:
                    with torch.no_grad():
                        P_test_tensor, Pnorm_test_tensor, Pnorm_dropout_test_tensor = self.project(X_test_global, final_E_device)
                        y_pred_test, _, _ = self.decode(Pnorm_dropout_test_tensor, y_test_global)

                    y_true_np = y_test_global.cpu().numpy()
                    y_pred_np = y_pred_test.cpu().numpy()
                    present_labels_idx = np.unique(np.concatenate((y_true_np, y_pred_np)))
                    present_labels_str = [self.y_reverse_label_map.get(i, f"Type_{i}") for i in present_labels_idx]

                    if len(present_labels_idx) > 0: 
                        cm = confusion_matrix(y_true_np, y_pred_np, labels=present_labels_idx) 
                        cm_sum = cm.sum(axis=1, keepdims=True)
                        cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) 
                        cm_df = pd.DataFrame(cm_norm, index=present_labels_str, columns=present_labels_str)
                        fig_width = min(max(10, len(present_labels_str) / 1.5), 25)
                        fig_height = min(max(8, len(present_labels_str) / 2), 25)
                        fig_cm = plt.figure(figsize=(fig_width, fig_height))
                        ax_cm = fig_cm.add_subplot(111)
                        sns.heatmap(cm_df, annot=False, cmap='jet', linewidths=0.1, ax=ax_cm, vmin=0, vmax=1) 
                        ax_cm.set_xlabel('Predicted Label')
                        ax_cm.set_ylabel('True Label')
                        ax_cm.set_title(f'Confusion Matrix (Test Set) - {global_name_str}')
                        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right')
                        plt.setp(ax_cm.get_yticklabels(), rotation=0)
                        fig_cm.tight_layout()
                        plot_filename = f"confusion_matrix_test_{global_fname_safe}.png"
                        plot_path = os.path.join(output_dir, plot_filename)
                        fig_cm.savefig(plot_path, dpi=300, bbox_inches='tight')
                        saved_plot_paths.append(plot_path)
                        self.log.info(f"Saved Test Confusion Matrix for {global_name_str} to {plot_path}")
                    else:
                        self.log.warning(f"Skipping confusion matrix for {global_name_str}: No labels found in true or predicted test data.")
                except Exception as e:
                    self.log.error(f"Error generating Test Confusion Matrix for {global_name_str}: {e}")
                finally:
                    if fig_cm is not None:
                        plt.close(fig_cm) 
            else:
                self.log.warning(f"Skipping confusion matrix for {global_name_str}: No test data found.")
        self.log.info("Visualization generation finished.")

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
                                   figsize=(12, 5 * total_rows), 
                                   squeeze=False) 

    color_mapper = {}
    used_colors_list = [np.array([0., 0., 0.]), np.array([1., 1., 1.])]

    plot_pair_idx = 0 
    for i_pair_start in range(num_measurements):
        if i_pair_start % 2 == 0 and plot_pair_idx < num_plot_pairs: 
            feature_idx1 = i_pair_start
            if feature_idx1 + 1 >= num_measurements:
                feature_idx2 = feature_idx1 - 1 
                if feature_idx2 < 0: continue 
            else:
                feature_idx2 = feature_idx1 + 1

            feature_name1 = labels[feature_idx1]
            feature_name2 = labels[feature_idx2]

            x = np.array(P[:, feature_idx1]).ravel()
            y = np.array(P[:, feature_idx2]).ravel()

            x_pos = x[x > 0]
            y_pos = y[y > 0]

            if len(x_pos) > 1:
                vmin_x, vmax_x = np.percentile(x_pos, [0.1, 99.9])
            elif len(x_pos) == 1: vmin_x, vmax_x = x_pos[0], x_pos[0]
            else: vmin_x, vmax_x = 0, 0

            if len(y_pos) > 1:
                vmin_y, vmax_y = np.percentile(y_pos, [0.1, 99.9])
            elif len(y_pos) == 1: vmin_y, vmax_y = y_pos[0], y_pos[0]
            else: vmin_y, vmax_y = 0, 0

            vmax_x = max(vmax_x, vmin_x)
            vmax_y = max(vmax_y, vmin_y)

            x = np.clip(x, vmin_x, vmax_x)
            if log:
                x = np.log10(x + 1) 
            x_min, x_max = x.min(), x.max()
            x_bins = np.linspace(x_min, x_max if x_max > x_min else x_max + 1, 100)

            y = np.clip(y, vmin_y, vmax_y)
            if log:
                y = np.log10(y + 1) 
            y_min, y_max = y.min(), y.max()
            y_bins = np.linspace(y_min, y_max if y_max > y_min else y_max + 1, 100)

            current_row_idx = plot_pair_idx
            
            ax1 = axes[current_row_idx, 0] 

            img, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
            img = np.log10(img + 1) 

            img_pos = img[img > 0]
            if len(img_pos) > 0:
                vmin_img, vmax_img = np.percentile(img_pos, [0.1, 99]) 
                if vmin_img == vmax_img: vmax_img += 1e-6 
            else: vmin_img, vmax_img = 0, 1 

            im1 = ax1.imshow(img.T, vmin=vmin_img, vmax=vmax_img, cmap='bwr', origin='lower', aspect='auto', interpolation='nearest',
                           extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])

            num_ticks = 5
            x_tick_labels_val = np.linspace(x_bins[0], x_bins[-1], num=num_ticks)
            y_tick_labels_val = np.linspace(y_bins[0], y_bins[-1], num=num_ticks)
            ax1.set_xticks(x_tick_labels_val)
            ax1.set_yticks(y_tick_labels_val)

            ax1.set_xticklabels(np.round(x_tick_labels_val, 1))
            ax1.set_yticklabels(np.round(y_tick_labels_val, 1))

            if log:
                ax1.set_xlabel(f"Bit {feature_name1} (log10)")
                ax1.set_ylabel(f"Bit {feature_name2} (log10)")
            else:
                ax1.set_xlabel(f"Bit {feature_name1}")
                ax1.set_ylabel(f"Bit {feature_name2}")
            ax1.grid(False)
            
            ax2 = axes[current_row_idx, 1] 

            composite_img = np.zeros((len(y_bins)-1, len(x_bins)-1, 3))
            legend_handles = [] 

            for ct in unique_cell_types:
                mask = y_labels == ct
                if np.sum(mask) < 2:
                    continue

                img_ct, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_bins, y_bins])
                img_ct = np.log10(img_ct + 1)

                img_ct_pos = img_ct[img_ct > 0]
                if len(img_ct_pos) >= 2: 
                    vmin_ct, vmax_ct = np.percentile(img_ct_pos, [0.1, 99]) 
                    vmin_ct = 0 
                    if vmax_ct <= vmin_ct: vmax_ct = vmin_ct + 1e-6 
                    if vmax_ct > 1e-9:
                        img_ct_norm = (img_ct - vmin_ct) / vmax_ct
                    else:
                        img_ct_norm = np.zeros_like(img_ct)
                elif len(img_ct_pos) == 1: 
                    img_ct_norm = (img_ct > 0).astype(float) 
                else: 
                    img_ct_norm = np.zeros_like(img_ct) 

                img_ct_norm = np.clip(img_ct_norm, 0, 1).T 

                if ct not in color_mapper:
                    attempts = 0
                    max_attempts = 200
                    min_dist_sq = 0.1 
                    min_sum = 0.5 

                    while attempts < max_attempts:
                        color = np.random.rand(3)
                        color_sum = np.sum(color)
                        distances_sq = [np.sum((color - existing_color)**2) for existing_color in used_colors_list]
                        min_d2 = min(distances_sq) if distances_sq else 1.0

                        if min_d2 > min_dist_sq and color_sum > min_sum:
                            color_mapper[ct] = color
                            used_colors_list.append(color)
                            break
                        attempts += 1
                    if ct not in color_mapper: 
                        color_mapper[ct] = np.random.rand(3) * 0.8
                        used_colors_list.append(color_mapper[ct])

                ct_layer = np.dstack([img_ct_norm] * 3) * color_mapper[ct]
                composite_img += ct_layer

            if composite_img.max() > 0:
                vmax_composite = 1.0 
                composite_img = composite_img / max(vmax_composite, 1e-9) 

            composite_img = np.clip(composite_img, 0, 1)

            ax2.imshow(composite_img, origin='lower', aspect='auto', interpolation='nearest',
                       extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]) 

            ax2.set_xticks(x_tick_labels_val)
            ax2.set_xticklabels(np.round(x_tick_labels_val, 1)) 
            ax2.set_yticks(y_tick_labels_val)
            ax2.set_yticklabels(np.round(y_tick_labels_val, 1)) 
            if log:
                ax2.set_xlabel(f"Bit {feature_name1} (log10)")
                ax2.set_ylabel(f"Bit {feature_name2} (log10)")
            else:
                ax2.set_xlabel(f"Bit {feature_name1}")
                ax2.set_ylabel(f"Bit {feature_name2}")
            ax2.grid(False)
            handles = [Patch(color=color_mapper[ct], label=ct) for ct in unique_cell_types if ct in color_mapper]
            # ax2.legend(handles=handles, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')

            plot_pair_idx += 1 

    try:
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Saved projection space density plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}")
    finally:
        plt.close(fig)


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
