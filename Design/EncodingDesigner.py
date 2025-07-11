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
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import confusion_matrix
import argparse
from typing import Optional, Dict, Any, List, Union
try:
    from IPython.core.getipython import get_ipython
    from IPython.display import Image, display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


class EncodingDesigner(nn.Module):
    def __init__(self, user_parameters_path: Optional[str] = None):
        super().__init__() 
        self.I: Dict[str, Any] = {
            # Core model parameters
            'n_cpu': 12,  # Number of CPU threads to use for PyTorch
            'n_bit': 24,  # Number of bits in the encoding (dimensionality of the projection)
            'n_iters': 10000,  # Total number of training iterations
            'batch_size': 1000,  # Batch size for training (0 = use full dataset)
            'brightness': 4.5,  # Target brightness in log10 scale
            'n_probes': 30e4,  # Target total number of probes across all genes
            'probe_wt': 1,  # Weight for probe count loss term
            'gene_constraint_wt': 1,  # Weight for gene constraint violation penalty
            'brightness_wt':1,  # Weight for target brightness loss term
            'dynamic_wt': 1,  # Weight for dynamic range loss terms
            'dynamic_fold_s': 2.0,  # Initial target fold change for dynamic range
            'dynamic_fold_e': 2.0,  # Final target fold change for dynamic range
            'separation_wt': 1,  # Weight for cell type separation loss term
            'separation_fold_s': 3.0,  # Initial minimum fold change required between cell type pairs
            'separation_fold_e': 3.0,  # Final minimum fold change required between cell type pairs
            'gradient_clip': 1,  # Maximum gradient norm for clipping
            'lr_s': 0.05,  # Initial learning rate
            'lr_e': 0.005,  # Final learning rate (linear interpolation)
            'report_rt': 250,  # How often to report training progress
            'sparsity_s': 0.8,  # Initial target sparsity ratio (fraction of zeros)
            'sparsity_e': 0.8,  # Final target sparsity ratio (fraction of zeros)
            'sparsity_wt': 0,  # Weight for sparsity loss term
            'categorical_wt': 1,  # Weight for categorical classification loss
            'label_smoothing': 0.1,  # Label smoothing factor for cross-entropy loss
            'best_model': 1,  # Whether to save the best model during training
            'device': 'cpu',  # Device to run computations on ('cpu' or 'cuda')
            'output': '/u/project/rwollman/rwollman/atlas_design/design_results',  # Output directory path
            'input': './',  # Input directory path
            'decoder_n_lyr': 0,  # Number of hidden layers in decoder
            'top_n_genes': 0,  # Number of top genes to keep (0 = keep all genes)
            'constraints': 'constraints.csv',  # Path to gene constraints file
            'X_test': 'X_test.pt',  # Path to test features tensor
            'y_test': 'y_test.pt',  # Path to test labels tensor
            'X_train': 'X_train.pt',  # Path to training features tensor
            'y_train': 'y_train.pt',  # Path to training labels tensor
            'y_label_converter_path': 'categorical_converter.csv',  # Path to label mapping file
            'P_scaling': 24,  # Scaling factor for sum normalization (defaults to n_bit)
            # Gene-level noise parameters
            'X_drp_s': 0,  # Initial proportion of genes to drop out (randomly set to 0)
            'X_drp_e': 0.1,  # Final proportion of genes to drop out (randomly set to 0)
            'X_noise_s': 0,  # Initial gene expression noise level 0.5 -> 50% decrease to 200% increase (0-1)
            'X_noise_e': 0.5,  # Final gene expression noise level 0.5 -> 50% decrease to 200% increase (0-1)
            # Weight-level noise parameters
            'E_drp_s': 0,  # Initial proportion of encoding weights to drop out (randomly set to 0)
            'E_drp_e': 0.1,  # Final proportion of encoding weights to drop out (randomly set to 0)
            'E_noise_s': 0,  # Initial encoding weight noise level (percentage decrease with minimum bound 0-1)
            'E_noise_e': 0.1,  # Final encoding weight noise level (percentage decrease with minimum bound 0-1)
            # Projection-level noise parameters
            'P_drp_s': 0,  # Initial proportion of projection values to drop out (randomly set to 0)
            'P_drp_e': 0.1,  # Final proportion of projection values to drop out (randomly set to 0)
            'P_noise_s': 0,  # Initial projection measurement noise level (percentage accuracy error 0-1)
            'P_noise_e': 0.1,  # Final projection measurement noise level (percentage accuracy error 0-1)
            # Decoder-level noise parameters
            'D_drp_s': 0.1,  # Initial decoder dropout rate
            'D_drp_e': 0,  # Final decoder dropout rate
            # Constant noise parameters
            'P_add_s': 1,  # Initial constant noise level (log10 scale, added to projections)
            'P_add_e': 3.0,  # Final constant noise level (log10 scale, added to projections)
            # Weight perturbation parameters
            'E_perturb_rt': 500,  # How often to perturb weights (every N iterations)
            'E_perb_prct': 0.01,  # Percentage of weights to perturb (0.0-1.0)
            'E_init_min': 0.05,  # Minimum probe fraction for initialization
            'E_init_max': 0.5,  # Maximum probe fraction for initialization
            'E_perturb_min': 0.05,  # Minimum probe fraction for perturbation
            'E_perturb_max': 0.5,  # Maximum probe fraction for perturbation
            # Activation and normalization parameters
            'encoder_act':'tanh',  # Activation function for encoding weights
            'decoder_act': 'gelu',  # Activation function for decoder hidden layers ('relu', 'leaky_relu', 'gelu', 'swish', 'tanh')
            'sum_norm': 0,  # Whether to normalize projection by sum
            'bit_norm': 0,  # Whether to normalize projection by bit-wise statistics
            'continue_training': 1,  # Whether to continue training if model is loaded from file (0 = skip training, 1 = continue training) 
            'use_noise': 1,  # Whether to apply noise/dropout during training (0 = no noise, 1 = use noise)
        }
        self._setup_logging(user_parameters_path)
        self._load_and_process_parameters(user_parameters_path)
        self._setup_output_and_symlinks(user_parameters_path)
        self.results = {}
        self.learning_stats = {}
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1
        self.training_completed = False
        # Weight tracking for change calculation
        self.prev_encoder_weights = None
        self.prev_decoder_weights = None

    def initialize(self):
        self.log.info("--- Starting Initialization ---")
        try:
            self.log.info("Loading Gene Constraints")
            self._load_constraints()
            self._load_data()
            if self.I['top_n_genes'] > 0 and self.I['top_n_genes'] < self.n_genes:
                self.train_gene_importance_decoder()
            self._initialize_encoder()
            self._initialize_decoder()
            self._load_pretrained_model()
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

    def get_E_clean(self):
        """Get encoding weights without noise/dropout for loss calculations."""
        if self.I['encoder_act'] == 'sigmoid':
            E = torch.sigmoid(self.encoder.weight) * self.constraints.unsqueeze(1)
        elif self.I['encoder_act'] == 'tanh':
            # modified tanh activation function (torch.tanh(x)+1)/2
            E = ((torch.tanh(self.encoder.weight)+1)/2) * self.constraints.unsqueeze(1)
        elif self.I['encoder_act'] == 'linear':
            E = self.encoder.weight * self.constraints.unsqueeze(1)
        elif self.I['encoder_act'] == 'relu':
            E = torch.relu(self.encoder.weight) * self.constraints.unsqueeze(1)
        else:
            raise ValueError(f"Invalid activation function: {self.I['encoder_act']}")
        return E

    def get_E(self):
        """Get encoding weights with noise/dropout for training."""
        E = self.get_E_clean()
        if self.I['use_noise'] == 0:
            return E
        if self.training and self.I['E_drp'] > 0:
            E = E * (torch.rand_like(E) > self.I['E_drp']).float()
        if self.training and self.I['E_noise'] > 0:
            # Set a lower bound to the percent of probes that can bind
            maximum_percent_decrease = self.I['E_noise']
            min_val = 1-maximum_percent_decrease
            E = E * (((1 - min_val) * torch.rand_like(E)) + min_val)
        return E

    def project_clean(self, X, E):
        """Project data without noise/dropout for loss calculations."""
        P = X.mm(E)
        return P

    def project(self, X, E):
        """Project data with noise/dropout for training."""
        if self.I['use_noise'] == 0:
            return X.mm(E)
        if self.training and self.I['X_noise'] != 0:
            fold = 1 / (1 - self.I['X_noise'])
            X = X * torch.exp(torch.rand_like(X) * 2 * torch.log(torch.tensor(fold)) - torch.log(torch.tensor(fold)))
        if self.training and self.I['X_drp'] != 0:
            X = X * (torch.rand_like(X) > self.I['X_drp']).float()
        P = X.mm(E)
        if self.training and self.I['P_noise'] > 0:
            # modify P by a percent change to account for measurement accuracy
            max_accuracy = self.I['P_noise']
            P = P + (P * ((2*torch.rand_like(P)-1)*max_accuracy))
        if self.training and self.I['P_add'] != 0:
            P = P + (torch.rand_like(P) * (10 ** self.I['P_add']))
        if self.training and self.I['P_drp'] > 0:
            P = P * (torch.rand_like(P) > self.I['P_drp']).float()
        return P

    def decode(self, P, y):
        if self.I['sum_norm'] != 0:
            P = self.I['P_scaling'] * P / P.sum(1).unsqueeze(1).clamp(min=1e-8)
            # P = (P - P.mean(1).unsqueeze(1)) / P.std(1).unsqueeze(1).clamp(min=1e-8)
        if self.I['bit_norm'] != 0:
            P = (P - P.mean(0)) / P.std(0).clamp(min=1e-8)
        R = self.decoder(P) 
        y_predict = R.max(1)[1]
        accuracy = (y_predict == y).float().mean()
        if self.I['categorical_wt'] != 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.I['label_smoothing']) 
            categorical_loss = loss_fn(R, y)
        else:
            categorical_loss = torch.tensor(0, device=R.device, dtype=torch.float32, requires_grad=True)
        return y_predict, accuracy, categorical_loss

    def calculate_loss(self, X, y, iteration, suffix='') -> tuple[torch.Tensor, dict]:
        # Get clean versions for loss calculations
        E_clean = self.get_E_clean()
        P_clean = self.project_clean(X, E_clean)
        # Get noisy versions for decoder training
        E_noisy = self.get_E()
        P_noisy = self.project(X, E_noisy)
        # Use noisy projections for decoder training
        y_predict, accuracy, raw_categorical_loss_component = self.decode(P_noisy, y)
        raw_losses = {}
        current_stats = {}

        # --- Probe count loss ---
        # if fold is double target loss is 1 * probe_wt
        # if fold is 0 loss is 0 * probe_wt
        # if fold is below target loss is approaching negative alpha * probe_wt
        fold = (E_clean.sum()-self.I['n_probes'])/self.I['n_probes']
        probe_wt_loss = self.I['probe_wt'] * F.elu(fold,alpha=0.05)
        raw_losses['probe_wt_loss'] = probe_wt_loss
        current_stats['E_n_probes' + suffix] = round(E_clean.sum().item(), 4)

        # --- Gene constraint loss ---
        # if 0% of probes are constrained loss is 0
        # if 1% of probes are constrained loss is 0.01 * gene_constraint_wt
        # if 100% of probes are constrained loss is 1 * gene_constraint_wt
        if self.constraints is None: raise RuntimeError("Constraints not initialized")
        total_probes_per_gene = E_clean.sum(1)
        non_zero_constraints = self.constraints>0
        total_probes_per_gene = total_probes_per_gene[non_zero_constraints]
        constraints = self.constraints[non_zero_constraints].clamp(min=1)
        difference = total_probes_per_gene-constraints
        violations = difference>1
        if violations.any():
            raw_losses['gene_constraint_loss'] = self.I['gene_constraint_wt'] * (difference[violations].mean()-1)
        else:
            raw_losses['gene_constraint_loss'] = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        current_stats['E_total_n_genes' + suffix] = (E_clean > 1).any(1).sum().item()
        current_stats['E_median_wt' + suffix] = round(E_clean[E_clean > 1].median().item() if (E_clean > 1).any() else 0, 4)
        current_stats['n_genes_over_constraint' + suffix] = violations.sum().item()
        current_stats['avg_over_constraint' + suffix] = round(difference[violations].mean().item(), 4) if violations.any() else 0
        current_stats['max_over_constraint' + suffix] = round(difference.max().item(), 4) if difference.numel() > 0 else 0
        current_stats['total_violation_probes' + suffix] = round(difference[violations].sum().item(), 4)

        # --- Sparsity loss ---
        sparsity_ratio = (E_clean < 1).float().mean()
        target = self.I['sparsity']
        fold = (target - sparsity_ratio) / target
        sparsity_loss = self.I['sparsity_wt'] * F.elu(fold,alpha=0.1)
        raw_losses['sparsity_loss'] = sparsity_loss
        current_stats['sparsity' + suffix] = round(sparsity_ratio.item(), 4)

        # Brightness and dynamic range analysis with configurable percentiles
        median_val = P_clean.sum(1).clamp(min=1e-8).median()
        P_clean_sum_norm = median_val * P_clean / P_clean.sum(1).unsqueeze(1).clamp(min=1e-8)
        quant = {
            'lower_min': 0.05, 
            'lower_max': 0.15, 
            'median_min': 0.45,
            'median_max':0.55,
            'upper_min': 0.85,
            'upper_max': 0.95}
        signal = {
            'lower': torch.zeros(P_clean.shape[1], device=P_clean.device),
            'median': torch.zeros(P_clean.shape[1], device=P_clean.device),
            'upper': torch.zeros(P_clean.shape[1], device=P_clean.device)
        }
        bit_dynamic_ranges = []
        bit_percentiles = []
        for bit_idx in range(P_clean.shape[1]):
            bit_values = P_clean_sum_norm[:, bit_idx]
            quantiles_tensor = torch.tensor([val for val in quant.values()], device=bit_values.device)
            quant_results = torch.quantile(bit_values, quantiles_tensor)
            for i, region in enumerate(signal.keys()):
                mask = (bit_values >= quant_results[i * 2]) & (bit_values <= quant_results[i * 2 + 1])
                signal[region][bit_idx] = bit_values[mask].mean() if mask.any() else torch.tensor(0.0, device=bit_values.device)
            fold_change = signal['upper'][bit_idx] / signal['lower'][bit_idx].clamp(min=1e-8)
            bit_dynamic_ranges.append(fold_change.item())
            bit_percentiles.append((signal['lower'][bit_idx].item(), 
                                  signal['median'][bit_idx].item(), 
                                  signal['upper'][bit_idx].item()))
        min_range_idx = bit_dynamic_ranges.index(min(bit_dynamic_ranges))
        max_range_idx = bit_dynamic_ranges.index(max(bit_dynamic_ranges))
        min_lower, min_median, min_upper = bit_percentiles[min_range_idx]
        max_lower, max_median, max_upper = bit_percentiles[max_range_idx]
        current_stats['E_worst_bit' + suffix] = f"min:{max(min_lower, 1):.2e}, med:{max(min_median, 1):.2e}, max:{max(min_upper, 1):.2e}, fold:{bit_dynamic_ranges[min_range_idx]:.2f}"
        current_stats['E_best_bit' + suffix] = f"min:{max(max_lower, 1):.2e}, med:{max(max_median, 1):.2e}, max:{max(max_upper, 1):.2e}, fold:{bit_dynamic_ranges[max_range_idx]:.2f}"
        
        # --- Brightness loss ---
        target = 10**self.I['brightness']
        median_brightness = signal['median'].clamp(min=1)
        fold = (target-median_brightness)/target
        positive_fold = fold[fold>0]
        if positive_fold.numel() > 0:
            brightness_loss = self.I['brightness_wt'] * positive_fold.sum()
        else:
            brightness_loss = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        raw_losses['brightness_loss'] = brightness_loss
        current_stats['lower brightness' + suffix] = round(signal['lower'].mean().item(), 4)
        current_stats['median brightness' + suffix] = round(signal['median'].mean().item(), 4)
        current_stats['upper brightness' + suffix] = round(signal['upper'].mean().item(), 4)
        current_stats['dynamic_range' + suffix] = round((signal['upper']-signal['lower']).mean().item(), 4)

        # --- Dynamic range loss ---
        dynamic_range = signal['median'] / signal['lower'].clamp(min=1e-8)
        current_stats['lower_dynamic_fold' + suffix] = round(dynamic_range.mean().item(), 4)
        dynamic_range = signal['upper'] / signal['median'].clamp(min=1e-8)
        current_stats['upper_dynamic_fold' + suffix] = round(dynamic_range.mean().item(), 4)
        target = self.I['dynamic_fold']
        dynamic_range = signal['upper'] / signal['lower'].clamp(min=1e-8)
        fold = (target - dynamic_range) / target
        fold = fold[fold>0]
        if fold.numel() > 0:
            raw_losses['dynamic_loss'] = self.I['dynamic_wt'] * fold.sum()
        else:
            raw_losses['dynamic_loss'] = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        current_stats['dynamic_fold' + suffix] = round(dynamic_range.mean().item(), 4)

        # --- Cell type separation loss ---
        batch_categories = torch.unique(y)
        if len(batch_categories) > 1:
            P_data = torch.zeros((len(batch_categories), P_clean.shape[1]), device=P_clean.device)
            for i, type_idx in enumerate(batch_categories):
                v = P_clean_sum_norm[y == type_idx].mean(dim=0)
                v = v/v.sum().clamp(min=1e-8)
                P_data[i] = v
            mask = ~torch.eye(len(batch_categories), dtype=torch.bool, device=P_clean.device)
            P_i = P_data.unsqueeze(1)
            P_j = P_data.unsqueeze(0)
            difference = torch.abs(P_i - P_j)
            smallest_value = torch.minimum(P_i, P_j).clamp(min=1e-8)
            separations = (difference / smallest_value)[mask].max(dim=1)[0]
            target = self.I['separation_fold']
            fold = (target - separations) / target
            fold = fold[fold>0]
            if fold.numel() > 0:
                raw_losses['separation_loss'] = self.I['separation_wt'] * fold.mean()
            else:
                raw_losses['separation_loss'] = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
            worst_separation = separations.min().item()
            p10,p50,p90 = torch.quantile(separations, torch.tensor([0.1, 0.5, 0.9]))
            best_separation = separations.max().item()
            current_stats['separation' + suffix] = round(p10.item(), 4)
            current_stats['separation_report' + suffix] = f"min:{worst_separation:.2f}, p10:{p10:.2f}, p50:{p50:.2f}, p90:{p90:.2f}, max:{best_separation:.2f}"

        # --- Categorical loss ---
        categorical_loss_component = self.I['categorical_wt'] * raw_categorical_loss_component
        raw_losses['categorical_loss'] = categorical_loss_component
        current_stats['accuracy' + suffix] = round(accuracy.item(), 4)
        
        # Calculate weight changes
        current_stats['E_change' + suffix] = 0.0
        if self.prev_encoder_weights is not None:
            current_weights = self.encoder.weight.data
            prev_weights = self.prev_encoder_weights
            non_zero_mask = (current_weights != 0) & (prev_weights != 0)
            if non_zero_mask.any():
                pct_changes = torch.abs((current_weights[non_zero_mask] - prev_weights[non_zero_mask]) / prev_weights[non_zero_mask])
                current_stats['E_change' + suffix] = round(pct_changes.mean().item(), 4)
        self.prev_encoder_weights = self.encoder.weight.data.clone().detach()
            
        module = self.decoder[0]
        current_stats['D_change' + suffix] = 0.0
        if (self.prev_decoder_weights is not None) and (isinstance(module, nn.Linear)):
            current_weights = module.weight.data
            prev_weights = self.prev_decoder_weights
            non_zero_mask = (current_weights != 0) & (prev_weights != 0)
            if non_zero_mask.any():
                pct_changes = torch.abs((current_weights[non_zero_mask] - prev_weights[non_zero_mask]) / prev_weights[non_zero_mask])
                current_stats['D_change' + suffix] = round(pct_changes.mean().item(), 4)
        if isinstance(module, nn.Linear):
            self.prev_decoder_weights = module.weight.data.clone().detach()

        for key, value in raw_losses.items():
            current_stats['$$$ - ' + key + suffix] = round(value.item(), 4)
        #total_loss is a tensor
        if len(raw_losses) == 0:
            total_loss = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        else:
            total_loss = sum(raw_losses.values()) # tensor not int
            if isinstance(total_loss, int):
                total_loss = torch.tensor(total_loss, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        current_stats['$$$ - total_loss' + suffix] = round(total_loss.item(), 4)
        
        return total_loss, current_stats

    def train_gene_importance_decoder(self):
        """Train a simple linear decoder from genes to cell types to identify important genes."""
        self.log.info("Training gene importance decoder...")
        device = self.X_train.device
        n_genes = self.X_train.shape[1]
        n_categories = len(torch.unique(self.y_train))
        # Create simple linear decoder: genes -> cell types
        decoder = nn.Linear(n_genes, n_categories).to(device)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        # Track training metrics
        losses = []
        accuracies = []
        # Training loop
        for epoch in range(500):  # Hardcoded 100 epochs
            decoder.train()
            optimizer.zero_grad()
            outputs = decoder(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
            # Calculate accuracy
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                accuracy = (predictions == self.y_train).float().mean()
                losses.append(loss.item())
                accuracies.append(accuracy.item())
            if (epoch + 1) % 50 == 0:
                self.log.info(f"Gene importance decoder epoch {epoch+1}/1000, loss: {loss.item():.4f}, accuracy: {accuracy.item():.4f}")
            if accuracy.item() > 0.99:
                break
        # Extract gene importance scores from the trained weights
        with torch.no_grad():
            gene_importance_scores = torch.abs(decoder.weight).sum(dim=0)  # Sum across output classes
        self.log.info(f"Gene importance decoder training complete. Score range: [{gene_importance_scores.min().item():.4f}, {gene_importance_scores.max().item():.4f}]")
        # Find top genes
        top_n = min(self.I['top_n_genes'], 100)
        self.log.info(f"Top {top_n} genes by importance score:")
        # Filter genes based on importance scores
        top_n_actual = self.I['top_n_genes']
        top_gene_indices = torch.argsort(gene_importance_scores, descending=True)[:top_n_actual]
        gene_mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
        gene_mask[top_gene_indices] = True
        # Filter data matrices
        self.X_train = self.X_train[:, gene_mask]
        self.X_test = self.X_test[:, gene_mask]
        self.constraints = self.constraints[gene_mask]
        self.genes = self.genes[gene_mask.cpu().numpy()]
        self.n_genes = top_n_actual
        gene_importance_df = pd.DataFrame({
            'gene_name': self.genes,
            'importance_score': gene_importance_scores[gene_mask].cpu().numpy()
        })
        gene_importance_path = os.path.join(self.I['output'], 'gene_importance_scores.csv')
        gene_mask_path = os.path.join(self.I['output'], 'gene_mask.pt')
        gene_importance_df.to_csv(gene_importance_path, index=False)
        torch.save(gene_mask.cpu(), gene_mask_path)
        self.log.info(f"Gene importance scores saved to {gene_importance_path}")
        self.log.info(f"Gene mask saved to {gene_mask_path}")
        # Create training plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(losses)
        ax1.set_title('Gene Importance Decoder Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax2.plot(accuracies)
        ax2.set_title('Gene Importance Decoder Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        plt.tight_layout()
        training_plots_path = os.path.join(self.I['output'], 'gene_importance_decoder_training.pdf')
        plt.savefig(training_plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.log.info(f"Training plots saved to {training_plots_path}")
        self.log.info(f"Gene filtering complete: {self.n_genes} genes retained")
        return gene_importance_scores

    def perturb_E(self):
        with torch.no_grad():
            mask = (torch.rand_like(self.encoder.weight.data) < self.I['E_perb_prct']).float()
            random_wts = torch.rand_like(self.encoder.weight.data)
            min_value = torch.tensor(self.I['E_perturb_min'], 
                                    dtype=random_wts.dtype, device=random_wts.device)
            max_value = torch.tensor(self.I['E_perturb_max'], 
                                    dtype=random_wts.dtype, device=random_wts.device)
            post_activation_wts = min_value + (max_value - min_value) * random_wts
            if self.I['encoder_act'] == 'sigmoid':
                new_wts = torch.logit(post_activation_wts)
            elif self.I['encoder_act'] == 'tanh':
                # modified tanh activation function (torch.tanh(x)+1)/2
                new_wts = torch.arctanh(2 * post_activation_wts - 1)
            elif self.I['encoder_act'] == 'linear':
                new_wts = post_activation_wts
            elif self.I['encoder_act'] == 'relu':
                new_wts = torch.abs(post_activation_wts)
            else:
                raise ValueError(f"Invalid activation function: {self.I['encoder_act']}")
            self.encoder.weight.data = self.encoder.weight.data * (1 - mask) + new_wts * mask
            num_perturbed = mask.sum().item()
            self.log.info(f"Perturbed {num_perturbed} weights out of {self.encoder.weight.data.numel()} total weights ({num_perturbed/self.encoder.weight.data.numel()*100:.2f}%)")

    def fit(self):
        if self.X_train is None or self.y_train is None or self.constraints is None or self.decoder is None : 
            self.log.error("Model is not initialized. Call initialize() before fit().")
            raise RuntimeError("Model is not initialized. Call initialize() before fit().")
        # Initialize training state variables
        self.learning_stats = {} 
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1
        delayed_perturbation_iter = None
        start_time = time.time()
        last_report_time = start_time
        last_report_iteration = 0
        try:
            for iteration in range(self.I['n_iters']):
                self._update_parameters_for_iteration(iteration, self.I['n_iters'])
                self.learning_stats[str(iteration)] = {}
                if iteration == 0:
                    self._setup_optimizer()
                for param_group in self.optimizer_gen.param_groups: 
                    param_group['lr'] = self.I['lr']
                is_report_iter = (iteration % self.I['report_rt'] == 0) or (iteration == self.I['n_iters'] - 1) 
                self.train() 
                X_batch, y_batch = self._get_training_batch(iteration, self.I['batch_size'], self.X_train.shape[0])
                self.optimizer_gen.zero_grad() 
                total_loss, batch_stats = self.calculate_loss(
                    X_batch, y_batch, iteration, suffix='_train')
                self.learning_stats[str(iteration)].update(batch_stats)
                self.learning_stats[str(iteration)]['total_loss_train'] = total_loss.item()
                total_loss.backward() 
                nan_detected = self._check_gradient_health(iteration)
                if not nan_detected:
                    self._apply_gradient_clipping()
                    self.optimizer_gen.step()
                    delayed_perturbation_iter = self._handle_weight_perturbation(iteration, self.I['report_rt'], delayed_perturbation_iter)
                    self._update_best_model(iteration, total_loss)
                    self._save_model_checkpoint(iteration, is_report_iter)
                else: 
                    self._revert_to_previous_state(iteration)
                if is_report_iter:
                    last_report_time, last_report_iteration = self._evaluate_on_test_set(iteration, last_report_time, last_report_iteration)
                
                # Save checkpoint every 10k iterations
                if (iteration + 1) % 10000 == 0:
                    self._save_checkpoint(iteration)
                
                # Run evaluation and visualization every 50k iterations
                if (iteration + 1) % 50000 == 0:
                    self._save_eval_and_viz(iteration)
                
                if delayed_perturbation_iter == iteration:
                    self.log.info(f"Performing delayed weight perturbation at iteration {iteration}")
                    self.perturb_E()
                    delayed_perturbation_iter = None
                self._cleanup_old_checkpoints(iteration)
        except Exception as e:
            self.log.exception(f"Error during training loop at iteration {iteration}: {e}")
        finally:
            self._save_final_model(start_time)

    def evaluate(self):
        if self.encoder is None or self.decoder is None or \
           self.X_train is None or self.X_test is None or self.y_train is None or \
           self.y_test is None : 
            self.log.error("Cannot evaluate: Model not initialized or trained. Run initialize() and fit() first.")
            return
        self.results = {}
        # Use clean versions for evaluation stats to be consistent with loss calculations
        E = self.get_E_clean()
        E_cpu = E.cpu().detach()
        self.results['Number of Probes (Constrained)'] = E_cpu.sum().item()
        all_P_type = []
        X_global_train = self.X_train 
        y_global_train = self.y_train
        if X_global_train.shape[0] > 0:
            with torch.no_grad():
                P_global = self.project_clean(X_global_train, E) 
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
            # Calculate percentiles efficiently
            p10, p50, p90 = torch.quantile(avg_P_type, torch.tensor([0.1, 0.5, 0.9]))
            self.results['10th Percentile Signal'] = p10.item()
            self.results['50th Percentile Signal'] = p50.item()
            self.results['90th Percentile Signal'] = p90.item()
            for bit in range(avg_P_type.shape[1]):
                self.results[f"Number of Probes Bit {bit}"] = E_cpu[:, bit].sum().item()
                # Calculate percentiles for each bit efficiently
                bit_p10, bit_p50, bit_p90 = torch.quantile(avg_P_type[:, bit], torch.tensor([0.1, 0.5, 0.9]))
                self.results[f"10th Percentile Signal Bit {bit}"] = bit_p10.item()
                self.results[f"50th Percentile Signal Bit {bit}"] = bit_p50.item()
                self.results[f"90th Percentile Signal Bit {bit}"] = bit_p90.item()
        else:
            self.log.warning("Could not calculate average P_type for evaluation stats.")
        self.log.info("--- Basic Evaluation Stats ---")
        for key, val in self.results.items():
            if isinstance(val, (float, int)): log_msg = f" {key}: {round(val, 4)}"
            else: log_msg = f" {key}: {val}"
            self.log.info(log_msg)
        self.log.info("-----------------------------")
        noise_levels = {
            "No Noise": {
                'P_add': 0,
                'E_noise': 0,
                'P_noise': 0,
                'X_noise': 0,
                'X_drp': 0,
                'P_drp': 0,
                'E_drp': 0.0
            },
            "Low Noise": {
                'P_add': 2.0,
                'E_noise': 0.05,
                'P_noise': 0.05,
                'X_drp': 0.05,
                'X_noise': 0.05,
                'P_drp': 0,
                'E_drp': 0.05
            },
            "Medium Noise": {
                'P_add': 2.5,
                'E_noise': 0.25,
                'P_noise': 0.1,
                'X_drp': 0.1,
                'X_noise': 0.1,
                'P_drp': 0,
                'E_drp': 0.1
            },
            "High Noise": {
                'P_add': 3.0,
                'E_noise': 0.5,
                'P_noise': 0.2,
                'X_noise': 0.2,
                'X_drp': 0.2,
                'P_drp': 0,
                'E_drp': 0.2
            }
        }
        self.eval()
        for level_name, params in noise_levels.items():
            self.log.info(f"Calculating {level_name} Accuracy (Global)")
            try:
                # Store original parameters and training state
                original_params = {}
                original_training_state = self.training
                for param_name in params.keys():
                    original_params[param_name] = self.I[param_name]
                # Update parameters for this noise level
                for param_name, param_value in params.items():
                    self.I[param_name] = param_value
                # Temporarily set to training mode so noise is applied
                self.training = True
                with torch.no_grad():
                    with torch.no_grad():
                        _, stats = self.calculate_loss(self.X_test, self.y_test, iteration="Final", suffix='_test')
                        accuracy = stats.get('accuracy_test', np.nan)
                        separation = stats.get('separation_test', np.nan)
                        dynamic_range = stats.get('dynamic_range_test', np.nan)
                        self.log.info(f"{level_name} Accuracy: {round(accuracy, 4)}")
                        self.log.info(f"{level_name} Separation: {round(separation, 4)}")
                        self.log.info(f"{level_name} Dynamic Range: {round(dynamic_range, 4)}")
                        self.results[f'{level_name} Accuracy'] = accuracy
                        self.results[f'{level_name} Separation'] = separation
                        self.results[f'{level_name} Dynamic Range'] = dynamic_range
                    # Save P_test averages for "No Noise" condition
                    if level_name == "No Noise":
                        self.log.info("Saving P_test averages for No Noise condition...")
                        try:
                            E = self.get_E()
                            P_test = self.project(self.X_test, E)
                            P_test_cpu = P_test.cpu()
                            n_bits = P_test_cpu.shape[1]
                            P_type_test = torch.zeros((self.n_categories, n_bits), device='cpu')
                            unique_y_test = torch.unique(self.y_test)
                            valid_type_indices = []
                            valid_type_labels = []
                            for type_idx_tensor in unique_y_test:
                                type_idx = type_idx_tensor.item()
                                type_mask = (self.y_test == type_idx_tensor)
                                if type_mask.sum() > 0 and 0 <= type_idx < self.n_categories:
                                    P_type_test[type_idx] = P_test_cpu[type_mask].mean(dim=0)
                                    valid_type_indices.append(type_idx)
                                    valid_type_labels.append(self.y_reverse_label_map.get(int(type_idx), f"Type_{type_idx}"))
                            if valid_type_indices:
                                # Create DataFrame with cell types as index and bits as columns
                                # Apply log10 transformation and round to 3 decimal places
                                P_type_log10 = torch.log10(P_type_test[valid_type_indices].clamp(min=1e-10))
                                P_type_rounded = torch.round(P_type_log10 * 1000) / 1000  # Round to 3 decimal places
                                P_type_df = pd.DataFrame(
                                    P_type_rounded.numpy(),
                                    index=pd.Index(valid_type_labels),
                                    columns=pd.Index([f"Bit_{b}" for b in range(n_bits)])
                                )
                                p_type_path = os.path.join(self.I['output'], 'P_Type.csv')
                                P_type_df.to_csv(p_type_path)
                                self.log.info(f"P_test averages for No Noise condition saved to {p_type_path}")
                                self.log.info(f"Saved data for {len(valid_type_indices)} cell types across {n_bits} bits")
                            else:
                                self.log.warning("No valid cell types found for P_test averages")
                        except Exception as e:
                            self.log.error(f"Error saving P_test averages for No Noise condition: {e}")
                # Restore original parameters and training state
                for param_name, original_value in original_params.items():
                    self.I[param_name] = original_value
                self.training = original_training_state
            except Exception as e:
                self.log.error(f"Error during {level_name} accuracy calculation: {e}")
                self.results[f'{level_name} Accuracy'] = np.nan
                # Ensure training state is restored even if there's an error
                self.training = original_training_state
        # Set back to eval mode after all noise evaluations
        self.eval()
        results_df = pd.DataFrame({
            'values': list(self.results.values())
        }, index=pd.Index(list(self.results.keys())))
        results_path = os.path.join(self.I['output'], 'Results.csv') 
        results_df.to_csv(results_path)
        self.log.info(f"Evaluation results saved to {results_path}")

    def visualize(self, show_plots=False): 
        import seaborn as sns
        self.log.info("Starting visualization generation...")
        if self.encoder is None or self.decoder is None or \
           self.X_train is None or self.y_train is None or \
           self.y_reverse_label_map is None : 
            self.log.error("Cannot visualize: Model not initialized. Run initialize() and fit() first.")
            return
        saved_plot_paths = []
        E = self.get_E_clean()
        self.eval()
        self.log.info(f"Generating visualization ...")
        X_data_vis = self.X_train # Use full training data
        y_data_vis = self.y_train # Mapped internal labels
        y_vis_str_labels = np.array([self.y_reverse_label_map.get(int
                    (idx.item()), f"Type_{idx.item()}") for idx in y_data_vis])
        if X_data_vis.shape[0] == 0:
            self.log.warning(f"Skipping visualization: No training data found.")
            return
        with torch.no_grad():
            self.log.info(f"Generating confusion matrix for test data (Global)...")
            X_test_global = self.X_test
            y_test_global = self.y_test # True internal labels
            fig_cm = None 
            try:
                with torch.no_grad():
                    P_test_tensor = self.project_clean(X_test_global, E)
                    y_pred_test, _, _ = self.decode(P_test_tensor, y_test_global)
                y_true_np = y_test_global.cpu().numpy()
                y_pred_np = y_pred_test.cpu().numpy()
                present_labels_idx = np.unique(np.concatenate((y_true_np, y_pred_np)))
                present_labels_str = [self.y_reverse_label_map.get(i, f"Type_{i}") for i in present_labels_idx]
                if len(present_labels_idx) > 0: 
                    cm = confusion_matrix(y_true_np, y_pred_np, labels=present_labels_idx) 
                    cm_sum = cm.sum(axis=1, keepdims=True)
                    cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) 
                    cm_df = pd.DataFrame(cm_norm, index=pd.Index(present_labels_str), columns=pd.Index(present_labels_str))
                    fig_width = min(max(10, len(present_labels_str) / 1.5), 25)
                    fig_height = min(max(8, len(present_labels_str) / 2), 25)
                    fig_cm = plt.figure(figsize=(fig_width, fig_height))
                    ax_cm = fig_cm.add_subplot(111)
                    sns.heatmap(cm_df, annot=False, cmap='jet', linewidths=0.1, ax=ax_cm, vmin=0, vmax=1) 
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('True')
                    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right')
                    plt.setp(ax_cm.get_yticklabels(), rotation=0)
                    fig_cm.tight_layout()
                    plot_filename = f"confusion_matrix_test.pdf"
                    plot_path = os.path.join(self.I['output'], plot_filename)
                    fig_cm.savefig(plot_path, dpi=300, bbox_inches='tight')
                    saved_plot_paths.append(plot_path)
                    self.log.info(f"Saved Test Confusion Matrix to {plot_path}")
                else:
                    self.log.warning(f"Skipping confusion matrix: No labels found in true or predicted test data.")
            except Exception as e:
                self.log.error(f"Error generating Test Confusion Matrix: {e}")
            finally:
                if fig_cm is not None:
                    plt.close(fig_cm)
        # Generate learning curves
        learning_curve = pd.DataFrame.from_dict(self.learning_stats, orient='index')
        unique_parameters = np.unique([i.replace('_test','').replace('_train','') for i in learning_curve.columns])
        unique_parameters = np.array([i for i in unique_parameters if (i+'_train' in learning_curve.columns) and (i+'_test' in learning_curve.columns)])
        numeric_parameters = np.array([i for i in unique_parameters if not isinstance(learning_curve[i+'_train'].iloc[1],str)])
        for parameter in numeric_parameters:
            try:
                plot_single_learning_curve(parameter, learning_curve, self.I['output'], self.log)
            except Exception as e:
                self.log.error(f"Error plotting learning curve for {parameter}: {e}")
        # Generate loss contributions plot
        try:
            plot_loss_contributions(learning_curve, self.I['output'], self.log)
        except Exception as e:
            self.log.error(f"Error plotting loss contributions: {e}")
        # Generate comprehensive performance plot
        try:
            plot_comprehensive_performance(learning_curve, self.I['output'], self.log)
        except Exception as e:
            self.log.error(f"Error plotting comprehensive performance: {e}")
        
        E = self.get_E_clean()
        E = E.to(self.I['device'])
        self.eval()
        # --- Visualization: Constraints vs Probes per Gene ---
        try:
            probes_per_gene = E.sum(dim=1).detach().cpu().numpy()
            constraints = self.constraints.detach().cpu().numpy()
            plt.figure(figsize=(6, 4))
            plt.scatter(constraints, probes_per_gene, s=2, alpha=0.5)
            # Add y=x line
            min_val = min(constraints.min(), probes_per_gene.min())
            max_val = max(constraints.max(), probes_per_gene.max())
            plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=1, label='y=x')
            plt.xlabel('Constraint (max probes per gene)')
            plt.ylabel('Probes per gene (E_clean)')
            plt.title('Constraints vs Probes per Gene')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.I['output'], 'constraints_vs_probes_per_gene.pdf'), dpi=200)
            plt.close()
        except Exception as e:
            self.log.error(f"Failed to plot constraints vs probes per gene: {e}")

        # --- Visualization: Clustermap of E_clean (genes clustered, no labels/dendrogram) ---
        try:
            import seaborn as sns
            # Convert E_clean to DataFrame for easier manipulation
            E_clean_np = E.detach().cpu().numpy()
            used_WeightMat = pd.DataFrame(E_clean_np)

            # Find the highest bit (column) for each gene (row)
            highest_bit = used_WeightMat.columns[np.argmax(used_WeightMat.values, axis=1)]
            gene_order = []
            for i in sorted(np.unique(highest_bit)):
                idx = used_WeightMat.index[np.where(highest_bit == i)[0]]
                vals = used_WeightMat.loc[idx, i]
                gene_order.extend(list(pd.DataFrame(vals).sort_values(by=i).index))

            # Reorder the DataFrame based on the sorted genes
            reordered_df = used_WeightMat.loc[gene_order].copy()
            reordered_df.columns = [int(i)+1 for i in reordered_df.columns]

            # Plot the clustermap (no clustering, just your order)
            plt.figure(figsize=(10, 10))
            sns.clustermap(
                np.log10(reordered_df + 1),
                cmap='Reds',
                col_cluster=False,
                row_cluster=False,
                figsize=(10, 10),
                yticklabels=[]
            )
            plt.savefig(os.path.join(self.I['output'], 'E.pdf'), dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.log.error(f"Failed to plot E_clean clustermap: {e}")

        with torch.no_grad():
            P = self.project_clean(X_data_vis, E)
            for normalization_strategy in ['Raw', 'Sum Norm', 'Bit Center', 'Bit Z-score', 'Sum and Bit Center', 'Sum and Bit Z-score']:
                P_norm = P.clone()
                if 'Sum' in normalization_strategy:
                    P_norm = sum_normalize_p_type(P_norm)
                if 'Bit Center' in normalization_strategy:
                    P_norm = bitwise_center_p_type(P_norm)
                if 'Bit Z-score' in normalization_strategy:
                    P_norm = bitwise_normalize_p_type(P_norm)
                n_bits = P.shape[1]
                P_type_global = torch.zeros((self.n_categories, n_bits), device=self.I['device'])
                unique_y_indices_global = torch.unique(y_data_vis)
                valid_type_indices = []
                valid_type_labels = []
                for type_idx_tensor in unique_y_indices_global:
                    type_idx = type_idx_tensor.item() 
                    mask = (y_data_vis == type_idx_tensor)
                    if mask.sum() > 0:
                        if 0 <= type_idx < self.n_categories:
                            P_type_global[type_idx] = P_norm[mask].mean(dim=0) 
                            valid_type_indices.append(type_idx)
                            valid_type_labels.append(self.y_reverse_label_map.get(int(type_idx), f"Type_{type_idx}"))
                        else:
                            self.log.warning(f"Skipping type index {type_idx} during P_type calculation (out of bounds).")
                if not valid_type_indices:
                    self.log.warning(f"Skipping visualization: No valid cell types found after projection.")
                    return
                P_type_global_present = P_type_global[valid_type_indices].cpu() 
                n_types_present = P_type_global_present.shape[0]
                if n_types_present > 0:
                    plot_P_Type(P_type_global_present, valid_type_labels, os.path.join(self.I['output'], f"P_type_{normalization_strategy}.pdf"), self.log)
                    plot_P_Type_correlation(P_type_global_present, valid_type_labels, os.path.join(self.I['output'], f"P_type_correlation_{normalization_strategy}.pdf"), self.log)
                    plot_projection_space_density(P_norm.cpu().numpy(), y_vis_str_labels, os.path.join(self.I['output'], f"projection_density_{normalization_strategy}.pdf"), sum_norm=False, log=self.log, use_log10_scale=False)
            

        self.log.info("Visualization generation finished.")

    def _setup_logging(self, user_parameters_path):
        if user_parameters_path is not None:
            try:
                df_temp = pd.read_csv(user_parameters_path, index_col=0, low_memory=False)
                if 'values' in df_temp.columns:
                    loaded_params_temp = dict(zip(df_temp.index, df_temp['values']))
                    self.I['output'] = loaded_params_temp.get('output', self.I['output'])
            except (FileNotFoundError, Exception):
                pass
        if not os.path.exists(self.I['output']):
            os.makedirs(self.I['output'])
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO) 
        input_filename = os.path.basename(user_parameters_path) if user_parameters_path else "default"
        input_filename = os.path.splitext(input_filename)[0]  
        self.log_file = os.path.join(self.I['output'], f'log_{input_filename}.log')
        logging.basicConfig(
            filename=self.log_file, filemode='a',
            format='%(message)s            |||| %(asctime)s %(name)s %(levelname)s',
            datefmt='%Y %B %d %H:%M:%S', level=logging.INFO, force=True)
        self.log = logging.getLogger("Designer")

    def _load_and_process_parameters(self, user_parameters_path):
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
        # Add loaded user parameters to default parameters
        for key, val in loaded_user_parameters.items():
            if key in self.I:
                self.I[key] = val 
            else:
                self.log.warning(f"Parameter '{key}' from file is not a default parameter. Adding it.")
                self.I[key] = val 
            if key.endswith('_s'):
                new_key = key.replace('_s', '')
                if new_key not in self.I:
                    self.I[new_key] = val
                    self.log.info(f"Created parameter '{new_key}' from '{key}' with value {val}")
                else:
                    self.log.info(f"Parameter '{new_key}' already exists. Skipping creation.")
        # Construct paths for input files
        input_dir = self.I['input']
        file_params_to_prefix = ['constraints', 'X_test', 'y_test', 'X_train', 'y_train', 'y_label_converter_path']
        for param_key in file_params_to_prefix:
            current_path = self.I[param_key]
            if current_path and not os.path.dirname(current_path) and not os.path.isabs(current_path):
                self.I[param_key] = os.path.join(input_dir, current_path)
                self.log.info(f"Constructed path for '{param_key}': {self.I[param_key]}")
        # Convert parameters to integers
        params_to_int = ['n_bit', 'n_iters', 'report_rt', 'batch_size', 'n_cpu',
                         'decoder_n_lyr'] 
        for param_key in params_to_int:
            self.convert_param_to_int(param_key) 
        # Log final parameters
        self.log.info(f"Final Parameters (after path construction & type conversion):")
        for key, val in self.I.items():
            self.log.info(f"{key}: {val} (type: {type(val).__name__})") 
        self.log.info(f"Limiting Torch to {self.I['n_cpu']} threads")
        torch.set_num_threads(self.I['n_cpu'])

    def _setup_output_and_symlinks(self, user_parameters_path):
        if not os.path.exists(self.I['output']):
            os.makedirs(self.I['output'])
            self.log.info(f"Created output directory: {self.I['output']}")
        param_df = pd.DataFrame({
            'values': list(self.I.values())
        }, index=pd.Index(list(self.I.keys())))
        param_df.to_csv(os.path.join(self.I['output'], 'used_user_parameters.csv'))
        # Create symlinks for input files
        self.log.info("Creating symlinks for input files in output directory...")
        input_files_to_link = []
        file_params_to_prefix = ['constraints', 'X_test', 'y_test', 'X_train', 'y_train', 'y_label_converter_path']
        for key in file_params_to_prefix:
            path = self.I.get(key)
            if isinstance(path, str) and path: 
                input_files_to_link.append(path)
        # Add user parameter file and used parameters file to symlinks
        if user_parameters_path is not None and isinstance(user_parameters_path, str):
            input_files_to_link.append(user_parameters_path)
            input_files_to_link.append(os.path.join(self.I['output'], 'used_user_parameters.csv'))
        # Create symlinks
        linked_count = 0
        skipped_count = 0
        error_count = 0
        for input_path in set(input_files_to_link): 
            try:
                abs_input_path = os.path.abspath(input_path)
                if not os.path.exists(abs_input_path):
                    self.log.warning(f"Input file not found, cannot create symlink: \n {abs_input_path}")
                    skipped_count += 1
                    continue
                filename = os.path.basename(abs_input_path)
                symlink_path = os.path.join(self.I['output'], filename)
                if os.path.lexists(symlink_path): 
                    if os.path.islink(symlink_path):
                        self.log.debug(f"Removing existing symlink: \n {symlink_path}")
                        os.remove(symlink_path)
                    else:
                        self.log.warning(f"Target path exists but is not a symlink, skipping: \n {symlink_path}")
                        skipped_count += 1
                        continue
                os.symlink(abs_input_path, symlink_path)
                self.log.info(f"Created symlink: \n {symlink_path} \n -> \n {abs_input_path}")
                linked_count += 1
            except OSError as e:
                self.log.error(f"Failed to create symlink for \n {input_path} \n -> \n {symlink_path}: {e}")
                error_count += 1
            except Exception as e:
                self.log.error(f"An unexpected error occurred while trying to symlink \n {input_path}: {e}")
                error_count += 1
        self.log.info(f"Symlinking complete. Created: {linked_count}, Skipped: {skipped_count}, Errors: {error_count}")

    def convert_param_to_int(self, param_key):
        try:
            original_value = self.I[param_key]
            float_value = float(original_value)
            if float_value.is_integer():
                self.I[param_key] = int(float_value)
            else:
                raise ValueError(f"Value '{original_value}' is not a whole number.")
        except KeyError:
            self.log.error(f"Parameter '{param_key}' not found for integer conversion.")
            raise KeyError(f"Required parameter '{param_key}' is missing.")
        except (ValueError, TypeError) as e:
            self.log.error(f"Error converting parameter '{param_key}' to int. Value was '{original_value}' (type: {type(original_value).__name__}). Error: {e}")
            raise ValueError(f"Could not convert parameter '{param_key}' to integer. Invalid value: '{original_value}'.")

    def _initialize_encoder(self):
        """Initialize the encoder with proper weight initialization."""
        self.encoder = nn.Embedding(self.n_genes, self.I['n_bit']).to(self.I['device'])
        with torch.no_grad():
            random_wts = torch.rand_like(self.encoder.weight.data)
            min_value = torch.tensor(self.I['E_init_min'], dtype=random_wts.dtype, device=self.I['device'])
            max_value = torch.tensor(self.I['E_init_max'], dtype=random_wts.dtype, device=self.I['device'])
            post_activation_wts = min_value + (max_value - min_value) * random_wts
            # Apply activation function
            if self.I['encoder_act'] == 'sigmoid':
                final_wts = torch.logit(post_activation_wts)
            elif self.I['encoder_act'] == 'tanh':
                # modified tanh activation function (torch.tanh(x)+1)/2
                final_wts = torch.arctanh(2 * post_activation_wts - 1)
            elif self.I['encoder_act'] == 'linear':
                final_wts = post_activation_wts
            elif self.I['encoder_act'] == 'relu':
                final_wts = torch.abs(post_activation_wts)
            else:
                raise ValueError(f"Invalid activation function: {self.I['encoder_act']}")
            # Set encoder weights
            self.encoder.weight.data = final_wts
            final_total_probes = (post_activation_wts * self.constraints.unsqueeze(1)).sum()
            self.log.info(f"Encoder initialization: range [{final_wts.min().item():.3f}, {final_wts.max().item():.3f}], total probes: {final_total_probes.item():.1f}")
        self.log.info(f"Initialized encoder with improved weight initialization.")

    def _initialize_decoder(self):
        """Initialize the decoder with specified architecture."""
        decoder_modules = []
        current_decoder_layer_input_dim = self.I['n_bit']
        if self.I['decoder_n_lyr'] == 0:
            decoder_modules.append(nn.Linear(current_decoder_layer_input_dim, self.n_categories))
            log_msg_decoder_structure = "Initialized single linear decoder."
        else:
            decoder_h_dim = int(3 * self.I['n_bit'])  # Hidden dimension is 3x the number of bits
            for i in range(self.I['decoder_n_lyr']):
                decoder_modules.append(nn.Linear(current_decoder_layer_input_dim, decoder_h_dim))
                # Add batch normalization for better training stability
                decoder_modules.append(nn.BatchNorm1d(decoder_h_dim))
                # Add activation function based on decoder_activation parameter
                if self.I['decoder_act'] == 'relu':
                    decoder_modules.append(nn.ReLU())
                elif self.I['decoder_act'] == 'leaky_relu':
                    decoder_modules.append(nn.LeakyReLU())
                elif self.I['decoder_act'] == 'gelu':
                    decoder_modules.append(nn.GELU())
                elif self.I['decoder_act'] == 'swish':
                    decoder_modules.append(nn.SiLU())  # SiLU is the same as Swish
                elif self.I['decoder_act'] == 'tanh':
                    decoder_modules.append(nn.Tanh())
                else:
                    raise ValueError(f"Invalid decoder activation function: {self.I['decoder_act']}")
                # Only add dropout if use_noise is enabled
                if self.I['use_noise'] == 1:
                    decoder_modules.append(nn.Dropout(p=self.I['D_drp']))
                current_decoder_layer_input_dim = decoder_h_dim 
            decoder_modules.append(nn.Linear(current_decoder_layer_input_dim, self.n_categories))
            dropout_info = f"dropout={self.I['D_drp']}" if self.I['use_noise'] == 1 else "no dropout"
            log_msg_decoder_structure = f"Initialized decoder with {self.I['decoder_n_lyr']} hidden layer(s) (dim={decoder_h_dim}, activation={self.I['decoder_act']}, {dropout_info}) and output layer."
        self.decoder = nn.Sequential(*decoder_modules).to(self.I['device'])
        self.log.info(f"Initialized decoder.")
        self.log.info(log_msg_decoder_structure)

    def _load_data(self):
        """Load and validate all data files."""
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
        self.X_train = load_tensor(self.I['X_train'], torch.float32, self.I['device'])
        self.X_test = load_tensor(self.I['X_test'], torch.float32, self.I['device'])
        self.y_train = load_tensor(self.I['y_train'], torch.long, self.I['device'])
        self.y_test = load_tensor(self.I['y_test'], torch.long, self.I['device'])
        # Process labels
        all_y_labels = torch.cat((self.y_train, self.y_test))
        self.updated_y_label_map = {label.item(): i for i, label in enumerate(torch.unique(all_y_labels))}
        self.y_train = torch.tensor([self.updated_y_label_map[y.item()] for y in self.y_train], dtype=torch.long, device=self.I['device'])
        self.y_test = torch.tensor([self.updated_y_label_map[y.item()] for y in self.y_test], dtype=torch.long, device=self.I['device'])
        # Load label converter
        y_converter_path = self.I['y_label_converter_path']
        self.log.info(f"Loading y label converter from: {y_converter_path}")
        y_converter_df = pd.read_csv(y_converter_path, index_col=0) 
        y_converter_dict = dict(zip(y_converter_df.index, y_converter_df['label'])) 
        self.y_label_map = {k: self.updated_y_label_map[j] for k, j in y_converter_dict.items()}
        self.y_reverse_label_map = {j: k for k, j in self.y_label_map.items()}
        # Determine number of categories
        all_y_labels_for_n_categories = torch.cat((self.y_train, self.y_test)) 
        unique_y_labels_tensor = torch.unique(all_y_labels_for_n_categories)
        self.n_categories = len(unique_y_labels_tensor)
        self.mapped_category_indices = list(range(self.n_categories))
        # Validate data shapes
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

    def _load_constraints(self):
        """Load gene constraints from file."""
        constraints_df = pd.read_csv(self.I['constraints'], index_col=0)
        self.genes = np.array(constraints_df.index)
        self.n_genes = len(self.genes)  # Set n_genes first
        
        if 'constraints' not in constraints_df.columns:
            raise KeyError(f"Column 'constraints' not found in {self.I['constraints']}")
        
        # If gene_constraint_wt is 0, set all constraints to 1000
        if self.I['gene_constraint_wt'] == 0:
            self.constraints = torch.full((self.n_genes,), 1000.0, dtype=torch.float32, device=self.I['device'])
            self.log.info(f"Gene constraint weight is 0. Setting all {self.n_genes} gene constraints to 1000.")
        else:
            self.constraints = torch.tensor(constraints_df['constraints'].values, dtype=torch.float32, device=self.I['device'])
            self.log.info(f"Loaded {self.n_genes} genes from constraints file.")

    def _load_pretrained_model(self):
        """Load pretrained model state, learning stats, and gene mask if available."""
        model_state_path = os.path.join(self.I['output'], 'final_model_state.pt')
        learning_stats_path = os.path.join(self.I['output'], 'learning_stats.json')
        if not os.path.exists(model_state_path):
            self.log.info(f"No existing model state file found at {model_state_path}. Model will use fresh initial weights.")
            self.is_initialized_from_file = False
            self.training_completed = False
            return
        self.log.info(f"Found existing model state file: {model_state_path}. Attempting to load.")
        try:
            loaded_state_dict = torch.load(model_state_path, map_location=self.I['device'])
            missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=False)
            if missing_keys: self.log.warning(f"Missing keys when loading state_dict: {missing_keys}")
            if unexpected_keys: self.log.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
            self.to(self.I['device']) 
            self.is_initialized_from_file = True
            self.log.info("Successfully loaded model state from file (strict=False).")
            if os.path.exists(learning_stats_path):
                try:
                    import json
                    with open(learning_stats_path, 'r') as f:
                        loaded_learning_stats = json.load(f)
                    self.learning_stats = loaded_learning_stats
                    self.log.info(f"Successfully loaded learning stats from {learning_stats_path}")
                    max_iteration = -1
                    for key in self.learning_stats.keys():
                        if key.isdigit():
                            max_iteration = max(max_iteration, int(key))
                    if max_iteration >= self.I['n_iters'] - 1:
                        self.training_completed = True
                        self.log.info(f"Training appears to be completed (max iteration: {max_iteration}, target: {self.I['n_iters']})")
                    else:
                        self.training_completed = False
                        self.log.info(f"Training appears to be incomplete (max iteration: {max_iteration}, target: {self.I['n_iters']})")
                except Exception as e:
                    self.log.error(f"Failed to load learning stats from {learning_stats_path}: {e}")
                    self.training_completed = False
            else:
                self.log.info("No learning stats file found. Assuming training was not completed.")
                self.training_completed = False
            # Check if gene mask exists and load it if present
            gene_mask_path = os.path.join(self.I['output'], 'gene_mask.pt')
            if os.path.exists(gene_mask_path):
                self.log.info(f"Found gene mask file: {gene_mask_path}. Loading gene filtering.")
                try:
                    gene_mask = torch.load(gene_mask_path, map_location=self.I['device'])
                    if gene_mask.dtype != torch.bool:
                        gene_mask = gene_mask.bool()
                    self.X_train = self.X_train[:, gene_mask]
                    self.X_test = self.X_test[:, gene_mask]
                    self.constraints = self.constraints[gene_mask]
                    self.genes = self.genes[gene_mask.cpu().numpy()]
                    self.n_genes = gene_mask.sum().item()
                    self.log.info(f"Applied gene mask: {self.n_genes} genes retained from {len(gene_mask)} total genes")
                    # Reinitialize encoder with correct dimensions
                    self._initialize_encoder()
                    self.log.info(f"Reinitialized encoder with {self.n_genes} genes")
                except Exception as e:
                    self.log.error(f"Failed to load gene mask from {gene_mask_path}: {e}")
                    self.log.warning("Continuing without gene filtering.")
            else:
                self.log.info("No gene mask file found. Using all genes.")
            self.eval() 
            with torch.no_grad():
                final_E = self.get_E_clean().detach().clone()
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

    def _update_parameters_for_iteration(self, iteration, n_iterations):
        """Update parameters based on training progress."""
        progress = iteration / (n_iterations - 1) if n_iterations > 1 else 0
        parameters_to_update = [i.replace('_s', '') for i in self.I if i.endswith('_s')]
        for param in parameters_to_update:
            self.I[param] = (self.I[f'{param}_s'] + (self.I[f'{param}_e'] - self.I[f'{param}_s']) * progress)

    def _setup_optimizer(self):
        """Setup optimizer for training."""
        if not self.is_initialized_from_file:
            self.log.info("Model not initialized from file, using randomly initialized weights.")
        else:
            self.log.info("Using model loaded during initialization.")
        self.to(self.I['device'])
        optimizer_gen = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.I['lr']},
            {'params': self.decoder.parameters(), 'lr': self.I['lr']}
        ], betas=(0.9, 0.9))
        self.optimizer_gen = optimizer_gen

    def _get_training_batch(self, iteration, batch_size, n_train_samples):
        """Get training batch for current iteration."""
        if (batch_size > 0) and (batch_size < n_train_samples):
            idxs = np.random.choice(n_train_samples, batch_size, replace=False)
            X_batch = self.X_train[idxs]
            y_batch = self.y_train[idxs]
        else:  
            X_batch = self.X_train
            y_batch = self.y_train
            if batch_size > 0:  
                self.log.debug(f"Batch size {batch_size} >= dataset size {n_train_samples}. Using full dataset for iteration {iteration}.")
        return X_batch, y_batch

    def _check_gradient_health(self, iteration):
        """Check for NaN or Inf values in gradients."""
        nan_detected = False
        for name, param in self.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                nan_detected = True
                self.log.warning(f"NaNs or Infs detected in gradients of model parameter '{name}' at iteration {iteration}. Skipping step and attempting revert.")
                self.optimizer_gen.zero_grad() 
                break
        return nan_detected

    def _apply_gradient_clipping(self):
        """Apply gradient clipping if enabled."""
        max_norm_value = self.I.get('gradient_clip', 1.0) 
        if max_norm_value > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm_value)

    def _handle_weight_perturbation(self, iteration, report_rt, delayed_perturbation_iter):
        """Handle weight perturbation logic."""
        if self.I['E_perturb_rt'] > 0:
            if iteration % self.I['E_perturb_rt'] == 0:
                next_test_iter = ((iteration // report_rt) + 1) * report_rt
                if next_test_iter - iteration <= 50:
                    self.log.info(f"Delaying weight perturbation from iteration {iteration} to after test at iteration {next_test_iter}")
                    delayed_perturbation_iter = next_test_iter
                else:
                    self.perturb_E()
        return delayed_perturbation_iter

    def _update_best_model(self, iteration, total_loss):
        """Update best model if current loss is better."""
        current_loss_item = total_loss.item()
        if not np.isnan(current_loss_item) and current_loss_item < self.best_loss:
            self.best_loss = current_loss_item
            self.best_model_state_dict = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
            self.best_iteration = iteration
            self.log.info(f"*** New best model found at iteration {iteration} (Train Loss: {self.best_loss:.4f}) ***")

    def _save_model_checkpoint(self, iteration, is_report_iter):
        """Save model checkpoint if needed."""
        if is_report_iter or iteration == self.best_iteration:  
            self.saved_models[iteration] = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
            self.saved_optimizer_states[iteration] = self.optimizer_gen.state_dict()

    def _revert_to_previous_state(self, iteration):
        """Revert to previous model state when NaN/Inf detected."""
        valid_iters = [k for k in self.saved_models if k < iteration]
        if valid_iters:
            revert_iter = max(valid_iters)
            self.log.warning(f"Reverting model and optimizer to state from iteration {revert_iter}")
            try:
                self.load_state_dict(self.saved_models[revert_iter])
                self.to(self.I['device'])
                optimizer_gen = torch.optim.Adam([ # Re-init optimizer for the reverted state
                    {'params': self.encoder.parameters(), 'lr': self.I['lr']},
                    {'params': self.decoder.parameters(), 'lr': self.I['lr']}
                ])
                optimizer_gen.load_state_dict(self.saved_optimizer_states[revert_iter])
                self.optimizer_gen = optimizer_gen
                for state in self.optimizer_gen.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.I['device'])
            except Exception as e:
                self.log.error(f"Failed to load state from iter {revert_iter}: {e}. Optimizer state might be reset.")
                self.optimizer_gen = torch.optim.Adam([
                    {'params': self.encoder.parameters(), 'lr': self.I['lr']},
                    {'params': self.decoder.parameters(), 'lr': self.I['lr']}
                ])
            self.learning_stats[str(iteration)] = {} 
            self.learning_stats[str(iteration)]['status'] = f'Reverted from NaN at {iteration}'
            return True
        else:
            self.log.error(f"NaNs/Infs detected in gradients at iter {iteration}, but no previous state found. Stopping.")
            raise ValueError("NaNs/Infs encountered and cannot revert.")

    def _evaluate_on_test_set(self, iteration, last_report_time, last_report_iteration):
        """Evaluate model on test set and log results."""
        self.eval()
        with torch.no_grad():
            total_test_loss, test_stats = self.calculate_loss(
                self.X_test, self.y_test, iteration, suffix='_test')
            test_stats['total_loss_test'] = total_test_loss.item()
            self.learning_stats[str(iteration)].update(test_stats)
        current_time = time.time()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = current_time - last_report_time
        iterations_since_last = iteration - last_report_iteration + 1 
        avg_iter_time = elapsed_time / iterations_since_last if iterations_since_last > 0 else 0
        self.log.info(f"Avg time/iter since last report: {avg_iter_time:.4f} seconds")
        red_start = "\033[91m"; reset_color = "\033[0m"
        log_msg_header = f"--- Iteration: {iteration}/{self.I['n_iters']} Eval (Global Test Set) ---"
        self.log.info(log_msg_header)
        log_msg_lr = f"Current LR: {self.I['lr']:.6f}"
        self.log.info(log_msg_lr)
        train_loss_key = 'total_loss_train' 
        if train_loss_key in self.learning_stats[str(iteration)]:
            log_msg = f'{train_loss_key}: {round(self.learning_stats[str(iteration)][train_loss_key], 4)}'
            self.log.info(log_msg)
        for name, item in test_stats.items():
            if isinstance(item, (float, int, np.number)) and not np.isnan(item):
                log_msg = f'{name}: {round(float(item), 4)}'
            else:
                log_msg = f'{name}: {item}'
            self.log.info(log_msg)
        self.log.info('------------------')
        
        return current_time, iteration

    def _save_checkpoint(self, iteration):
        """Save a checkpoint with model state, optimizer state, and learning stats."""
        try:
            checkpoint_dir = os.path.join(self.I['output'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
            learning_stats_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}_stats.json')
            
            # Save model state
            torch.save({
                'iteration': iteration,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer_gen.state_dict(),
                'best_loss': self.best_loss,
                'best_iteration': self.best_iteration,
                'parameters': self.I
            }, checkpoint_path)
            
            # Save learning stats
            import json
            with open(learning_stats_path, 'w') as f:
                json.dump(self.learning_stats, f, indent=2)
            
            self.log.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")
            
        except Exception as e:
            self.log.error(f"Failed to save checkpoint at iteration {iteration}: {e}")

    def _save_eval_and_viz(self, iteration):
        """Save evaluation results and visualizations at specific iteration."""
        try:
            eval_dir = os.path.join(self.I['output'], 'eval_checkpoints')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Create iteration-specific output directory
            iter_output_dir = os.path.join(eval_dir, f'iter_{iteration}')
            os.makedirs(iter_output_dir, exist_ok=True)
            
            # Temporarily change output directory
            original_output = self.I['output']
            self.I['output'] = iter_output_dir
            
            self.log.info(f"Running evaluation and visualization for iteration {iteration}")
            
            # Run evaluation
            self.evaluate()
            
            # Run visualization
            self.visualize(show_plots=False)
            
            # Restore original output directory
            self.I['output'] = original_output
            
            self.log.info(f"Saved evaluation and visualization for iteration {iteration} to {iter_output_dir}")
            
        except Exception as e:
            self.log.error(f"Failed to save evaluation and visualization for iteration {iteration}: {e}")
            # Restore original output directory even if there's an error
            self.I['output'] = original_output

    def _cleanup_old_checkpoints(self, iteration):
        """Remove old model checkpoints to save memory."""
        if iteration > 20:
            keys_to_delete = sorted([k for k in self.saved_models if k < iteration - 20 and k != 0 and k != self.best_iteration])
            for key_to_del in keys_to_delete:
                self.saved_models.pop(key_to_del, None)
                self.saved_optimizer_states.pop(key_to_del, None)

    def _save_final_model(self, start_time):
        """Save final model and perform cleanup."""
        if self.I['best_model'] == 0:
            self.best_model_state_dict = None
            self.log.info("Best model turned off. Saving the final iteration state.")
        
        if self.best_model_state_dict is not None:
            self.log.info(f"Loading best model state from iteration {self.best_iteration} (Train Loss: {self.best_loss:.4f}) before final save.")
            try:
                missing_keys, unexpected_keys = self.load_state_dict(self.best_model_state_dict, strict=False)
                if missing_keys: self.log.warning(f"Missing keys when loading best state_dict: {missing_keys}")
                if unexpected_keys: self.log.warning(f"Unexpected keys when loading best state_dict: {unexpected_keys}")
                self.to(self.I['device'])
                self.log.info(f"Successfully loaded best model state for final saving.")
            except Exception as e:
                self.log.error(f"Failed to load best model state before saving: {e}. Saving the final iteration state instead.")
        else:
            self.log.warning("No best model state was saved during training. Saving the final iteration state.")
        final_model_path = os.path.join(self.I['output'], 'final_model_state.pt')
        learning_stats_path = os.path.join(self.I['output'], 'learning_stats.json')
        try:
            torch.save(self.state_dict(), final_model_path)
            self.log.info(f"Final model state dictionary saved to {final_model_path}")
            import json
            with open(learning_stats_path, 'w') as f:
                json.dump(self.learning_stats, f, indent=2)
            self.log.info(f"Learning stats saved to {learning_stats_path}")
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
        for name, item in self.learning_stats[final_iter_key].items():
            if isinstance(item, (float, int, np.number)) and not np.isnan(item):
                log_msg = f'{name}: {round(float(item), 4)}'
            else:
                log_msg = f'{name}: {item}'
            self.log.info(log_msg)
        self.log.info('------------------')
        self.log.info('Total time taken: {:.2f} seconds'.format(time.time() - start_time))
        self.eval() 
        with torch.no_grad():
            E = self.get_E_clean().detach().clone() 
        self.log.info("Enforcing constraints on the final E matrix...")
        if self.constraints is None:
            self.log.error("Cannot enforce constraints: self.constraints is None.")
            self.E = E 
        else:
            E_final_constrained = torch.clip(E.round(), 0, None)
            T = self.constraints.clone().detach()
            m = E_final_constrained.sum(1) > T
            if m.any():
                scaling_factors = (T[m] / E_final_constrained.sum(1)[m].clamp(min=1e-8)).unsqueeze(1)
                E_final_constrained[m, :] = (E_final_constrained[m, :] * scaling_factors).floor()
                E_final_constrained = E_final_constrained.clamp(min=0)
            self.E = E_final_constrained.clone().detach() 
            e_csv_path = os.path.join(self.I['output'], 'E_constrained.csv')
            e_pt_path = os.path.join(self.I['output'], 'E_constrained.pt')
            pd.DataFrame(self.E.cpu().numpy(), index=self.genes).to_csv(e_csv_path)
            torch.save(self.E.cpu(), e_pt_path)
            self.log.info(f"Final constrained E matrix saved to {e_csv_path} and {e_pt_path}")
            self.log.info(f"Final constrained probe count: {self.E.sum().item():.2f}")
        try:
            learning_df = pd.DataFrame.from_dict(self.learning_stats, orient='index')
            learning_curve_path = os.path.join(self.I['output'], 'learning_curve.csv')
            learning_df.to_csv(learning_curve_path)
            self.log.info(f"Learning curve data saved to {learning_curve_path}")
        except Exception as e:
            self.log.error(f"Failed to save learning curve: {e}")

# Helper function for smooth, non-negative penalty
def swish(fold, scale=3.0, shift=-1.278, offset=0.3):
    """
    Smooth, non-negative penalty function for soft constraints.
    Args:
        fold: Tensor of normalized constraint violations (e.g., (value - bound) / bound)
        scale: Controls sharpness of penalty ramp-up
        shift: Shifts minimum to zero (default -1.278 for Swish)
        offset: Makes minimum value zero (default 0.278 for Swish)
    Returns:
        penalty: Tensor of penalties, minimum at fold=0, non-negative
    """
    z = scale * fold + shift
    return z * torch.sigmoid(z) + offset

def sanitize_filename(name):
    """Removes or replaces characters invalid for filenames."""
    name = name.strip()
    name = re.sub(r'[\s/\\:]+', '_', name)
    name = re.sub(r'[<>:"|?*]+', '', name)
    return name

def generate_intuitive_ticks(min_val, max_val, num_ticks=5):
    """Generate intuitive tick values that are easy to read."""
    if min_val == max_val:return np.array([min_val])
    range_val = max_val - min_val
    initial_step = range_val/num_ticks
    potential_good_values = [1*10**i for i in np.arange(-10,10).astype(float)]
    potential_good_values.extend([2*10**i for i in np.arange(-10,10).astype(float)])
    potential_good_values.extend([2.5*10**i for i in np.arange(-10,10).astype(float)])
    potential_good_values.extend([5*10**i for i in np.arange(-10,10).astype(float)])
    potential_good_values = np.sort(np.unique(potential_good_values))
    potential_step1 = potential_good_values[potential_good_values<initial_step][-1]
    start_val = math.ceil(min_val/potential_step1) * potential_step1
    potential_ticks1 = np.arange(start_val, max_val, potential_step1)
    potential_step2 = potential_good_values[potential_good_values>initial_step][0]
    start_val = math.ceil(min_val/potential_step2) * potential_step2
    potential_ticks2 = np.arange(start_val, max_val, potential_step2)
    # pick the one with the number of ticks closest to num_ticks
    if (potential_ticks1.shape[0] - num_ticks) < (potential_ticks2.shape[0] - num_ticks):
        ticks = potential_ticks1
        step = potential_step1
    else:
        ticks = potential_ticks2
        step = potential_step2
    return ticks

def plot_projection_space_density(P,y_labels,plot_path,sum_norm=False,log=None,use_log10_scale=False):
    if log is None:
        log = logging.getLogger("ProjectionPlotDensity")
    log.info(f"Generating projection space density plot: {plot_path}")
    if sum_norm:
        P = P * (np.clip(P.sum(1),1,None).mean() / (np.clip(P.sum(1),1,None)[:, None]))
    labels = np.array([f"Bit {str(bit)}" for bit in range(P.shape[1])])
    unique_cell_types = np.unique(y_labels)
    num_measurements = labels.shape[0]
    num_plot_pairs = math.ceil(num_measurements / 2)
    total_rows = num_plot_pairs
    total_cols = 2
    fig, axes = plt.subplots(total_rows, total_cols,figsize=(12, 5 * total_rows), squeeze=False) 
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
            
            # Only filter positive values if using log10 scale
            if use_log10_scale:
                x_pos = x[x > 0]
                y_pos = y[y > 0]
                if len(x_pos) > 1: vmin_x, vmax_x = np.percentile(x_pos, [0.1, 99.9])
                elif len(x_pos) == 1: vmin_x, vmax_x = x_pos[0], x_pos[0]
                else: vmin_x, vmax_x = 0, 0
                if len(y_pos) > 1: vmin_y, vmax_y = np.percentile(y_pos, [0.1, 99.9])
                elif len(y_pos) == 1: vmin_y, vmax_y = y_pos[0], y_pos[0]
                else: vmin_y, vmax_y = 0, 0
            else:
                # Use all values including negatives for linear scale
                if len(x) > 1: vmin_x, vmax_x = np.percentile(x, [0.1, 99.9])
                elif len(x) == 1: vmin_x, vmax_x = x[0], x[0]
                else: vmin_x, vmax_x = 0, 0
                if len(y) > 1: vmin_y, vmax_y = np.percentile(y, [0.1, 99.9])
                elif len(y) == 1: vmin_y, vmax_y = y[0], y[0]
                else: vmin_y, vmax_y = 0, 0
            
            vmax_x = max(vmax_x, vmin_x)
            vmax_y = max(vmax_y, vmin_y)
            x = np.clip(x, vmin_x, vmax_x)
            if use_log10_scale: x = np.log10(x + 1) 
            x_min, x_max = x.min(), x.max()
            x_bins = np.linspace(x_min, x_max if x_max > x_min else x_max + 1, 100)
            y = np.clip(y, vmin_y, vmax_y)
            if use_log10_scale: y = np.log10(y + 1) 
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
            im1 = ax1.imshow(img.T, vmin=vmin_img, vmax=vmax_img, cmap='coolwarm', origin='lower', aspect='auto', interpolation='nearest',
                           extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], rasterized=True)
            
            # Generate intuitive tick values
            x_tick_values = generate_intuitive_ticks(x_bins[0], x_bins[-1], num_ticks=5)
            y_tick_values = generate_intuitive_ticks(y_bins[0], y_bins[-1], num_ticks=5)
            
            ax1.set_xticks(x_tick_values)
            ax1.set_yticks(y_tick_values)
            ax1.set_xticklabels([f"{val:.1e}" for val in x_tick_values])
            ax1.set_yticklabels([f"{val:.1e}" for val in y_tick_values])
            if use_log10_scale:
                ax1.set_xlabel(f"{feature_name1} (log10)")
                ax1.set_ylabel(f"{feature_name2} (log10)")
            else:
                ax1.set_xlabel(f"{feature_name1}")
                ax1.set_ylabel(f"{feature_name2}")
            ax1.grid(False)
            ax2 = axes[current_row_idx, 1] 
            composite_img = np.zeros((len(y_bins)-1, len(x_bins)-1, 3))
            for ct in unique_cell_types:
                mask = y_labels == ct
                if np.sum(mask) < 2: continue
                img_ct, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_bins, y_bins])
                img_ct = np.log10(img_ct + 1)
                img_ct_pos = img_ct[img_ct > 0]
                if len(img_ct_pos) >= 2: 
                    vmin_ct, vmax_ct = np.percentile(img_ct_pos, [0.1, 99]) 
                    vmin_ct = 0 
                    if vmax_ct <= vmin_ct: vmax_ct = vmin_ct + 1e-6 
                    if vmax_ct > 1e-9: img_ct_norm = (img_ct - vmin_ct) / vmax_ct
                    else: img_ct_norm = np.zeros_like(img_ct)
                elif len(img_ct_pos) == 1: img_ct_norm = (img_ct > 0).astype(float) 
                else: img_ct_norm = np.zeros_like(img_ct) 
                img_ct_norm = np.clip(img_ct_norm, 0, 1).T 
                if ct not in color_mapper:
                    attempts = 0
                    max_attempts = 200
                    min_dist_sq = 0.1 
                    min_sum = 0.5 
                    while attempts < max_attempts:
                        color = np.random.rand(3)
                        color_sum = np.sum(color)
                        distances_sq = [float(np.sum((color - existing_color)**2)) for existing_color in used_colors_list]
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
                       extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], rasterized=True)
            ax2.set_xticks(x_tick_values)
            ax2.set_xticklabels([f"{val:.1e}" for val in x_tick_values])
            ax2.set_yticks(y_tick_values)
            ax2.set_yticklabels([f"{val:.1e}" for val in y_tick_values])
            if use_log10_scale:
                ax2.set_xlabel(f"{feature_name1} (log10)")
                ax2.set_ylabel(f"{feature_name2} (log10)")
            else:
                ax2.set_xlabel(f"{feature_name1}")
                ax2.set_ylabel(f"{feature_name2}")
            ax2.grid(False)
            plot_pair_idx += 1 
    try:
        plt.savefig(plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        log.info(f"Saved projection space density plot to {plot_path.replace('.png', '.pdf')}")
    except Exception as e:
        log.error(f"Failed to save plot {plot_path.replace('.png', '.pdf')}: {e}")
    finally:
        plt.close(fig)

def plot_P_Type(P_type_data, valid_type_labels, plot_path, log):
    """Generate P_type heatmap."""
    n_bits = P_type_data.shape[1]
    fig_width = min(max(6, n_bits / 1.5), 25)
    fig_height = min(max(6, len(valid_type_labels) / 2), 25)
    
    heatmap_fig = None
    try:
        # Sort cell types alphabetically
        sorted_indices = np.argsort(valid_type_labels)
        sorted_labels = [valid_type_labels[i] for i in sorted_indices]
        sorted_data = P_type_data[sorted_indices]
        vmin, vmax = torch.quantile(sorted_data, torch.tensor([0.01, 0.99])).tolist()
        p_type_df = pd.DataFrame(sorted_data.numpy(),
                                 index=pd.Index(sorted_labels),
                                 columns=pd.Index([f"Bit_{b}" for b in range(n_bits)]))
        
        heatmap_fig = plt.figure(figsize=(fig_width, fig_height))
        ax_heatmap = heatmap_fig.add_subplot(111)
        if 'Bit' in plot_path:
            center = 0
            cmap = 'coolwarm'
        else:
            center = None
            cmap = 'inferno'
        sns.heatmap(p_type_df, 
                   cmap=cmap,
                   center=center,
                   vmin=vmin,
                   vmax=vmax,
                   linewidths=0.1,
                   ax=ax_heatmap,
                   cbar=True)
        ax_heatmap.set_xlabel("Bit")
        ax_heatmap.set_ylabel("Cell Type")
        plt.setp(ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(ax_heatmap.get_yticklabels(), rotation=0)
        
        heatmap_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        log.info(f"Saved P_type heatmap to {plot_path}")
        
    except Exception as e:
        log.error(f"Error generating P_type heatmap: {e}")
    finally:
        if heatmap_fig is not None:
            plt.close(heatmap_fig)

def plot_P_Type_correlation(P_type_data, valid_type_labels, plot_path, log):
    """Generate type-by-type correlation heatmap."""
    n_bits = P_type_data.shape[1]
    fig_width = min(max(8, len(valid_type_labels) / 1.5), 25)
    fig_height = min(max(6, len(valid_type_labels) / 2), 25)
    
    corr_fig = None
    try:
        # Calculate correlation matrix
        P_type_centered = P_type_data - P_type_data.mean(dim=1, keepdim=True)
        P_type_std = P_type_centered.std(dim=1, keepdim=True).clamp(min=1e-6)
        P_type_norm_corr = P_type_centered / P_type_std
        correlation_matrix = (P_type_norm_corr @ P_type_norm_corr.T / n_bits).numpy()
        
        # Sort cell types alphabetically
        sorted_indices = np.argsort(valid_type_labels)
        sorted_labels = [valid_type_labels[i] for i in sorted_indices]
        sorted_correlation_matrix = correlation_matrix[sorted_indices][:, sorted_indices]
        
        corr_df = pd.DataFrame(sorted_correlation_matrix, 
                              index=pd.Index(sorted_labels), 
                              columns=pd.Index(sorted_labels))
        
        corr_fig = plt.figure(figsize=(fig_width, fig_height))
        ax_corr = corr_fig.add_subplot(111)
        sns.heatmap(corr_df, annot=False, cmap='coolwarm', fmt=".2f", 
                   vmin=-1, vmax=1, center=0, linewidths=.5, ax=ax_corr, cbar=True)
        ax_corr.set_xlabel("Cell Type")
        ax_corr.set_ylabel("Cell Type")
        plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax_corr.get_yticklabels(), rotation=0)
        corr_fig.tight_layout()
        
        corr_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        log.info(f"Saved type correlation heatmap to {plot_path}")
        
    except Exception as e:
        log.error(f"Error generating type correlation heatmap: {e}")
    finally:
        if corr_fig is not None:
            plt.close(corr_fig)

def sum_normalize_p_type(P_type_data):
    """Sum normalize P_type data to average sum."""
    P_type_sum_norm = P_type_data.clone()
    avg_sum = P_type_sum_norm.sum(dim=1).mean()
    P_type_sum_norm = P_type_sum_norm * (avg_sum / P_type_sum_norm.sum(dim=1, keepdim=True).clamp(min=1e-8))
    return P_type_sum_norm

def bitwise_center_p_type(P_type_data):
    """Bitwise center P_type data by median."""
    P_type_bit_center = P_type_data.clone()
    median_values = P_type_bit_center.median(dim=0, keepdim=True).values  # Extract values from named tuple
    P_type_bit_center = P_type_bit_center - median_values
    return P_type_bit_center

def bitwise_normalize_p_type(P_type_data):
    """Bitwise z-score normalize P_type data."""
    P_type_bit_norm = P_type_data.clone()
    P_type_bit_norm = (P_type_bit_norm - P_type_bit_norm.mean(dim=0, keepdim=True)) / P_type_bit_norm.std(dim=0, keepdim=True).clamp(min=1e-8)
    return P_type_bit_norm

def plot_single_learning_curve(parameter, learning_curve, output_dir, log=None):
    """
    Plot learning curve for a single parameter (training vs test).
    
    Args:
        parameter (str): Parameter name (without _train or _test suffix)
        learning_curve (pd.DataFrame): DataFrame containing learning statistics
        output_dir (str): Directory to save the learning curve plot
        log (logging.Logger, optional): Logger instance for logging messages
    """
    if log is None:
        log = logging.getLogger("LearningCurvePlotter")
    
    n_start = 0
    x = np.array(learning_curve.index)[n_start:-1].astype(float)
    y1 = np.array(learning_curve[parameter+'_train'])[n_start:-1].astype(float)
    y2 = np.array(learning_curve[parameter+'_test'])[n_start:-1].astype(float)
    
    # Create mask for finite values
    mask1 = np.isfinite(y1)
    mask2 = np.isfinite(y2)
    
    # Only proceed if we have valid data
    if np.any(mask1) and np.any(mask2):
        y_min, y_max = np.percentile(y1[mask1], [1, 99])
        plt.figure(figsize=(5, 3), dpi=200)
        plt.scatter(x[mask1], y1[mask1], label='Train', s=1, alpha=0.6, rasterized=True)
        plt.scatter(x[mask2], y2[mask2], label='Test', c='orange', s=1, alpha=0.6, rasterized=True)
        plt.xlabel('Epoch')
        # Format parameter name for better readability
        param_display = parameter.replace('_', ' ').title()
        plt.ylabel(param_display)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"learning_curve_{parameter}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Saved learning curve for {parameter} to {plot_path}")
    else:
        log.warning(f"Skipping learning curve for {parameter}: insufficient valid data")

def plot_loss_contributions(learning_curve, output_dir, log=None):
    """Plot relative contribution of each loss term to total loss across training epochs."""
    if log is None:
        log = logging.getLogger("LossContributionPlotter")
    loss_columns = [col for col in learning_curve.columns if 'train' in col.lower() and 'loss' in col.lower()]
    total_loss_col = [col for col in learning_curve.columns if 'train' in col.lower() and 'loss' in col.lower()and 'total' in col.lower()][0]
    valid_iterations = [idx for idx in learning_curve.index if str(idx).replace('.', '').isdigit()]
    if not valid_iterations:
        log.warning("No valid numeric iterations found")
        return
    data = learning_curve.loc[valid_iterations]
    x = np.array([float(idx) for idx in data.index])
    total_loss_values = data[total_loss_col].astype(float)
    # Calculate relative contributions
    contributions = []
    valid_loss_names = []
    for col in loss_columns:
        if col == total_loss_col:
            continue
        loss_values = data[col].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_contribution = np.divide(loss_values, total_loss_values, 
                                           out=np.zeros_like(loss_values), 
                                           where=total_loss_values != 0)
        if np.any(np.isfinite(relative_contribution)):
            contributions.append(relative_contribution)
            clean_name = col.replace('_train', '').replace('$$$ - ', '').replace('_', ' ').replace('loss', '').title()
            valid_loss_names.append(clean_name)
    if not contributions:
        log.warning("No valid loss contributions found")
        return
    # Create stacked area plot
    plt.figure(figsize=(12, 8))
    plt.stackplot(x, np.vstack(contributions), labels=valid_loss_names, alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Relative Contribution to Total Loss')
    plt.title('Loss Term Contributions (Training)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_contributions.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved loss contributions plot to {plot_path}")

def plot_comprehensive_performance(learning_curve, output_dir, log=None):
    """Plot accuracy, separation, and dynamic range metrics with three y-axes on the left side."""
    if log is None:
        log = logging.getLogger("ComprehensivePerformancePlotter")
    metrics = {
        'dynamic_range': {'train': 'dynamic_range_train', 'test': 'dynamic_range_test', 'color': '#17becf', 'label': 'Dynamic Range'},
        'accuracy': {'train': 'accuracy_train', 'test': 'accuracy_test', 'color': '#ff7f0e', 'label': 'Accuracy'},
        'separation': {'train': 'separation_train', 'test': 'separation_test', 'color': '#9467bd', 'label': 'Separation'}
    }
    valid_iterations = [idx for idx in learning_curve.index if str(idx).replace('.', '').isdigit()]
    if not valid_iterations:
        log.warning("No valid numeric iterations found")
        return
    data = learning_curve.loc[valid_iterations]
    x = np.array([float(idx) for idx in data.index])
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    axes = []
    for i, (metric_name, metric_info) in enumerate(metrics.items()):
        if i == 0:
            ax = ax1
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_label_position('left')
        elif i == 1:
            ax = ax1.twinx()
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
        else:
            ax = ax1.twinx()
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
            ax.spines['right'].set_position(('axes', 0))
        axes.append(ax)
        y_train = data[metric_info['train']].astype(float)
        y_test = data[metric_info['test']].astype(float)
        mask_train = np.isfinite(y_train)
        mask_test = np.isfinite(y_test)
        if np.any(mask_train) and np.any(mask_test):
            ax.scatter(x[mask_train], y_train[mask_train], 
                      color=metric_info['color'], 
                      s=1, alpha=0.6, marker='o', rasterized=True)
            ax.scatter(x[mask_test], y_test[mask_test], 
                      color=metric_info['color'], 
                      s=10, alpha=0.6, marker='o', rasterized=True)
            ax.tick_params(axis='y', labelcolor=metric_info['color'], pad=1.5)
            if metric_name == 'dynamic_range':
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            if metric_name == 'accuracy':
                ax.set_ylim(0, 1)
            else:  # separation and dynamic_range
                y_min, y_max = np.percentile(np.concatenate([y_train[mask_train], y_test[mask_test]]), [1, 99])
                ax.set_ylim(0, y_max)  # Set lower limit to 0
    ax1.set_xlabel('Epoch')
    ax1.grid(True, alpha=0.3)
    y_positions = [0.25, 0.20, 0.15]
    for i, (metric_name, metric_info) in enumerate(metrics.items()):
        if i < len(y_positions):
            plt.figtext(0.2, y_positions[i], metric_info['label'], 
                       color=metric_info['color'], fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comprehensive_performance.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved comprehensive performance plot to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("user_parameters_path", type=str, help="Path to csv containing parameters for model")
    args = parser.parse_args()
    user_parameters_path = args.user_parameters_path
    model = EncodingDesigner(user_parameters_path=user_parameters_path)
    if not model.initialize():
        print("Initialization failed. Check the log file for details.")
        exit(1)
    
    # Check if we should continue training or skip to evaluation
    if model.training_completed:
        print("Training appears to be completed. Skipping training and proceeding to evaluation.")
        model.log.info("Skipping training as training appears to be completed.")
    elif model.I['continue_training'] == 0 and model.is_initialized_from_file:
        print("Model loaded from file and continue_training=0. Skipping training and proceeding to evaluation.")
        model.log.info("Skipping training due to continue_training=0 and model loaded from file.")
    else:
        print("Starting training...")
        model.fit()
    
    model.evaluate()
    model.visualize(show_plots=False)
