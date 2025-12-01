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


class CIPHER(nn.Module):
    """
    CIPHER (Cell Identity Projection using Hybridization Encoding Rules) Model.
    
    A PyTorch neural network that learns to encode high-dimensional gene expression data
    into low-dimensional bit projections for multiplexed in situ hybridization probe design.
    The model simultaneously optimizes probe allocation, cell type classification accuracy,
    and experimental constraints.
    
    Attributes:
        I (Dict[str, Any]): Dictionary containing all model parameters and configuration
        encoder (nn.Embedding): Encoder layer that maps genes to bit contributions
        decoder (nn.Sequential): Decoder network that maps projections to cell type predictions
        constraints (torch.Tensor): Maximum allowed probes per gene
        X_train, X_test (torch.Tensor): Training and test gene expression matrices
        y_train, y_test (torch.Tensor): Training and test cell type labels
        n_genes (int): Number of genes in the dataset
        n_bit (int): Number of bits in the encoding
        n_categories (int): Number of cell type categories
        learning_stats (pd.DataFrame): Training statistics recorded during training
        log (logging.Logger): Logger instance for training logs
    
    Args:
        user_parameters_path (Optional[str]): Path to CSV file containing user-defined parameters.
            If None, uses default parameters. Parameters in the CSV override defaults.
    """
    def __init__(self, user_parameters_path: Optional[str] = None):
        super().__init__() 
        self.I: Dict[str, Any] = {
            # Core model parameters
            'n_cpu': 12,  # Number of CPU threads to use for PyTorch
            'n_bit': 24,  # Number of bits in the encoding (dimensionality of the projection)
            'n_iters': 10000,  # Total number of training iterations
            'batch_size': 1000,  # Batch size for training (0 = use full dataset)
            'brightness_s': 4.5,  # Initial target brightness in log10 scale
            'brightness_e': 4.5,  # Final target brightness in log10 scale
            'saturation': 1.0,  # When to reach final values for all _s/_e parameters (0.0-1.0, 1.0 = end of training)
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
            'sparsity_threshold': 0.01,  # Threshold below which weights are considered sparse (for sparsity calculation)
            'categorical_wt': 1,  # Weight for categorical classification loss
            'label_smoothing': 0.1,  # Label smoothing factor for cross-entropy loss
            'gene_importance_wt': 0,  # Weight for gene importance loss term (penalizes genes contributing >25% to any bit)
            'gene_importance': 0.25,  # Maximum allowed contribution percentage per gene per bit
            'bit_usage_wt': 0,  # Weight for bit usage loss term (encourages decoder to use each bit)
            'bit_usage': 0.01,  # Minimum decoder weight magnitude required for a bit to be considered "used"
            'bit_corr_wt': 0,  # Weight for bit correlation loss term (penalizes high correlation between bits)
            'bit_corr': 0.5,  # Maximum allowed correlation between any two bits (0-1)
            'step_size_wt': 0,  # Weight for step size loss term (ensures minimum step sizes between cell types)
            'step_size_threshold': 0.1,  # Minimum step size required between cell types (absolute value)
            'step_size_n_steps': 0.1,  # Fraction of cell types to enforce minimum step size (0.1 = top 10% of steps)
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
            'X_drp_e': 0,  # Final proportion of genes to drop out (randomly set to 0)
            'X_noise_s': 0,  # Initial gene expression noise level 0.5 -> 50% decrease to 200% increase (0-1)
            'X_noise_e': 0,  # Final gene expression noise level 0.5 -> 50% decrease to 200% increase (0-1)
            # Weight-level noise parameters
            'E_drp_s': 0,  # Initial proportion of encoding weights to drop out (randomly set to 0)
            'E_drp_e': 0,  # Final proportion of encoding weights to drop out (randomly set to 0)
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
            'central_brain': 0,  # Whether to only use central brain data (0 = no, 1 = yes)
        }
        self._setup_logging(user_parameters_path)
        self._load_and_process_parameters(user_parameters_path)
        self._setup_output_and_symlinks(user_parameters_path)
        self.learning_stats = pd.DataFrame()
        self.saved_models = {}
        self.saved_optimizer_states = {}
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.best_iteration = -1
        self.training_completed = False
        self.prev_encoder_weights = None
        self.prev_decoder_weights = None

    def initialize(self) -> bool:
        """
        Initialize the CIPHER model by loading data, constraints, and setting up encoder/decoder.
        
        This method performs the following steps:
        1. Loads gene constraints from file
        2. Loads training and test data (gene expression and labels)
        3. Optionally filters to top N genes if top_n_genes > 0
        4. Initializes encoder with proper weight initialization
        5. Initializes decoder with specified architecture
        6. Loads pretrained model if available
        
        Returns:
            bool: True if initialization succeeded, False otherwise
            
        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data shapes are inconsistent
        """
        self.log.info("--- Starting Initialization ---")
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

    def get_E_clean(self) -> torch.Tensor:
        """
        Get encoding weights without noise/dropout for loss calculations.
        
        Applies the encoder activation function to raw weights and multiplies by gene constraints
        to get the final probe allocation matrix. This is the "clean" version used for
        loss calculations to ensure consistent evaluation.
        
        Returns:
            torch.Tensor: Encoding weights E of shape (n_genes, n_bit) representing
                the number of probes allocated from each gene to each bit.
                Values are constrained by gene limits and activation function ranges.
                
        Note:
            The activation function converts raw weights to probe fractions:
            - tanh: maps to [0, 1] via (tanh(x) + 1) / 2
            - sigmoid: maps to [0, 1]
            - linear: uses weights directly
            - relu: maps to [0, ∞)
        """
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

    def get_E(self) -> torch.Tensor:
        """
        Get encoding weights with noise/dropout applied for training.
        
        This method applies noise and dropout to encoding weights during training to
        improve robustness and prevent overfitting. The noise simulates experimental
        variability in probe binding.
        
        Returns:
            torch.Tensor: Encoding weights E of shape (n_genes, n_bit) with noise/dropout
                applied if training mode is enabled and use_noise=1.
                
        Note:
            - Only applies noise if self.training is True and use_noise=1
            - E_drp: Randomly sets some weights to 0 (dropout)
            - E_noise: Applies multiplicative noise with minimum bound
        """
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

    def project_clean(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Project gene expression data to bit space without noise for loss calculations.
        
        Computes the matrix multiplication X @ E to get bit projections. This clean version
        is used for loss calculations to ensure consistent evaluation metrics.
        
        Args:
            X (torch.Tensor): Gene expression matrix of shape (n_samples, n_genes)
            E (torch.Tensor): Encoding weights of shape (n_genes, n_bit)
            
        Returns:
            torch.Tensor: Projections P of shape (n_samples, n_bit) representing
                expected signal intensity for each bit for each sample.
        """
        P = X.mm(E)
        return P

    def project(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Project gene expression data to bit space with noise/dropout for training.
        
        Computes projections with various noise types applied to simulate experimental
        conditions. This noisy version is used during training to improve robustness.
        
        Args:
            X (torch.Tensor): Gene expression matrix of shape (n_samples, n_genes)
            E (torch.Tensor): Encoding weights of shape (n_genes, n_bit)
            
        Returns:
            torch.Tensor: Projections P of shape (n_samples, n_bit) with noise applied.
            
        Note:
            Noise types applied (if training and use_noise=1):
            - X_noise: Multiplicative noise on gene expression
            - X_drp: Dropout on gene expression (randomly set to 0)
            - P_noise: Measurement accuracy error on projections
            - P_add: Constant background signal (log10 scale)
            - P_drp: Dropout on projections
        """
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

    def decode(self, P: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode bit projections to cell type predictions.
        
        Applies optional normalization, then passes projections through the decoder network
        to get cell type logits. Computes predictions, accuracy, and categorical loss.
        
        Args:
            P (torch.Tensor): Bit projections of shape (n_samples, n_bit)
            y (torch.Tensor): True cell type labels of shape (n_samples,)
            
        Returns:
            tuple containing:
                - y_predict (torch.Tensor): Predicted cell type indices of shape (n_samples,)
                - accuracy (torch.Tensor): Classification accuracy (scalar)
                - categorical_loss (torch.Tensor): Cross-entropy loss with label smoothing
                
        Note:
            Normalization options:
            - sum_norm: Normalizes by sum across bits (P_scaling * P / sum(P))
            - bit_norm: Z-score normalizes each bit across samples
        """
        if self.I['sum_norm'] != 0:
            P = self.I['P_scaling'] * P / P.sum(1).unsqueeze(1).clamp(min=1e-8)
            # P = (P - P.mean(1).unsqueeze(1)) / P.std(1).unsqueeze(1).clamp(min=1e-8)
        if self.I['bit_norm'] != 0:
            P = (P - P.mean(0)) / P.std(0).clamp(min=1e-8)
        R = self.decoder(P) 
        y_predict = R.max(1)[1]
        accuracy = (y_predict == y).float().mean()
        if self.I['categorical_wt'] != 0:
            # OPTIMIZATION: Cache loss function to avoid recreating it every time
            if not hasattr(self, '_cached_loss_fn'):
                self._cached_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.I['label_smoothing'])
            categorical_loss = self._cached_loss_fn(R, y)
        else:
            categorical_loss = torch.tensor(0, device=R.device, dtype=torch.float32, requires_grad=True)
        return y_predict, accuracy, categorical_loss

    def calculate_loss(self, X: torch.Tensor, y: torch.Tensor, iteration: int, suffix: str = '') -> tuple[torch.Tensor, dict]:
        """
        Calculate total loss and all loss components for a batch of data.
        
        Computes all loss terms (LA, LH, LM, and robustness losses) and returns the
        total weighted loss along with detailed statistics. Uses clean encoding weights
        for loss calculations but noisy projections for decoder training.
        
        Args:
            X (torch.Tensor): Gene expression matrix of shape (n_samples, n_genes)
            y (torch.Tensor): Cell type labels of shape (n_samples,)
            iteration (int): Current training iteration (for parameter updates)
            suffix (str): Suffix to append to statistic names (e.g., '_train', '_test')
            
        Returns:
            tuple containing:
                - total_loss (torch.Tensor): Sum of all weighted loss terms
                - current_stats (dict): Dictionary of statistics for this batch including:
                    - Loss values: '$$$ - <loss_name>' for each loss term
                    - Accuracy: 'accuracy'
                    - Probe counts: 'E_n_probes'
                    - Brightness metrics: 'lower brightness', 'median brightness', etc.
                    - Dynamic range: 'dynamic_fold'
                    - Separation: 'separation'
                    - And many other training metrics
                    
        Note:
            Loss categories:
            - LA (Accuracy): categorical_loss
            - LH (Hybridization): probe_wt_loss, gene_constraint_loss
            - LM (Measurability): brightness_loss, dynamic_loss, separation_loss, step_size_loss
            - Robustness: sparsity_loss, gene_importance_loss, bit_usage_loss, correlation_loss
        """
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
        sparsity_ratio = (E_clean < self.I['sparsity_threshold']).float().mean()
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
        # current_stats['E_worst_bit' + suffix] = f"min:{max(min_lower, 1):.2e}, med:{max(min_median, 1):.2e}, max:{max(min_upper, 1):.2e}, fold:{bit_dynamic_ranges[min_range_idx]:.2f}"
        # current_stats['E_best_bit' + suffix] = f"min:{max(max_lower, 1):.2e}, med:{max(max_median, 1):.2e}, max:{max(max_upper, 1):.2e}, fold:{bit_dynamic_ranges[max_range_idx]:.2f}"
        
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
        dynamic_range = signal['upper']-signal['lower']
        current_stats['dynamic_range' + suffix] = round(torch.quantile(dynamic_range,0.1).item(), 4)

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
        current_stats['dynamic_fold' + suffix] = round(torch.quantile(dynamic_range,0.1).item(), 4)
        # current_stats['dynamic_fold' + suffix] = round(dynamic_range.mean().item(), 4)

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
            # current_stats['separation_report' + suffix] = f"min:{worst_separation:.2f}, p10:{p10:.2f}, p50:{p50:.2f}, p90:{p90:.2f}, max:{best_separation:.2f}"

        # --- Gene importance loss ---
        if not hasattr(self, '_cached_avg_gene_expression'):
            self._cached_avg_gene_expression = self.X_train.mean(dim=0)
        avg_gene_expression = self._cached_avg_gene_expression
        gene_contributions = avg_gene_expression.unsqueeze(1) * E_clean
        gene_percentages = gene_contributions / gene_contributions.sum(dim=0, keepdim=True).clamp(min=1e-8)
        target = self.I['gene_importance']
        fold = (gene_percentages - target) / target
        positive_fold = fold[fold > 0]
        if positive_fold.numel() > 0:
            raw_losses['gene_importance_loss'] = self.I['gene_importance_wt'] * positive_fold.sum()
        else:
            raw_losses['gene_importance_loss'] = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        current_stats['gene_importance_violations' + suffix] = positive_fold.numel()
        current_stats['gene_importance_max_pct' + suffix] = round(gene_percentages.max().item(), 4)
        current_stats['gene_importance_mean_excess' + suffix] = round(positive_fold.mean().item() if positive_fold.numel() > 0 else 0, 4)

        # --- Bit usage loss ---
        first_layer = self.decoder[0]
        if isinstance(first_layer, nn.Linear):
            bit_importance = torch.abs(first_layer.weight.data).sum(dim=0)
            total_importance = bit_importance.sum()
            bit_percentage = bit_importance / total_importance.clamp(min=1e-8)
            bit_utilization = bit_percentage * self.I['n_bit']
            target = self.I['bit_usage']
            fold = (target - bit_utilization) / target
            positive_fold = fold[fold > 0]
            if positive_fold.numel() > 0:
                raw_losses['bit_usage_loss'] = self.I['bit_usage_wt'] * positive_fold.sum()
            else:
                raw_losses['bit_usage_loss'] = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
            current_stats['bit_usage_unused_bits' + suffix] = positive_fold.numel()
            current_stats['bit_usage_min_utilization' + suffix] = round(bit_utilization.min().item(), 4)
            current_stats['bit_usage_mean_excess' + suffix] = round(positive_fold.mean().item() if positive_fold.numel() > 0 else 0, 4)
        
        # --- Bit correlation loss ---
        P_sum_norm = P_clean / P_clean.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P_centered = P_sum_norm - P_sum_norm.mean(dim=0, keepdim=True)
        # Normalize by standard deviation for correlation calculation
        P_std = P_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
        P_normalized = P_centered / P_std
        # Compute correlation matrix: (n_bits, n_samples) @ (n_samples, n_bits) = (n_bits, n_bits)
        correlation_matrix = (P_normalized.T @ P_normalized) / (P_normalized.shape[0] - 1)
        upper_tri_mask = torch.triu(torch.ones_like(correlation_matrix), diagonal=1).bool()
        correlations = correlation_matrix[upper_tri_mask]
        target = self.I['bit_corr']
        fold = (correlations - target) / target
        positive_fold = fold[fold > 0]
        if positive_fold.numel() > 0:
            raw_losses['correlation_loss'] = self.I['bit_corr_wt'] * positive_fold.sum()
        else:
            raw_losses['correlation_loss'] = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        current_stats['correlation_violations' + suffix] = positive_fold.numel()
        current_stats['correlation_max' + suffix] = round(correlations.max().item(), 4)
        current_stats['correlation_mean' + suffix] = round(correlations.mean().item(), 4)
        current_stats['correlation_mean_excess' + suffix] = round(positive_fold.mean().item() if positive_fold.numel() > 0 else 0, 4)
        
        # --- Step size loss ---
        step_size_loss = torch.tensor(0, device=P_clean.device, dtype=torch.float32, requires_grad=True)
        total_violations = 0
        all_step_sizes = []
        for bit_idx in range(P_clean.shape[1]):
            bit_values = P_clean[:, bit_idx]
            sorted_values = torch.sort(bit_values)[0]
            steps = sorted_values[1:] - sorted_values[:-1]
            all_step_sizes.append(steps)
            n_steps_to_enforce = max(1, int(len(steps) * self.I['step_size_n_steps']))
            top_steps, _ = torch.topk(steps, min(n_steps_to_enforce, len(steps)))
            target = self.I['step_size_threshold']
            fold = (target - top_steps) / target
            positive_fold = fold[fold > 0]
            if positive_fold.numel() > 0:
                step_size_loss = step_size_loss + positive_fold.sum()
                total_violations += positive_fold.numel()
        raw_losses['step_size_loss'] = self.I['step_size_wt'] * step_size_loss
        # Calculate overall statistics across all bits
        all_steps = torch.cat(all_step_sizes)
        current_stats['step_size_violations' + suffix] = total_violations
        current_stats['step_size_max' + suffix] = round(all_steps.max().item(), 4)
        current_stats['step_size_median' + suffix] = round(all_steps.median().item(), 4)
        current_stats['step_size_mean' + suffix] = round(all_steps.mean().item(), 4)
        current_stats['step_size_mean_excess' + suffix] = round(step_size_loss.item() / max(total_violations, 1), 4)
        
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

        # --- Categorical loss ---
        categorical_loss_component = self.I['categorical_wt'] * raw_categorical_loss_component
        raw_losses['categorical_loss'] = categorical_loss_component
        current_stats['accuracy' + suffix] = round(accuracy.item(), 4)

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

    def train_gene_importance_decoder(self) -> torch.Tensor:
        """
        Train a simple linear decoder from genes to cell types to identify important genes.
        
        This method trains a linear classifier directly from gene expression to cell types
        to identify which genes are most important for classification. The top N genes (based
        on top_n_genes parameter) are then selected for the main CIPHER model.
        
        Returns:
            torch.Tensor: Gene importance scores of shape (n_genes,) indicating the
                importance of each gene for cell type classification.
                
        Note:
            - Saves gene importance scores to 'gene_importance_scores.csv'
            - Saves gene mask to 'gene_mask.pt'
            - Filters X_train, X_test, constraints, and genes based on top genes
            - Creates training plots showing loss and accuracy over epochs
        """
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

    def perturb_E(self) -> None:
        """
        Randomly perturb a fraction of encoder weights to escape local minima.
        
        Selects a random subset of encoder weights (E_perb_prct) and reinitializes them
        to random values within the perturbation range. This helps the model escape
        local minima during training.
        
        Note:
            - Only perturbs weights if E_perturb_rt > 0
            - Perturbation range: [E_perturb_min, E_perturb_max]
            - Weights are transformed to pre-activation space before perturbation
            - Logs the number and percentage of weights perturbed
        """
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

    def fit(self) -> None:
        """
        Train the CIPHER model for the specified number of iterations.
        
        Main training loop that:
        1. Updates dynamic parameters (_s/_e parameters) based on training progress
        2. Samples training batches
        3. Computes loss and backpropagates
        4. Applies gradient clipping
        5. Updates model weights
        6. Evaluates on test set periodically
        7. Saves checkpoints and visualizations
        
        Raises:
            RuntimeError: If model is not initialized (missing data or components)
            
        Note:
            - Training progress is logged every report_rt iterations
            - Best model is saved if best_model=1
            - Weight perturbation occurs every E_perturb_rt iterations
            - Evaluation and visualization saved every 10k iterations
            - Handles NaN/Inf gradients by reverting to previous state
        """
        if self.X_train is None or self.y_train is None or self.constraints is None or self.decoder is None : 
            self.log.error("Model is not initialized. Call initialize() before fit().")
            raise RuntimeError("Model is not initialized. Call initialize() before fit().")
        # Initialize training state variables
        self.learning_stats = pd.DataFrame()
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
                # Initialize row for this iteration if DataFrame is empty
                if self.learning_stats.empty:
                    self.learning_stats = pd.DataFrame(index=pd.Index([str(iteration)]))
                elif str(iteration) not in self.learning_stats.index:
                    self.learning_stats.loc[str(iteration)] = pd.Series(dtype=object)
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
                # Add batch stats to the DataFrame row
                for key, value in batch_stats.items():
                    self.learning_stats.loc[str(iteration), key] = value
                self.learning_stats.loc[str(iteration), 'total_loss_train'] = total_loss.item()
                total_loss.backward() 
                nan_detected = self._check_gradient_health(iteration)
                if not nan_detected:
                    self._apply_gradient_clipping()
                    self.optimizer_gen.step()
                    delayed_perturbation_iter = self._handle_weight_perturbation(iteration, self.I['report_rt'], delayed_perturbation_iter)
                    self._update_best_model(iteration, total_loss)
                else: 
                    self._revert_to_previous_state(iteration)
                if is_report_iter:
                    last_report_time, last_report_iteration = self._evaluate_on_test_set(iteration, last_report_time, last_report_iteration)
                
                # # Save checkpoint every 10k iterations
                # if (iteration + 1) % 10000 == 0:
                #     self._save_checkpoint(iteration)
                
                # Run evaluation and visualization every 10k iterations
                if (iteration + 1) % 10000 == 0:
                    self.save_eval_and_viz(iteration)
                
                if delayed_perturbation_iter == iteration:
                    self.log.info(f"Performing delayed weight perturbation at iteration {iteration}")
                    self.perturb_E()
                    delayed_perturbation_iter = None
        except Exception as e:
            self.log.exception(f"Error during training loop at iteration {iteration}: {e}")
        finally:
            self.save_eval_and_viz(iteration=self.I['n_iters']-1)

    def evaluate(self) -> None:
        """
        Evaluate the model on test data under various noise conditions.
        
        Computes evaluation metrics including:
        - Probe counts and distributions
        - Signal percentiles (10th, 50th, 90th)
        - Classification accuracy under different noise levels
        - Separation and dynamic range metrics
        
        The evaluation is performed under four noise conditions:
        - No Noise: Clean evaluation
        - Low Noise: Minimal experimental noise
        - Medium Noise: Moderate experimental noise
        - High Noise: High experimental noise
        
        Results are saved to 'Results.csv' in the output directory.
        
        Raises:
            RuntimeError: If model is not initialized or trained
        """
        if self.encoder is None or self.decoder is None or \
           self.X_train is None or self.X_test is None or self.y_train is None or \
           self.y_test is None : 
            self.log.error("Cannot evaluate: Model not initialized or trained. Run initialize() and fit() first.")
            return
        results_dict = {}
        # Use clean versions for evaluation stats to be consistent with loss calculations
        E = self.get_E_clean()
        E_cpu = E.cpu().detach()
        results_dict['Number of Probes (Constrained)'] = E_cpu.sum().item()
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
            results_dict['10th Percentile Signal'] = p10.item()
            results_dict['50th Percentile Signal'] = p50.item()
            results_dict['90th Percentile Signal'] = p90.item()
            for bit in range(avg_P_type.shape[1]):
                results_dict[f"Number of Probes Bit {bit+1}"] = E_cpu[:, bit].sum().item()
                # Calculate percentiles for each bit efficiently
                bit_p10, bit_p50, bit_p90 = torch.quantile(avg_P_type[:, bit], torch.tensor([0.1, 0.5, 0.9]))
                results_dict[f"10th Percentile Signal Bit {bit+1}"] = bit_p10.item()
                results_dict[f"50th Percentile Signal Bit {bit+1}"] = bit_p50.item()
                results_dict[f"90th Percentile Signal Bit {bit+1}"] = bit_p90.item()
        else:
            self.log.warning("Could not calculate average P_type for evaluation stats.")
        self.log.info("--- Basic Evaluation Stats ---")
        for key, val in results_dict.items():
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
                        dynamic_fold = stats.get('dynamic_fold_test', np.nan)
                        self.log.info(f"{level_name} Accuracy: {round(accuracy, 4)}")
                        self.log.info(f"{level_name} Separation: {round(separation, 4)}")
                        self.log.info(f"{level_name} Dynamic Range: {round(dynamic_range, 4)}")
                        self.log.info(f"{level_name} Dynamic Fold: {round(dynamic_fold, 4)}")
                        results_dict[f'{level_name} Accuracy'] = accuracy
                        results_dict[f'{level_name} Separation'] = separation
                        results_dict[f'{level_name} Dynamic Range'] = dynamic_range
                        results_dict[f'{level_name} Dynamic Fold'] = dynamic_fold
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
                                    columns=pd.Index([f"Bit_{b+1}" for b in range(n_bits)])
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
                results_dict[f'{level_name} Accuracy'] = np.nan
                # Ensure training state is restored even if there's an error
                self.training = original_training_state
        # Set back to eval mode after all noise evaluations
        self.eval()
        results_df = pd.DataFrame({
            'values': list(results_dict.values())
        }, index=pd.Index(list(results_dict.keys())))
        results_path = os.path.join(self.I['output'], 'Results.csv') 
        results_df.loc['Dynamic Fold'] = results_df.loc['No Noise Dynamic Fold']
        results_df.loc['Dynamic Range'] = results_df.loc['No Noise Dynamic Range']
        results_df.loc['Accuracy'] = results_df.loc['No Noise Accuracy']
        results_df.loc['Separation'] = results_df.loc['No Noise Separation']
        results_df.loc['Probes'] = results_df.loc['Number of Probes (Constrained)']
        results_df.to_csv(results_path)

        self.log.info(f"Evaluation results saved to {results_path}")

    def visualize(self, show_plots: bool = False) -> None:
        """
        Generate comprehensive visualizations of model performance and learned representations.
        
        Creates the following visualizations:
        1. Confusion matrix for test set predictions
        2. Learning curves for all tracked metrics (train vs test)
        3. Loss contribution plots showing relative importance of each loss term
        4. Comprehensive performance plot (accuracy, separation, dynamic range)
        5. Constraints vs probes per gene scatter plot
        6. Encoder weight matrix (E) heatmap
        7. Decoder weight matrices for each layer
        8. P_type heatmaps for different normalization strategies
        9. P_type correlation heatmaps
        10. Projection space density plots
        
        Args:
            show_plots (bool): Whether to display plots interactively (currently not used)
            
        Raises:
            RuntimeError: If model is not initialized or trained
        """
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
        learning_curve = self.learning_stats
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
            # Convert E_clean to DataFrame for easier manipulation
            E_clean_np = E.detach().cpu().numpy().astype(int)
            used_WeightMat = pd.DataFrame(E_clean_np)
            # if below 1 set to 0
            used_WeightMat = used_WeightMat.applymap(lambda x: 0 if x < 1 else x)

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
            # show only non zero genes
            reordered_df = reordered_df[reordered_df.sum(axis=1) > 0]
            plot_weight_matrix(reordered_df, os.path.join(self.I['output'], 'E.pdf'), 
                             title="Encoder Weights", cmap='Reds', log_scale=True, log=self.log)
        except Exception as e:
            self.log.error(f"Failed to plot E_clean clustermap: {e}")

        # --- Visualization: Decoder weights for each layer ---
        try:
            linear_layers = [(i, module) for i, module in enumerate(self.decoder) if isinstance(module, nn.Linear)]
            if linear_layers:
                for layer_idx, layer in linear_layers:
                    weights = layer.weight.data.detach().cpu().numpy()
                    col_names = [f"Bit_{i+1}" if layer_idx == 0 and weights.shape[1] == self.I['n_bit'] else f"Input_{i+1}" for i in range(weights.shape[1])]
                    weight_df = pd.DataFrame(weights, columns=pd.Index(col_names))
                    plot_path = os.path.join(self.I['output'], f'D_{layer_idx}.pdf')
                    plot_weight_matrix(weight_df, plot_path, 
                                     title=f"Decoder Layer {layer_idx} Weights", log_scale=False, log=self.log)
        except Exception as e:
            self.log.error(f"Failed to plot decoder weights: {e}")

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

    def save_cell_type_averages(self) -> tuple[Optional[torch.Tensor], Optional[List[str]]]:
        """
        Save gene expression averages for each cell type from test data.
        
        Computes the mean gene expression for each cell type in the test set and saves
        the results. If files already exist in input or output directories, loads them
        instead of recomputing.
        
        Returns:
            tuple containing:
                - X_avg (Optional[torch.Tensor]): Average gene expression per cell type,
                    shape (n_cell_types, n_genes), or None if computation failed
                - cell_type_names (Optional[List[str]]): List of cell type names
                    
        Note:
            - Saves to 'X_Type_test.csv' and 'X_Type_test.pt'
            - Creates symlinks from output to input directory
            - Returns None, None if test data is not loaded
        """
        self.log.info("--- Saving Cell Type Averages ---")
        
        if self.X_test is None or self.y_test is None:
            self.log.error("Cannot save cell type averages: Test data not loaded.")
            return
        
        # Define file paths
        output_csv_path = os.path.join(self.I['output'], 'X_Type_test.csv')
        output_pt_path = os.path.join(self.I['output'], 'X_Type_test.pt')
        input_csv_path = os.path.join(self.I['input'], 'X_Type_test.csv')
        input_pt_path = os.path.join(self.I['input'], 'X_Type_test.pt')
        
        # Check if files already exist in output directory
        if os.path.exists(output_csv_path) and os.path.exists(output_pt_path):
            self.log.info("Cell type averages already exist in output directory. Skipping calculation.")
            return None, None
        
        # Check if files exist in input directory
        if os.path.exists(input_csv_path) and os.path.exists(input_pt_path):
            self.log.info("Cell type averages found in input directory. Loading and creating symlinks.")
            try:
                # Load the data from input directory
                X_avg = torch.load(input_pt_path, map_location=self.I['device'])
                df = pd.read_csv(input_csv_path, index_col=0)
                cell_type_names = list(df.columns)
                
                # Create symlinks to output directory
                if os.path.exists(output_csv_path):
                    os.remove(output_csv_path)
                if os.path.exists(output_pt_path):
                    os.remove(output_pt_path)
                
                os.symlink(input_csv_path, output_csv_path)
                os.symlink(input_pt_path, output_pt_path)
                
                self.log.info(f"Cell type averages loaded from input directory and symlinked to output:")
                self.log.info(f"  CSV: {output_csv_path} -> {input_csv_path}")
                self.log.info(f"  PT: {output_pt_path} -> {input_pt_path}")
                self.log.info(f"Shape: {X_avg.shape} ({len(cell_type_names)} cell types × {X_avg.shape[1]} genes)")
                
                return X_avg, cell_type_names
                
            except Exception as e:
                self.log.error(f"Failed to load cell type averages from input directory: {e}")
                self.log.info("Will calculate cell type averages from test data.")
        
        # Calculate averages for each cell type
        self.log.info("Calculating cell type averages from test data.")
        unique_cell_types = torch.unique(self.y_test)
        n_cell_types = len(unique_cell_types)
        n_genes = self.X_test.shape[1]
        X_sum = torch.zeros((n_cell_types, n_genes), device=self.X_test.device)
        X_count = torch.zeros(n_cell_types, device=self.X_test.device)
        idx_map = {cell_type.item(): i for i, cell_type in enumerate(unique_cell_types)}
        for i in range(len(self.y_test)):
            cell_type_idx = idx_map[self.y_test[i].item()]
            X_sum[cell_type_idx] += self.X_test[i]
            X_count[cell_type_idx] += 1
        X_avg = X_sum / X_count.unsqueeze(1)
        cell_type_names = []
        for type_idx_tensor in unique_cell_types:
            type_idx = type_idx_tensor.item()
            cell_type_name = self.y_reverse_label_map.get(int(type_idx), f"Type_{type_idx}")
            cell_type_names.append(cell_type_name)
        X_avg_cpu = X_avg.cpu().numpy()
        df = pd.DataFrame(
            X_avg_cpu.T,  # Transpose to get genes as rows, cell types as columns
            index=pd.Index(self.genes, name='Gene'),
            columns=pd.Index(cell_type_names, name='Cell_Type')
        )
        
        # Save to input directory for future use
        df.to_csv(input_csv_path, float_format='%.2f')
        torch.save(X_avg.cpu(), input_pt_path)
        self.log.info(f"Cell type averages calculated and saved to input directory:")
        self.log.info(f"  CSV: {input_csv_path}")
        self.log.info(f"  PT: {input_pt_path}")
        self.log.info(f"Shape: {X_avg.shape} ({n_cell_types} cell types × {n_genes} genes)")
        
        # Create symlinks to output directory
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
        if os.path.exists(output_pt_path):
            os.remove(output_pt_path)
        
        os.symlink(input_csv_path, output_csv_path)
        os.symlink(input_pt_path, output_pt_path)
        self.log.info(f"Created symlinks to output directory:")
        self.log.info(f"  CSV: {output_csv_path} -> {input_csv_path}")
        self.log.info(f"  PT: {output_pt_path} -> {input_pt_path}")
        
        return X_avg, cell_type_names

    def calculate_bit_importance(self) -> Dict[str, np.ndarray]:
        """
        Calculate bit-wise importance scores using multiple criteria.
        
        Computes importance scores for each bit based on:
        1. Probe efficiency: probe strength per gene usage
        2. Decoder importance: magnitude of decoder weights
        3. Dynamic range: fold change between percentiles
        
        Returns:
            dict containing:
                - 'importance_scores' (np.ndarray): Combined importance scores (n_bit,)
                - 'sorted_indices' (np.ndarray): Bit indices sorted by importance (descending)
                - 'probe_efficiency' (np.ndarray): Probe efficiency scores (n_bit,)
                - 'decoder_importance' (np.ndarray): Decoder weight magnitudes (n_bit,)
                - 'dynamic_ranges' (np.ndarray): Dynamic range fold changes (n_bit,)
                - 'gene_usage_per_bit' (np.ndarray): Number of genes used per bit (n_bit,)
                - 'bit_strengths' (np.ndarray): Total probe count per bit (n_bit,)
                
        Note:
            Importance score = 0.4 * probe_efficiency + 0.3 * decoder_importance + 0.3 * dynamic_ranges
            All components are normalized to [0, 1] before combination
        """
        self.log.info("--- Calculating Bit Importance Scores ---")
        E_clean = self.get_E_clean()
        n_bits = E_clean.shape[1]
        with torch.no_grad():
            P_clean = self.project_clean(self.X_train, E_clean)
            median_val = P_clean.sum(1).clamp(min=1e-8).median()
            P_clean_sum_norm = median_val * P_clean / P_clean.sum(1).unsqueeze(1).clamp(min=1e-8)
            bit_dynamic_ranges = []
            bit_percentiles = []
            for bit_idx in range(n_bits):
                bit_values = P_clean_sum_norm[:, bit_idx]
                quantiles_tensor = torch.tensor([0.05, 0.15, 0.45, 0.55, 0.85, 0.95], device=bit_values.device)
                quant_results = torch.quantile(bit_values, quantiles_tensor)
                fold_change = quant_results[5] / quant_results[0].clamp(min=1e-8)  # p95 / p05
                bit_dynamic_ranges.append(fold_change.item())
                bit_percentiles.append((quant_results[0].item(), quant_results[2].item(), quant_results[4].item()))
        gene_usage_per_bit = (E_clean > 0.1).float().sum(dim=0)
        bit_strengths = E_clean.sum(dim=0)
        first_layer = self.decoder[0]
        if isinstance(first_layer, nn.Linear):
            decoder_weights = first_layer.weight.data
            decoder_importance = torch.abs(decoder_weights).sum(dim=0)
        else:
            decoder_importance = torch.zeros(n_bits, device=E_clean.device)
        dynamic_ranges = torch.tensor(bit_dynamic_ranges, device=E_clean.device)
        probe_efficiency = bit_strengths / gene_usage_per_bit.clamp(min=1)
        probe_efficiency_norm = probe_efficiency / probe_efficiency.max()
        decoder_importance_norm = decoder_importance / decoder_importance.max()
        dynamic_ranges_norm = dynamic_ranges / dynamic_ranges.max()
        importance_scores = (0.4 * probe_efficiency_norm + 
                           0.3 * decoder_importance_norm + 
                           0.3 * dynamic_ranges_norm)
        sorted_indices = torch.argsort(importance_scores, descending=True)
        self.log.info(f"Bit importance scores calculated for {n_bits} bits")
        self.log.info(f"Top 5 bits: {sorted_indices[:5].detach().cpu().numpy()}")
        self.log.info(f"Bottom 5 bits: {sorted_indices[-5:].detach().cpu().numpy()}")
        return {
            'importance_scores': importance_scores.detach().cpu().numpy(),
            'sorted_indices': sorted_indices.detach().cpu().numpy(),
            'probe_efficiency': probe_efficiency.detach().cpu().numpy(),
            'decoder_importance': decoder_importance.detach().cpu().numpy(),
            'dynamic_ranges': dynamic_ranges.detach().cpu().numpy(),
            'gene_usage_per_bit': gene_usage_per_bit.detach().cpu().numpy(),
            'bit_strengths': bit_strengths.detach().cpu().numpy()
        }

    def _setup_logging(self, user_parameters_path: Optional[str]) -> None:
        """
        Set up logging configuration for the CIPHER model.
        
        Configures logging to write to both console and a log file. The log file is
        named based on the user parameters file name and saved in the output directory.
        
        Args:
            user_parameters_path (Optional[str]): Path to user parameters file.
                Used to name the log file. If None, uses "default" as the name.
        """
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

    def _load_and_process_parameters(self, user_parameters_path: Optional[str]) -> None:
        """
        Load and process user-defined parameters from CSV file.
        
        Loads parameters from a CSV file and merges them with default parameters.
        Handles type conversion, path construction, and parameter validation.
        
        Args:
            user_parameters_path (Optional[str]): Path to CSV file with parameters.
                CSV should have 'values' column and parameter names as index.
                If None, uses only default parameters.
                
        Note:
            - Parameters ending in '_s' automatically create corresponding parameter without '_s'
            - File paths are constructed relative to input directory if not absolute
            - Integer parameters are converted from float if they are whole numbers
            - Sets PyTorch thread count based on n_cpu parameter
        """
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

    def _setup_output_and_symlinks(self, user_parameters_path: Optional[str]) -> None:
        """
        Set up output directory and create symlinks to input files.
        
        Creates the output directory if it doesn't exist, saves used parameters,
        and creates symlinks from output directory to input files for easy reference.
        
        Args:
            user_parameters_path (Optional[str]): Path to user parameters file.
                This file is also symlinked to the output directory.
                
        Note:
            Symlinks are created for:
            - Input data files (constraints, X_train, X_test, y_train, y_test, etc.)
            - User parameters file
            - Used parameters file (saved in output directory)
        """
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

    def convert_param_to_int(self, param_key: str) -> None:
        """
        Convert a parameter value to integer if it's a whole number.
        
        Args:
            param_key (str): Key of the parameter in self.I to convert
            
        Raises:
            KeyError: If parameter doesn't exist
            ValueError: If parameter value cannot be converted to integer
        """
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

    def _initialize_encoder(self) -> None:
        """
        Initialize the encoder with proper weight initialization.
        
        Creates an nn.Embedding layer that maps genes to bit contributions.
        Initializes weights in pre-activation space such that after applying
        the activation function and constraints, probe fractions fall within
        the specified initialization range [E_init_min, E_init_max].
        
        Note:
            - Encoder shape: (n_genes, n_bit)
            - Weights are initialized to produce probe fractions in [E_init_min, E_init_max]
            - Activation function is applied during forward pass (via get_E_clean)
            - Logs the final weight range and total probe count after initialization
        """
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

    def _initialize_decoder(self) -> None:
        """
        Initialize the decoder network with specified architecture.
        
        Creates a decoder network that maps bit projections to cell type logits.
        Architecture depends on decoder_n_lyr:
        - If decoder_n_lyr == 0: Single linear layer (n_bit -> n_categories)
        - If decoder_n_lyr > 0: Multiple hidden layers with BatchNorm, activation, and dropout
        
        Note:
            - Hidden dimension is 3 * n_bit
            - Activation function specified by decoder_act parameter
            - Dropout only applied if use_noise=1
            - Final layer always outputs n_categories logits
        """
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

    def _load_data(self) -> None:
        """
        Load and validate all data files required for training.
        
        Loads training and test data (gene expression and labels), processes labels,
        and optionally filters to central brain data if central_brain=1.
        
        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data shapes are inconsistent
            
        Note:
            - Remaps cell type labels to consecutive integers (no gaps)
            - If central_brain=1, filters data to cell types in central brain region
            - Validates that X_train and X_test have matching gene dimensions
            - Saves cell type averages after loading
        """
        def load_tensor(path: str, dtype: torch.dtype, device: str) -> torch.Tensor:
            """
            Load a tensor from a .pt file with error handling.
            
            Args:
                path (str): Path to the .pt file
                dtype (torch.dtype): Desired data type for the tensor
                device (str): Device to load tensor onto ('cpu' or 'cuda')
                
            Returns:
                torch.Tensor: Loaded tensor on specified device with specified dtype
                
            Raises:
                FileNotFoundError: If the file doesn't exist
            """
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
        y_converter_path = self.I['y_label_converter_path']
        self.log.info(f"Loading y label converter from: {y_converter_path}")
        y_converter_df = pd.read_csv(y_converter_path, index_col=0) 
        y_converter_dict = dict(zip(y_converter_df.index, y_converter_df['label'])) # Readable to numerical
        self.log.info(f"{y_converter_dict}")
        y_reverse_converter_dict = {v: k for k, v in y_converter_dict.items()} # Numerical to readable
        # Central Brain Only
        if self.I['central_brain'] == 1:
            self.log.info(f"Selecting cell types for Central Brain Only.")
            import anndata
            # issues if another code is using this file
            file_loaded = False
            attempt = 0
            while not file_loaded:
                try:
                    adata = anndata.read_h5ad(os.path.join(self.I['input'],'minimal_spatial_data.h5ad'),backed='r')
                    file_loaded = True
                except:
                    self.log.error(f"Failed to load adata from {os.path.join(self.I['input'],'minimal_spatial_data.h5ad')}. Likely due to another code using this file.")
                    time.sleep(60)
                    attempt += 1
                    if attempt > 50:
                        raise Exception(f"Failed to load adata from {os.path.join(self.I['input'],'minimal_spatial_data.h5ad')}. Max attempts reached.")
            pivot_table = pd.read_csv(os.path.join(self.I['input'],'pivot_table.csv'),index_col=0)
            cluster_alias_to_subclass = dict(zip(pivot_table.index.astype(str),pivot_table['subclass']))
            ccf_x_min_threshold = 4.5
            ccf_x_max_threshold = 9.5
            self.log.info(f"Selecting cell types for Central Brain Only. CCF x range: [{ccf_x_min_threshold}, {ccf_x_max_threshold}]")
            good_cluster_aliases = adata.obs['cluster_alias'].map(cluster_alias_to_subclass)[(adata.obs['ccf_x']>ccf_x_min_threshold) & (adata.obs['ccf_x']<ccf_x_max_threshold)].value_counts()
            bad_cluster_aliases = adata.obs['cluster_alias'].map(cluster_alias_to_subclass)[(adata.obs['ccf_x']<ccf_x_min_threshold) | (adata.obs['ccf_x']>ccf_x_max_threshold)].value_counts()
            del adata, pivot_table, cluster_alias_to_subclass
            good_cluster_aliases.name = 'good'
            bad_cluster_aliases.name = 'bad'
            cluster_aliases = pd.concat([good_cluster_aliases, bad_cluster_aliases],axis=1).fillna(0)
            cluster_aliases = cluster_aliases/cluster_aliases.sum(axis=1).values[:,None]
            cluster_aliases = cluster_aliases[cluster_aliases['good']>0.05]
            selected_y_labels_readable = list(cluster_aliases.index)
            self.log.info(f"Selected {len(selected_y_labels_readable)} cell types for training and testing For Central Brain Only.")
            selected_y_labels = [y_converter_dict[readable] for readable in selected_y_labels_readable if readable in y_converter_dict.keys()]
            self.log.info(f"Selected {len(selected_y_labels)} cell types for training and testing For Central Brain Only.")
            
            # cluster_aliases = cluster_aliases.loc[[i for i in cluster_aliases.index if i in y_converter_dict.keys()]]
            # self.log.info(f"Selected {np.sum(cluster_aliases>0.05)} cell types for training and testing For Central Brain Only.")
            # self.log.info(cluster_aliases[cluster_aliases>0.05].index)
            # self.log.info(y_converter_dict)
            # selected_y_labels = [y_converter_dict[i] for i in cluster_aliases[cluster_aliases['good']>0.05].index]
            # self.log.info(f"Selected {len(selected_y_labels)} cell types for training and testing For Central Brain Only.")
            test_m = np.isin(self.y_test,selected_y_labels)
            train_m = np.isin(self.y_train,selected_y_labels)
            self.log.info(f"Selected {np.mean(train_m)} training samples and {np.mean(test_m)} testing samples for Central Brain Only.")
            self.X_train = self.X_train[train_m]
            self.y_train = self.y_train[train_m]
            self.X_test = self.X_test[test_m]
            self.y_test = self.y_test[test_m]

        # Process labels update numerics to not have gaps
        all_y_labels = torch.cat((self.y_train, self.y_test))
        unique_y_labels = torch.unique(all_y_labels)
        self.updated_y_label_map = {old_numeric.item(): new_numeric for new_numeric, old_numeric in enumerate(torch.unique(all_y_labels))} # old Numerical to new Numerical
        self.y_train = torch.tensor([self.updated_y_label_map[old_numeric.item()] for old_numeric in self.y_train], dtype=torch.long, device=self.I['device'])
        self.y_test = torch.tensor([self.updated_y_label_map[old_numeric.item()] for old_numeric in self.y_test], dtype=torch.long, device=self.I['device'])
        self.y_label_map = {y_reverse_converter_dict[old_numeric.item()]:self.updated_y_label_map[old_numeric.item()] for old_numeric in all_y_labels}
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
        
        # Save cell type averages after data is loaded
        self.save_cell_type_averages()

    def _load_constraints(self) -> None:
        """
        Load gene constraints from CSV file.
        
        Loads maximum allowed probes per gene from a CSV file. If gene_constraint_wt=0,
        sets all constraints to 1000 (effectively unconstrained).
        
        Raises:
            KeyError: If 'constraints' column is missing from the CSV file
            
        Note:
            - CSV should have gene names as index and 'constraints' column
            - Sets self.genes and self.n_genes based on loaded constraints
            - Constraints are stored as torch.Tensor on the specified device
        """
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

    def _load_pretrained_model(self) -> None:
        """
        Load pretrained model state, learning stats, and gene mask if available.
        
        Searches for saved model state files in the output directory and loads the
        most recent one. Also loads learning statistics and gene mask if available.
        Sets training_completed flag if training appears to be finished.
        
        Note:
            - Searches for files ending in 'model_state.pt'
            - Looks for 'learning_stats.csv' in the same or parent directory
            - If gene_mask.pt exists, applies gene filtering
            - Sets is_initialized_from_file flag to True if model loaded successfully
            - Reinitializes encoder if gene mask is applied (to match filtered dimensions)
        """
        model_files = []
        for root, dirs, files in os.walk(self.I['output']):
            if 'trimmed' in root:
                continue
            # Look for model state files in results directory
            results_dirs = [d for d in dirs if d == 'results']
            if results_dirs:
                results_path = os.path.join(root, 'results')
                if os.path.exists(results_path):
                    results_files = os.listdir(results_path)
                    model_state_files = [os.path.join(results_path, f) for f in results_files if f.endswith('model_state.pt')]
                    if len(model_state_files) > 0:
                        model_files.extend(model_state_files)
            else:
                # Fallback to old structure
                model_state_files = [os.path.join(root,i) for i in files if i.endswith('model_state.pt')]
                learning_stats_files = [i for i in files if i.endswith('learning_stats.csv')]
                if (len(model_state_files)>0) and (len(learning_stats_files)>0):
                    model_files.extend(model_state_files) 
        # Sort by modification time to get the newest
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            newest_model_path = model_files[0]
            model_dir = os.path.dirname(newest_model_path)
            self.log.info(f"Found model state file: {newest_model_path}. Attempting to load.")
            try:
                loaded_state_dict = torch.load(newest_model_path, map_location=self.I['device'])
                missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=False)
                if missing_keys: self.log.warning(f"Missing keys when loading state_dict: {missing_keys}")
                if unexpected_keys: self.log.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
                self.to(self.I['device']) 
                self.is_initialized_from_file = True
                self.log.info("Successfully loaded model state from file (strict=False).")
                # Look for learning stats in the same directory as the model
                learning_stats_path = os.path.join(model_dir, 'learning_stats.csv')
                # Also check if we're in a results directory and look for learning stats there
                if not os.path.exists(learning_stats_path) and model_dir.endswith('results'):
                    # We're already in the results directory, so learning_stats.csv should be here
                    pass
                elif not os.path.exists(learning_stats_path) and os.path.exists(os.path.join(model_dir, '..', 'learning_stats.csv')):
                    # Check parent directory for learning stats
                    learning_stats_path = os.path.join(model_dir, '..', 'learning_stats.csv')
                elif not os.path.exists(learning_stats_path):
                    # Check if there's a results directory with learning stats
                    results_dir = os.path.join(os.path.dirname(model_dir), 'results')
                    if os.path.exists(results_dir):
                        results_learning_stats_path = os.path.join(results_dir, 'learning_stats.csv')
                        if os.path.exists(results_learning_stats_path):
                            learning_stats_path = results_learning_stats_path
                
                if os.path.exists(learning_stats_path):
                    try:
                        self.learning_stats = pd.read_csv(learning_stats_path, index_col=0)
                        self.log.info(f"Successfully loaded learning stats from {learning_stats_path}")
                        max_iteration = -1
                        for key in self.learning_stats.index:
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
                    self.log.info(f"No learning stats file found in {model_dir}. Assuming training was not completed.")
                    self.training_completed = False
                
                # Check if gene mask exists in the same directory and load it if present
                gene_mask_path = os.path.join(model_dir, 'gene_mask.pt')
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
                    
            except Exception as e:
                self.log.error(f"Failed to load model state from {newest_model_path}: {e}. Model will use fresh initial weights.")
                self.is_initialized_from_file = False
        else:
            self.log.info("No existing model state file found. Model will use fresh initial weights.")
            self.is_initialized_from_file = False
        
        # If still not initialized from file, use fresh weights
        if not self.is_initialized_from_file:
            self.log.info("No existing model state found. Model will use fresh initial weights.")
            self.training_completed = False
            return

    def _update_parameters_for_iteration(self, iteration: int, n_iterations: int) -> None:
        """
        Update dynamic parameters (_s/_e parameters) based on training progress.
        
        Linearly interpolates between start (_s) and end (_e) values for all parameters
        that have both _s and _e versions. The interpolation is controlled by the
        saturation parameter, which determines when the final values are reached.
        
        Args:
            iteration (int): Current training iteration
            n_iterations (int): Total number of training iterations
            
        Note:
            - Progress is calculated as iteration / (n_iterations - 1)
            - Progress is scaled by saturation: progress = progress / saturation
            - If saturation=0, immediately uses final (_e) values
            - Parameters updated: brightness, dynamic_fold, separation_fold, lr, sparsity,
              and all noise parameters (X_drp, X_noise, E_drp, E_noise, P_drp, P_noise, etc.)
        """
        progress = iteration / (n_iterations - 1) if n_iterations > 1 else 0
        if self.I.get('saturation', 1.0) == 0:
            progress = 1.0
        else:
            progress = np.clip(progress / self.I.get('saturation', 1.0), 0, 1)
        parameters_to_update = [i.replace('_s', '') for i in self.I if i.endswith('_s')]
        for param in parameters_to_update:
            self.I[param] = (self.I[f'{param}_s'] + (self.I[f'{param}_e'] - self.I[f'{param}_s']) * progress)

    def _setup_optimizer(self) -> None:
        """
        Set up the Adam optimizer for training encoder and decoder.
        
        Creates separate parameter groups for encoder and decoder, both with the
        same learning rate. Uses betas=(0.9, 0.9) for momentum.
        
        Note:
            - Encoder and decoder can have different learning rates if modified later
            - Optimizer state can be saved/loaded for resuming training
        """
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

    def _get_training_batch(self, iteration: int, batch_size: int, n_train_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of training data for the current iteration.
        
        Args:
            iteration (int): Current training iteration (not used, for consistency)
            batch_size (int): Desired batch size. If 0 or >= n_train_samples, uses full dataset
            n_train_samples (int): Total number of training samples
            
        Returns:
            tuple containing:
                - X_batch (torch.Tensor): Gene expression batch of shape (batch_size, n_genes)
                - y_batch (torch.Tensor): Cell type labels batch of shape (batch_size,)
        """
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

    def _check_gradient_health(self, iteration: int) -> bool:
        """
        Check for NaN or Inf values in model gradients.
        
        Args:
            iteration (int): Current training iteration (for logging)
            
        Returns:
            bool: True if NaN/Inf detected, False otherwise
            
        Note:
            If NaN/Inf is detected, zeros gradients and logs a warning.
            The training loop should then revert to a previous state.
        """
        nan_detected = False
        for name, param in self.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                nan_detected = True
                self.log.warning(f"NaNs or Infs detected in gradients of model parameter '{name}' at iteration {iteration}. Skipping step and attempting revert.")
                self.optimizer_gen.zero_grad() 
                break
        return nan_detected

    def _apply_gradient_clipping(self) -> None:
        """
        Apply gradient clipping to prevent exploding gradients.
        
        Clips gradients to have maximum norm of gradient_clip. Only applies if
        gradient_clip > 0.
        
        Note:
            Uses torch.nn.utils.clip_grad_norm_ with max_norm=gradient_clip
        """
        if self.I.get('gradient_clip', 1.0)  > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.I.get('gradient_clip', 1.0) )

    def _handle_weight_perturbation(self, iteration: int, report_rt: int, delayed_perturbation_iter: Optional[int]) -> Optional[int]:
        """
        Handle weight perturbation scheduling with delay logic.
        
        Determines when to perturb encoder weights. If perturbation would occur
        close to a test evaluation, delays it until after the evaluation.
        
        Args:
            iteration (int): Current training iteration
            report_rt (int): How often test evaluations occur
            delayed_perturbation_iter (Optional[int]): Previously scheduled delayed perturbation,
                or None if no delay is scheduled
                
        Returns:
            Optional[int]: Next delayed perturbation iteration, or None
        """
        if self.I['E_perturb_rt'] > 0:
            if iteration % self.I['E_perturb_rt'] == 0:
                next_test_iter = ((iteration // report_rt) + 1) * report_rt
                if next_test_iter - iteration <= 50:
                    self.log.info(f"Delaying weight perturbation from iteration {iteration} to after test at iteration {next_test_iter}")
                    delayed_perturbation_iter = next_test_iter
                else:
                    self.perturb_E()
        return delayed_perturbation_iter

    def _update_best_model(self, iteration: int, total_loss: torch.Tensor) -> None:
        """
        Update the best model state if current loss is lower.
        
        Args:
            iteration (int): Current training iteration
            total_loss (torch.Tensor): Current total loss value
            
        Note:
            - Only updates if loss is finite and lower than previous best
            - Saves model state_dict to CPU to save memory
            - Logs when a new best model is found
        """
        current_loss_item = total_loss.item()
        if not np.isnan(current_loss_item) and current_loss_item < self.best_loss:
            self.best_loss = current_loss_item
            self.best_model_state_dict = {k: v.cpu().detach().clone() for k, v in self.state_dict().items()}
            self.best_iteration = iteration
            self.log.info(f"*** New best model found at iteration {iteration} (Train Loss: {self.best_loss:.4f}) ***")

    def _revert_to_previous_state(self, iteration: int) -> bool:
        """
        Revert model and optimizer to a previous saved state.
        
        Called when NaN/Inf gradients are detected. Loads the most recent saved
        state from before the current iteration.
        
        Args:
            iteration (int): Current iteration where NaN/Inf was detected
            
        Returns:
            bool: True if revert succeeded, False otherwise
            
        Raises:
            ValueError: If no previous state is available to revert to
            
        Note:
            - Reinitializes optimizer after loading state
            - Moves optimizer state tensors to correct device
            - Logs a status message in learning_stats
        """
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
            # Initialize row for this iteration if it doesn't exist
            if str(iteration) not in self.learning_stats.index:
                self.learning_stats.loc[str(iteration)] = pd.Series(dtype=object)
            self.learning_stats.loc[str(iteration), 'status'] = f'Reverted from NaN at {iteration}'
            return True
        else:
            self.log.error(f"NaNs/Infs detected in gradients at iter {iteration}, but no previous state found. Stopping.")
            raise ValueError("NaNs/Infs encountered and cannot revert.")

    def _evaluate_on_test_set(self, iteration: int, last_report_time: float, last_report_iteration: int) -> tuple[float, int]:
        """
        Evaluate model on test set and log detailed results.
        
        Args:
            iteration (int): Current training iteration
            last_report_time (float): Timestamp of last report (for timing calculations)
            last_report_iteration (int): Iteration of last report
            
        Returns:
            tuple containing:
                - current_time (float): Current timestamp
                - iteration (int): Current iteration
                
        Note:
            - Computes test loss and all statistics
            - Logs average time per iteration since last report
            - Logs all test statistics with formatted output
        """
        self.eval()
        with torch.no_grad():
            total_test_loss, test_stats = self.calculate_loss(
                self.X_test, self.y_test, iteration, suffix='_test')
            test_stats['total_loss_test'] = total_test_loss.item()
            # Add test stats to the DataFrame row
            for key, value in test_stats.items():
                self.learning_stats.loc[str(iteration), key] = value
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
        if train_loss_key in self.learning_stats.columns and str(iteration) in self.learning_stats.index:
            train_loss_value = self.learning_stats.loc[str(iteration), train_loss_key]
            if pd.notna(train_loss_value):
                log_msg = f'{train_loss_key}: {round(train_loss_value, 4)}'
                self.log.info(log_msg)
        for name, item in test_stats.items():
            if isinstance(item, (float, int, np.number)) and not np.isnan(item):
                log_msg = f'{name}: {round(float(item), 4)}'
            else:
                log_msg = f'{name}: {item}'
            self.log.info(log_msg)
        self.log.info('------------------')
        
        return current_time, iteration

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint with model state, optimizer state, and learning stats.
        
        Saves everything needed to resume training from this point. Creates a new
        results directory, removing any existing one.
        
        Args:
            iteration (int): Current training iteration
            
        Note:
            - Saves to 'results/checkpoint.pt' and 'results/learning_stats.csv'
            - Checkpoint includes: iteration, model_state_dict, optimizer_state_dict,
              best_loss, best_iteration, and all parameters
        """
        try:
            results_dir = os.path.join(self.I['output'], 'results')
            # Remove existing results directory if it exists
            if os.path.exists(results_dir):
                import shutil
                shutil.rmtree(results_dir)
            os.makedirs(results_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(results_dir, 'checkpoint.pt')
            learning_stats_path = os.path.join(results_dir, 'learning_stats.csv')
            
            # Save model state
            torch.save({
                'iteration': iteration,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer_gen.state_dict(),
                'best_loss': self.best_loss,
                'best_iteration': self.best_iteration,
                'parameters': self.I
            }, checkpoint_path)
            
            # Save learning stats as CSV
            self.learning_stats.to_csv(learning_stats_path)
            
            self.log.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")
            
        except Exception as e:
            self.log.error(f"Failed to save checkpoint at iteration {iteration}: {e}")

    def save_eval_and_viz(self, iteration: int, eval_dir: Optional[str] = None) -> None:
        """
        Save evaluation results and visualizations to results directory.
        
        Runs full evaluation and visualization, then saves the model state.
        Creates a new results directory, removing any existing one.
        
        Args:
            iteration (int): Current training iteration
            eval_dir (Optional[str]): Subdirectory name for evaluation (not currently used)
            
        Note:
            - Temporarily changes output directory to results/
            - Runs evaluate() and visualize()
            - Saves model state via save_model()
            - Restores original output directory after completion
        """
        try:
            results_dir = os.path.join(self.I['output'], 'results')
            # Remove existing results directory if it exists
            if os.path.exists(results_dir):
                import shutil
                shutil.rmtree(results_dir)
            os.makedirs(results_dir, exist_ok=True)
            
            original_output = self.I['output']
            self.I['output'] = results_dir
            self.log.info(f"Running evaluation and visualization for iteration {iteration}")
            self.evaluate()
            self.visualize(show_plots=False)
            self.save_model()
            self.I['output'] = original_output
            self.log.info(f"Saved evaluation, visualization, and model state for iteration {iteration} to {results_dir}")
            
        except Exception as e:
            self.log.error(f"Failed to save evaluation and visualization for iteration {iteration}: {e}")
            # Restore original output directory even if there's an error
            self.I['output'] = original_output



    def save_model(self) -> None:
        """
        Save final model state and constrained encoder weights.
        
        Saves the best model (if best_model=1) or final model state, along with
        learning statistics. Also enforces constraints on final encoder weights
        and saves the constrained E matrix.
        
        Note:
            - Saves to 'results/model_state.pt' and 'results/learning_stats.csv'
            - Creates 'E_constrained.csv' and 'E_constrained.pt' with final probe allocations
            - Constraints are enforced by scaling down genes that exceed limits
            - Logs final evaluation statistics
        """
        # Ensure we're saving to the results directory
        if not self.I['output'].endswith('results'):
            output_dir = os.path.join(self.I['output'], 'results')
            # Remove existing results directory if it exists
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.I['output']
            
        if self.I['best_model'] == 0:
            self.best_model_state_dict = None
            self.log.info("Best model turned off. Saving the final iteration state.")
        
        # Take the last index and add 1 for the final iteration
        final_iter_key = int(self.learning_stats.index[-1]) + 1
        # Initialize row for final iteration if it doesn't exist
        if final_iter_key not in self.learning_stats.index:
            self.learning_stats.loc[final_iter_key] = pd.Series(dtype=object)
            
        if self.best_model_state_dict is not None:
            self.log.info(f"Loading best model state from iteration {self.best_iteration} (Train Loss: {self.best_loss:.4f}) before final save.")
            try:
                missing_keys, unexpected_keys = self.load_state_dict(self.best_model_state_dict, strict=False)
                if missing_keys: self.log.warning(f"Missing keys when loading best state_dict: {missing_keys}")
                if unexpected_keys: self.log.warning(f"Unexpected keys when loading best state_dict: {unexpected_keys}")
                self.to(self.I['device'])
                self.log.info(f"Successfully loaded best model state for final saving.")
                self.eval()
                with torch.no_grad():
                    total_final_loss, final_stats_dict = self.calculate_loss(
                        self.X_test, self.y_test, iteration=str(final_iter_key), suffix='_test'
                        )
                # Add final stats to the DataFrame row
                for key, value in final_stats_dict.items():
                    self.learning_stats.loc[final_iter_key, key] = value
                self.learning_stats.loc[final_iter_key, 'total_loss_test_avg'] = total_final_loss.item()
            except Exception as e:
                self.log.error(f"Failed to load best model state before saving: {e}. Saving the final iteration state instead.")
        else:
            self.log.warning("No best model state was saved during training. Saving the final iteration state.")
        model_path = os.path.join(output_dir, 'model_state.pt')
        learning_stats_path = os.path.join(output_dir, 'learning_stats.csv')
        try:
            torch.save(self.state_dict(), model_path)
            self.log.info(f"Final model state dictionary saved to {model_path}")
            # Save learning stats as CSV
            self.learning_stats.to_csv(learning_stats_path)
            self.log.info(f"Learning stats saved to {learning_stats_path}")
        except Exception as e:
            self.log.error(f"Failed to save final model state: {e}")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        red_start = "\033[91m"; reset_color = "\033[0m"
        log_prefix = f"--- Final Eval Stats (Global Test Set) at {now_str} ---"
        self.log.info(log_prefix)
        final_row = self.learning_stats.loc[final_iter_key]
        for name, item in final_row.items():
            if not pd.isna(item) and isinstance(item, (float, int, np.number)) and not np.isnan(item):
                log_msg = f'{name}: {round(float(item), 4)}'
                self.log.info(log_msg)
            elif pd.isna(item)==False:
                log_msg = f'{name}: {item}'
                self.log.info(log_msg)
            else:
                continue  # Skip NaN values
        self.log.info('------------------')
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
            e_csv_path = os.path.join(output_dir, 'E_constrained.csv')
            e_pt_path = os.path.join(output_dir, 'E_constrained.pt')
            pd.DataFrame(self.E.cpu().numpy().astype(int), index=self.genes).to_csv(e_csv_path)
            torch.save(self.E.cpu(), e_pt_path)
            self.log.info(f"Final constrained E matrix saved to {e_csv_path} and {e_pt_path}")
            self.log.info(f"Final constrained probe count: {self.E.sum().item():.2f}")
        try:
            learning_curve_path = os.path.join(output_dir, 'learning_curve.csv')
            self.learning_stats.to_csv(learning_curve_path)
            self.log.info(f"Learning curve data saved to {learning_curve_path}")
        except Exception as e:
            self.log.error(f"Failed to save learning curve: {e}")

    def trim_bits(self, n_keep: Optional[int] = None) -> np.ndarray:
        """
        Trim the model to keep only the top n_keep most important bits.
        
        Removes the least important bits from both encoder and decoder, reducing
        model complexity. Bit importance is calculated using calculate_bit_importance().
        
        Args:
            n_keep (Optional[int]): Number of bits to keep. If None, keeps all but 1 bit.
            
        Returns:
            np.ndarray: Indices of kept bits (sorted)
            
        Note:
            - Updates encoder weights to remove columns for trimmed bits
            - Updates decoder first layer to remove input features for trimmed bits
            - Updates n_bit parameter
            - Logs which bits were kept and removed
        """
        bit_info = self.calculate_bit_importance()
        sorted_indices = bit_info['sorted_indices']
        n_bits = len(sorted_indices)
        if n_keep is None:
            n_keep = n_bits - 1
        if n_keep >= n_bits:
            self.log.info(f"Requested to keep {n_keep} bits, but model only has {n_bits} bits. No trimming performed.")
            return np.sort(sorted_indices)
        keep_indices = np.sort(sorted_indices[:n_keep])
        remove_indices = np.setdiff1d(np.arange(n_bits), keep_indices)
        # Update encoder weights
        with torch.no_grad():
            self.encoder.weight = torch.nn.Parameter(self.encoder.weight[:, keep_indices])
        # Update decoder first layer weights (assume nn.Linear as first layer)
        first_layer = self.decoder[0]
        if isinstance(first_layer, nn.Linear):
            old_weight = first_layer.weight.data
            new_weight = old_weight[:, keep_indices]
            new_first_layer = nn.Linear(len(keep_indices), first_layer.out_features)
            new_first_layer.weight = torch.nn.Parameter(new_weight)
            self.decoder[0] = new_first_layer.to(self.I['device'])
        else:
            self.log.warning("Decoder first layer is not nn.Linear. Skipping decoder weight trimming.")
        # Update n_bit
        self.I['n_bit'] = self.encoder.weight.shape[1]
        self.log.info(f"Trimmed to {self.I['n_bit']} bits. Kept bits: {keep_indices}. Removed bits: {remove_indices}")

    def iterative_trim_to_n_bits(self, target_n_bits: int = 12, max_iterations: int = 10) -> None:
        """
        Iteratively trim bits down to target_n_bits over multiple iterations.
        
        Progressively reduces the number of bits, retraining the decoder after each
        trim to maintain performance. Creates a plot showing accuracy vs bit count.
        
        Args:
            target_n_bits (int): Target number of bits to reach
            max_iterations (int): Maximum number of trimming iterations
            
        Note:
            - Trims bits using linear spacing from current n_bit to target_n_bits
            - Retrains decoder for 10k iterations after each trim
            - Saves evaluation and visualization after each trim
            - Creates accuracy_vs_bits plot showing performance degradation
            - Saves results to 'trimmed' subdirectory
        """
        if self.I['n_bit'] > target_n_bits:
            n_keeps = np.unique(np.linspace(target_n_bits, self.I['n_bit']-1, max_iterations).astype(int))[::-1]
            bit_counts = []
            accuracies = []
            for n_keep in n_keeps:
                self.log.info(f"Trimming from {self.I['n_bit']} to {n_keep} bits")
                try:
                    self.trim_bits(n_keep=n_keep)
                    self.train_decoder_only(n_iterations=10000)
                    self.save_eval_and_viz(iteration=n_keep, eval_dir='trimmed')
                    self.eval()
                    with torch.no_grad():
                        E_clean = self.get_E_clean()
                        P_clean = self.project_clean(self.X_test, E_clean)
                        y_predict, accuracy, _ = self.decode(P_clean, self.y_test)
                        bit_counts.append(self.I['n_bit'])
                        accuracies.append(accuracy.item())
                        self.log.info(f"Accuracy at {self.I['n_bit']} bits: {accuracy.item():.4f}")
                except Exception as e:
                    self.log.error(f"Error during trimming to {n_keep} bits: {e}")
                    break
            if len(bit_counts) > 1:
                self.plot_accuracy_vs_bits(bit_counts, accuracies)
            self.log.info(f"Iterative trimming complete. Final n_bits: {self.I['n_bit']}")

    def plot_accuracy_vs_bits(self, bit_counts: List[int], accuracies: List[float]) -> None:
        """
        Plot accuracy vs bit count after iterative trimming.
        
        Creates a line plot showing how accuracy changes as bits are removed.
        Annotates the first, last, and maximum accuracy points.
        
        Args:
            bit_counts (List[int]): Number of bits at each trimming step
            accuracies (List[float]): Test accuracy at each trimming step
            
        Note:
            - Saves plot to 'trimmed/accuracy_vs_bits.pdf'
            - Saves data to 'trimmed/accuracy_vs_bits.csv'
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(bit_counts, accuracies, 'bo-', linewidth=2, markersize=8, label='Test Accuracy')
            plt.xlabel('Number of Bits')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Bit Count After Trimming')
            plt.grid(True, alpha=0.3)
            plt.legend()
            for i, (bits, acc) in enumerate(zip(bit_counts, accuracies)):
                if i == 0 or i == len(bit_counts) - 1 or acc == max(accuracies):
                    plt.annotate(f'{acc:.3f}', (bits, acc), 
                               xytext=(5, 5), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            plt.tight_layout()
            plot_path = os.path.join(self.I['output'], 'trimmed', 'accuracy_vs_bits.pdf')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log.info(f"Accuracy vs bits plot saved to {plot_path}")
            data_df = pd.DataFrame({
                'bit_count': bit_counts,
                'accuracy': accuracies
            })
            csv_path = os.path.join(self.I['output'], 'trimmed', 'accuracy_vs_bits.csv')
            data_df.to_csv(csv_path, index=False)
            self.log.info(f"Accuracy vs bits data saved to {csv_path}")
        except Exception as e:
            self.log.error(f"Error creating accuracy vs bits plot: {e}")

    def train_decoder_only(self, n_iterations: int = 1000) -> None:
        """
        Train only the decoder while keeping encoder weights fixed.
        
        Useful for fine-tuning the decoder after encoder weights have been learned
        or after bit trimming. Only optimizes the categorical classification loss.
        
        Args:
            n_iterations (int): Number of training iterations
            
        Note:
            - Sets encoder learning rate to 0
            - Only uses categorical loss (no probe/hybridization/measurability losses)
            - Evaluates on test set every 100 iterations
            - Restores encoder learning rate after training
        """
        self.log.info(f"--- Starting Decoder-Only Training for {n_iterations} iterations ---")
        if not hasattr(self, 'optimizer_gen'):
            self._setup_optimizer()
        # overwrite the learning stats
        self.learning_stats = pd.DataFrame()
        # Set encoder learning rate to 0
        for param_group in self.optimizer_gen.param_groups:
            if 'encoder' in str(param_group['params'][0]):
                param_group['lr'] = 0.0
        self.train()
        for iteration in range(n_iterations):
            X, y = self._get_training_batch(iteration, self.I['batch_size'], self.X_train.shape[0])
            self.optimizer_gen.zero_grad()
            E_noisy = self.get_E()
            P_noisy = self.project(X, E_noisy)
            y_predict, accuracy, total_loss = self.decode(P_noisy, y)
            total_loss.backward()
            self.optimizer_gen.step()
            # Initialize row for this iteration if it doesn't exist
            if str(iteration) not in self.learning_stats.index:
                self.learning_stats.loc[str(iteration)] = pd.Series(dtype=object)
            self.learning_stats.loc[str(iteration), 'accuracy_train'] = accuracy.item()
            self.learning_stats.loc[str(iteration), 'total_loss_train'] = total_loss.item()
            # Evaluate on test set periodically
            if iteration % 100 == 0 or iteration == n_iterations - 1:
                self.eval()
                with torch.no_grad():
                    E_clean = self.get_E_clean()
                    P_clean = self.project_clean(self.X_test, E_clean)
                    y_predict_test, accuracy_test, total_loss_test = self.decode(P_clean, self.y_test)
                    self.learning_stats.loc[str(iteration), 'accuracy_test'] = accuracy_test.item()
                    self.learning_stats.loc[str(iteration), 'total_loss_test'] = total_loss_test.item()
                self.train()
        self.eval()
        # Restore encoder learning rate
        for param_group in self.optimizer_gen.param_groups:
            if 'encoder' in str(param_group['params'][0]):
                param_group['lr'] = self.I['lr']
        self.log.info("--- Decoder-Only Training Complete ---")

# Helper function for smooth, non-negative penalty
def swish(fold: torch.Tensor, scale: float = 3.0, shift: float = -1.278, offset: float = 0.3) -> torch.Tensor:
    """
    Smooth, non-negative penalty function for soft constraints.
    
    Implements a Swish-like activation function that provides smooth penalties
    for constraint violations. The function is non-negative and has minimum at fold=0.
    
    Args:
        fold (torch.Tensor): Normalized constraint violations, typically computed as
            (value - bound) / bound. Positive values indicate violations.
        scale (float): Controls sharpness of penalty ramp-up. Higher values create
            steeper penalties. Default: 3.0
        shift (float): Shifts the function so minimum is at fold=0. Default: -1.278
        offset (float): Makes minimum value zero. Default: 0.3
            
    Returns:
        torch.Tensor: Penalty values, minimum at fold=0, non-negative.
            Shape matches input fold tensor.
            
    Note:
        Formula: z = scale * fold + shift
        Result: z * sigmoid(z) + offset
        This ensures smooth, differentiable penalties that are zero when fold=0.
    """
    z = scale * fold + shift
    return z * torch.sigmoid(z) + offset

def sanitize_filename(name: str) -> str:
    """
    Remove or replace characters invalid for filenames.
    
    Sanitizes a string to be safe for use as a filename by replacing
    problematic characters with underscores and removing others.
    
    Args:
        name (str): Original filename string that may contain invalid characters
        
    Returns:
        str: Sanitized filename safe for filesystem use
        
    Note:
        - Replaces whitespace, slashes, backslashes, and colons with underscores
        - Removes characters: < > : " | ? *
        - Strips leading/trailing whitespace
    """
    name = name.strip()
    name = re.sub(r'[\s/\\:]+', '_', name)
    name = re.sub(r'[<>:"|?*]+', '', name)
    return name

def plot_weight_matrix(df: pd.DataFrame, plot_path: str, title: str = "Weight Matrix", cmap: str = 'coolwarm', log_scale: bool = True, log: Optional[logging.Logger] = None) -> None:
    """
    Plot a weight matrix as a heatmap with optional log scaling and sorting.
    
    Creates a clustered heatmap visualization of a weight matrix, useful for
    visualizing encoder or decoder weights. Sorts rows by maximum weight magnitude.
    
    Args:
        df (pd.DataFrame): Weight matrix with rows as outputs and columns as inputs
        plot_path (str): Path where the plot will be saved (PDF format)
        title (str): Plot title. Default: "Weight Matrix"
        cmap (str): Colormap name for the heatmap. Default: 'coolwarm'
        log_scale (bool): Whether to apply log10 scaling to weights. Default: True
        log (Optional[logging.Logger]): Logger instance for logging messages.
            If None, creates a default logger.
            
    Note:
        - Sorts rows by maximum absolute weight value (descending)
        - Applies log10(x + 1) transformation if log_scale=True
        - Disables row/column clustering for deterministic ordering
        - Saves as PDF with 200 DPI
    """
    if log is None:
        log = logging.getLogger("WeightMatrixPlotter")
    
    try:
        import seaborn as sns
        
        # Sort by maximum weight magnitude for each output
        sorted_idx = df.abs().max(axis=1).sort_values(ascending=False).index
        df_sorted = df.loc[sorted_idx]
        
        # Apply log scaling if requested
        plot_data = np.log10(df_sorted + 1) if log_scale else df_sorted
        
        plt.figure(figsize=(10, 10))
        if cmap == 'coolwarm':
            sns.clustermap(
                plot_data,
                cmap=cmap,
                center=0,
                col_cluster=False,
                row_cluster=False,
                figsize=(10, 10),
                yticklabels=[]
            )
        else:
            sns.clustermap(
                plot_data,
                cmap=cmap,
                col_cluster=False,
                row_cluster=False,
                figsize=(10, 10),
                yticklabels=[]
            )
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        log.info(f"Weight matrix saved to {plot_path}")
        
    except Exception as e:
        log.error(f"Failed to plot weight matrix: {e}")

def generate_intuitive_ticks(min_val: float, max_val: float, num_ticks: int = 5) -> np.ndarray:
    """
    Generate intuitive tick values that are easy to read.
    
    Creates tick values using "nice" numbers (1, 2, 2.5, 5 times powers of 10)
    that are close to the desired number of ticks and easy for humans to read.
    
    Args:
        min_val (float): Minimum value for the axis
        max_val (float): Maximum value for the axis
        num_ticks (int): Desired number of ticks. Default: 5
        
    Returns:
        np.ndarray: Array of tick values between min_val and max_val
        
    Note:
        - Uses powers of 10 multiplied by 1, 2, 2.5, or 5
        - Selects step size closest to desired number of ticks
        - Handles edge case where min_val == max_val
    """
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

def plot_projection_space_density(P: np.ndarray, y_labels: np.ndarray, plot_path: str, sum_norm: bool = False, log: Optional[logging.Logger] = None, use_log10_scale: bool = False) -> None:
    """
    Plot 2D density plots of projection space for all bit pairs.
    
    Creates a grid of 2D density plots showing the distribution of samples in
    projection space. Each plot shows a pair of bits, with both overall density
    and cell-type-specific density overlays.
    
    Args:
        P (np.ndarray): Projection matrix of shape (n_samples, n_bits)
        y_labels (np.ndarray): Cell type labels of shape (n_samples,)
        plot_path (str): Path to save the plot (PDF format)
        sum_norm (bool): Whether to apply sum normalization. Default: False
        log (Optional[logging.Logger]): Logger instance. If None, creates default logger.
        use_log10_scale (bool): Whether to use log10 scale for axes. Default: False
            
    Note:
        - Creates plots for all bit pairs (arranged in 2 columns)
        - Left column: Overall density heatmap
        - Right column: Cell-type-specific colored overlays
        - Uses 2D histograms with 100 bins per axis
        - Applies percentile-based clipping to handle outliers
        - Saves as PDF with 300 DPI
    """
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

def plot_P_Type(P_type_data: torch.Tensor, valid_type_labels: List[str], plot_path: str, log: logging.Logger) -> None:
    """
    Generate a heatmap of average projections per cell type (P_type).
    
    Creates a heatmap showing the average bit projection values for each cell type.
    Useful for visualizing how different cell types are encoded in the bit space.
    
    Args:
        P_type_data (torch.Tensor): Average projections per cell type,
            shape (n_cell_types, n_bits)
        valid_type_labels (List[str]): Cell type names corresponding to rows
        plot_path (str): Path to save the plot (PDF format)
        log (logging.Logger): Logger instance for logging messages
            
    Note:
        - Sorts cell types alphabetically
        - Uses 1st and 99th percentiles for color scale
        - Automatically adjusts figure size based on number of cell types and bits
        - Saves as PDF with 300 DPI
    """
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
                                 columns=pd.Index([f"Bit_{b+1}" for b in range(n_bits)]))
        
        heatmap_fig = plt.figure(figsize=(fig_width, fig_height))
        ax_heatmap = heatmap_fig.add_subplot(111)
        if 'Bit' in plot_path.split('/')[-1]:
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

def plot_P_Type_correlation(P_type_data: torch.Tensor, valid_type_labels: List[str], plot_path: str, log: logging.Logger) -> None:
    """
    Generate a correlation heatmap between cell types based on their projections.
    
    Computes pairwise correlations between cell types' average bit projections
    and visualizes them as a heatmap. High correlation indicates similar encoding.
    
    Args:
        P_type_data (torch.Tensor): Average projections per cell type,
            shape (n_cell_types, n_bits)
        valid_type_labels (List[str]): Cell type names corresponding to rows
        plot_path (str): Path to save the plot (PDF format)
        log (logging.Logger): Logger instance for logging messages
            
    Note:
        - Computes correlation after z-score normalizing each cell type's projections
        - Correlation range: -1 to 1 (centered colormap)
        - Sorts cell types alphabetically
        - Saves as PDF with 300 DPI
    """
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

def sum_normalize_p_type(P_type_data: torch.Tensor) -> torch.Tensor:
    """
    Sum normalize P_type data to average sum across cell types.
    
    Normalizes each cell type's projections so that the sum across bits equals
    the average sum across all cell types. This makes projections comparable
    across cell types regardless of total signal intensity.
    
    Args:
        P_type_data (torch.Tensor): P_type matrix of shape (n_cell_types, n_bits)
        
    Returns:
        torch.Tensor: Sum-normalized P_type matrix of same shape
    """
    P_type_sum_norm = P_type_data.clone()
    avg_sum = P_type_sum_norm.sum(dim=1).mean()
    P_type_sum_norm = P_type_sum_norm * (avg_sum / P_type_sum_norm.sum(dim=1, keepdim=True).clamp(min=1e-8))
    return P_type_sum_norm

def bitwise_center_p_type(P_type_data: torch.Tensor) -> torch.Tensor:
    """
    Bitwise center P_type data by subtracting the median across cell types.
    
    For each bit, subtracts the median value across all cell types. This centers
    the data around zero for each bit independently.
    
    Args:
        P_type_data (torch.Tensor): P_type matrix of shape (n_cell_types, n_bits)
        
    Returns:
        torch.Tensor: Bitwise-centered P_type matrix of same shape
    """
    P_type_bit_center = P_type_data.clone()
    median_values = P_type_bit_center.median(dim=0, keepdim=True).values  # Extract values from named tuple
    P_type_bit_center = P_type_bit_center - median_values
    return P_type_bit_center

def bitwise_normalize_p_type(P_type_data: torch.Tensor) -> torch.Tensor:
    """
    Bitwise z-score normalize P_type data.
    
    For each bit, applies z-score normalization: (value - mean) / std across
    all cell types. This standardizes the scale of each bit independently.
    
    Args:
        P_type_data (torch.Tensor): P_type matrix of shape (n_cell_types, n_bits)
        
    Returns:
        torch.Tensor: Z-score normalized P_type matrix of same shape
        
    Note:
        - Uses clamp(min=1e-8) on std to avoid division by zero
    """
    P_type_bit_norm = P_type_data.clone()
    P_type_bit_norm = (P_type_bit_norm - P_type_bit_norm.mean(dim=0, keepdim=True)) / P_type_bit_norm.std(dim=0, keepdim=True).clamp(min=1e-8)
    return P_type_bit_norm

def plot_single_learning_curve(parameter: str, learning_curve: pd.DataFrame, output_dir: str, log: Optional[logging.Logger] = None) -> None:
    """
    Plot learning curve for a single parameter comparing training vs test.
    
    Creates a scatter plot showing how a metric evolves during training,
    with separate series for training and test sets.
    
    Args:
        parameter (str): Parameter name without '_train' or '_test' suffix
            (e.g., 'accuracy' for 'accuracy_train' and 'accuracy_test')
        learning_curve (pd.DataFrame): DataFrame containing learning statistics
            with columns like '{parameter}_train' and '{parameter}_test'
        output_dir (str): Directory to save the plot (PDF format)
        log (Optional[logging.Logger]): Logger instance. If None, creates default logger.
            
    Note:
        - Saves to '{output_dir}/learning_curve_{parameter}.pdf'
        - Uses 1st and 99th percentiles for y-axis limits
        - Skips plotting if insufficient valid data
        - Saves as PDF with 300 DPI
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

def plot_loss_contributions(learning_curve: pd.DataFrame, output_dir: str, log: Optional[logging.Logger] = None) -> None:
    """
    Plot relative contribution of each loss term to total loss across training.
    
    Creates a stacked area plot showing how the relative importance of each
    loss term changes during training. Useful for understanding which objectives
    dominate at different stages.
    
    Args:
        learning_curve (pd.DataFrame): DataFrame containing learning statistics
            with columns like '$$$ - <loss_name>_train' and 'total_loss_train'
        output_dir (str): Directory to save the plot (PDF format)
        log (Optional[logging.Logger]): Logger instance. If None, creates default logger.
            
    Note:
        - Computes relative contributions as loss_term / total_loss
        - Creates stacked area plot with all loss terms
        - Y-axis range: 0 to 1 (100% of total loss)
        - Saves to '{output_dir}/loss_contributions.pdf' with 300 DPI
        - Skips if no total loss column found
    """
    if log is None:
        log = logging.getLogger("LossContributionPlotter")
    loss_columns = [col for col in learning_curve.columns if 'train' in col.lower() and 'loss' in col.lower()]
    total_loss_cols = [col for col in learning_curve.columns if 'train' in col.lower() and 'loss' in col.lower() and 'total' in col.lower()]
    if not total_loss_cols:
        log.warning("No total loss column found. Skipping loss contributions plot.")
        return
    total_loss_col = total_loss_cols[0]
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
    plt.stackplot(x, np.vstack(contributions), labels=valid_loss_names)
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

def plot_comprehensive_performance(learning_curve: pd.DataFrame, output_dir: str, log: Optional[logging.Logger] = None) -> None:
    """
    Plot accuracy, separation, and dynamic range metrics on a single figure.
    
    Creates a multi-axis plot showing three key performance metrics simultaneously:
    accuracy (left y-axis), separation (right y-axis), and dynamic range (right y-axis).
    Each metric shows both training and test performance.
    
    Args:
        learning_curve (pd.DataFrame): DataFrame containing learning statistics
            with columns for accuracy, separation, and dynamic_range (with _train/_test suffixes)
        output_dir (str): Directory to save the plot (PDF format)
        log (Optional[logging.Logger]): Logger instance. If None, creates default logger.
            
    Note:
        - Uses different colors for each metric (orange, purple, cyan)
        - Accuracy y-axis: 0 to 1
        - Separation and dynamic range: 0 to 99th percentile
        - Dynamic range uses scientific notation
        - Saves to '{output_dir}/comprehensive_performance.pdf' with 300 DPI
        - Includes color-coded legend boxes
    """
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
    model = CIPHER(user_parameters_path=user_parameters_path)
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
    
    # # Check if trimming has been run and run it if not
    # trimmed_dir = os.path.join(model.I['output'], 'trimmed')
    # if not os.path.exists(trimmed_dir):
    #     print("Trimming directory not found. Running iterative trimming...")
    #     model.log.info("Trimming directory not found. Running iterative trimming...")
    #     try:
    #         # Default target is 12 bits, but can be adjusted based on model parameters
    #         target_n_bits = min(12, model.I['n_bit'] // 2)  # Use 12 or half the current bits, whichever is smaller
    #         model.iterative_trim_to_n_bits(target_n_bits=target_n_bits, max_iterations=10)
    #         print(f"Iterative trimming completed. Final n_bits: {model.I['n_bit']}")
    #     except Exception as e:
    #         print(f"Error during trimming: {e}")
    #         model.log.error(f"Error during trimming: {e}")
    # else:
    #     print("Trimming directory found. Skipping trimming.")
    #     model.log.info("Trimming directory found. Skipping trimming.")
