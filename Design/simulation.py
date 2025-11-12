#!/usr/bin/env python
# Standard library imports
# conda activate dredfish_3.9 ; python /scratchdata1/GeneralStorage/Zach/Designs/Sync/code/simulation.py "/scratchdata1/GeneralStorage/Zach/Designs/Sync/params_fig_Probe Number Tradeoff (36 Bits)_decoder_n_lyr_0_n_probes_50000_n_bit_36_replicate_1"
import os
import json
import logging
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
# Third-party imports
import numpy as np
import pandas as pd
import torch
import anndata
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
# Scientific computing
from sklearn.linear_model import LogisticRegression, LinearRegression, RANSACRegressor
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.ndimage import gaussian_filter

""" Parse Command Line Arguments """
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run simulation analysis')
    parser.add_argument('input_path', type=str, 
                       help='Path to the design results directory')
    parser.add_argument('--data_path', type=str, default='/u/home/z/zeh/rwollman/data',
                       help='Path to external data directory (default: /scratchdata1/ExternalData)')
    parser.add_argument('--ccf_x_min', type=float, default=0,
                       help='Minimum CCF x coordinate (default: 4.5)')
    parser.add_argument('--ccf_x_max', type=float, default=20,
                       help='Maximum CCF x coordinate (default: 9.5)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
input_path = args.input_path
data_path = args.data_path
ccf_x_min = args.ccf_x_min
ccf_x_max = args.ccf_x_max
design_name = input_path.split('/')[-1]

""" Setup Logging """
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_file = os.path.join(input_path, 'simulation.log')
logging.basicConfig(
    filename=log_file, filemode='a',
    format='%(message)s            |||| %(asctime)s %(name)s %(levelname)s',
    datefmt='%Y %B %d %H:%M:%S', level=logging.INFO, force=True)
logger = logging.getLogger("Simulation")

""" Setup Paths """
paths = {}
paths['WeightMat'] = f"{input_path}/results/E_constrained.csv"
paths['Design'] = f"{data_path}/Allen_Cortex_Hippocampus_SmartSeq_2023Sep07"
paths['Reference'] = f"{data_path}/Allen_WMB_2024Mar06"
paths['Simulation'] = f"{data_path}/Zhaung_WMB/"
for key,val in paths.items():
    if not os.path.exists(val):
        logger.error(f"Path {val} does not exist.")
        raise ValueError(f"Path {val} does not exist.")

paths['Save'] = f"{input_path}/Simulation"
if not os.path.exists(paths['Save']):
    os.mkdir(paths['Save'])
paths['Output'] = {}
for path in ['Design','Reference','Simulation','Results']:
    paths['Output'][path] = os.path.join(paths['Save'],path)
    if not os.path.exists(paths['Output'][path]):
        os.mkdir(paths['Output'][path])

""" Load Weights """
WeightMat = pd.read_csv(paths['WeightMat'], index_col=0)
logger.info(f"Weights loaded. Shape: {WeightMat.shape}")

""" Build Reference """
def process_single_file(file_name, current_dataset_path, current_cell_annotation_index, current_cell_extended, current_WeightMat, current_design_name, current_projected_path, current_dataset_batch, current_dataset_name_for_path):
    """
    Processes a single data file.
    Loads data, filters cells and genes, performs projection, saves individual output, and returns AnnData object.
    """
    try:
        individual_file_out_dir = os.path.join(current_projected_path, current_dataset_batch, current_dataset_name_for_path)
        os.makedirs(individual_file_out_dir, exist_ok=True)
        out_path = os.path.join(individual_file_out_dir, file_name)
        if os.path.exists(out_path):
            logger.info(f"File {file_name} already processed. Skipping.")
            try:
                out_data = anndata.read_h5ad(out_path)
                return out_data
            except Exception as e:
                logger.error(f"Error reading existing output file {out_path}: {e}")
        data_path = os.path.join(current_dataset_path, file_name)
        logger.info(f"Starting processing for: {file_name}")
        data = anndata.read_h5ad(data_path)
        # """ Remove Cells not in annotation """
        mask_cells = data.obs.index.isin(current_cell_annotation_index)
        logger.info(f"File {file_name}: {100*np.sum(mask_cells)/mask_cells.shape[0]:.2f}% of cells match annotation.")
        if np.sum(mask_cells) == 0:
            logger.warning(f"File {file_name}: No cells match annotation. Skipping.")
            return None
        data = data[mask_cells].copy()
        # """ Add useful info to obs """
        data.obs = current_cell_extended.loc[data.obs.index].copy() # Ensure it's a copy
        data.obs['library_size'] = data.X.sum(axis=1) # sum along axis 1 for rows
        # """ Match up genes with weights """
        # Create converter from gene symbol to gene ID for the current data object
        converter = {data.var.loc[gid]['gene_symbol']:gid for gid in data.var.index if 'gene_symbol' in data.var.columns}
        # Filter WeightMat (indexed by gene symbols) to those symbols present in the current data's gene symbols
        mask_genes_in_data = current_WeightMat.index.isin(data.var['gene_symbol'])
        filtered_WeightMat_local = current_WeightMat[mask_genes_in_data].copy()
        # Convert the gene symbols in filtered_WeightMat_local.index to GIDs using the converter
        # Only include symbols that are actually in the converter (i.e., in the current data's gene_symbols)
        valid_symbols_for_gid_conversion = [sym for sym in filtered_WeightMat_local.index if sym in converter]
        if not valid_symbols_for_gid_conversion:
            logger.warning(f"File {file_name}: No gene symbols from WeightMat found in this file's converter. Skipping.")
            return None
        filtered_WeightMat_local = filtered_WeightMat_local.loc[valid_symbols_for_gid_conversion]
        new_gid_index = [converter[symbol] for symbol in filtered_WeightMat_local.index]
        filtered_WeightMat_local.index = new_gid_index # Now filtered_WeightMat_local is indexed by GIDs
        logger.info(f"File {file_name}: {100*len(new_gid_index)/len(current_WeightMat.index):.2f}% of initial WeightMat genes selected after matching with data.")
        # """ filter data """
        # Filter data to include only genes (GIDs) that are in the index of (GID-indexed) filtered_WeightMat_local
        common_gids = [gid for gid in data.var.index if gid in filtered_WeightMat_local.index]
        if not common_gids:
            logger.warning(f"File {file_name}: No common GIDs found between data and WeightMat after conversion. Skipping.")
            return None
        filtered_data = data[:, common_gids].copy()
        # """ match order """
        # Order filtered_WeightMat_local (indexed by GIDs) according to the GID order in filtered_data.var.index
        ordered_filtered_WeightMat = filtered_WeightMat_local.loc[filtered_data.var.index].copy()
        # """ project """
        projected_X = filtered_data.X.dot(ordered_filtered_WeightMat.values) # Use .values for dot product with sparse matrix
        out_data = anndata.AnnData(
            X=projected_X.astype('float32'),
            var=pd.DataFrame(ordered_filtered_WeightMat.columns, index=np.array([f"readout{i}" for i in range(ordered_filtered_WeightMat.shape[1])]), columns=['bit']),
            obs=filtered_data.obs.copy() # Use obs from filtered_data
        )
        out_data.obs['probe_set'] = current_design_name
        # Ensure path for individual file output exists
        individual_file_out_dir = os.path.join(current_projected_path, current_dataset_batch, current_dataset_name_for_path)
        os.makedirs(individual_file_out_dir, exist_ok=True)
        out_path = os.path.join(individual_file_out_dir, file_name)
        logger.info(f"File {file_name}: Writing output to: {out_path}")
        out_data.write(out_path)
        logger.info(f"Finished processing {file_name}")
        
        # Explicit cleanup to prevent memory leaks
        del data, filtered_data, filtered_WeightMat_local, ordered_filtered_WeightMat, projected_X
        gc.collect()
        
        return out_data
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main processing block
if not os.path.exists(os.path.join(paths['Output']['Reference'], f"{design_name}.h5ad")):
    logger.info("Starting main processing: Combined reference file does not exist.")
    logger.info("Loading 10X data manifest and metadata...")
    manifest_path = os.path.join(paths['Reference'], 'manifest.json')
    try:
        manifest = json.load(open(manifest_path))
    except:
        logger.info(f"Manifest file does not exist. Downloading from AWS. {manifest_path}")
        url = 'https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/releases/%s/manifest.json' % '20230830'
        manifest = json.loads(requests.get(url).text)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
    
    metadata = manifest['file_listing']['WMB-10X']['metadata']
    rpath_cell_meta = metadata['cell_metadata']['files']['csv']['relative_path']
    file_cell_meta = os.path.join(paths['Reference'], rpath_cell_meta)
    cell = pd.read_csv(file_cell_meta, dtype={'cell_label':str})
    cell.set_index('cell_label', inplace=True)

    taxonomy_metadata = manifest['file_listing']['WMB-taxonomy']['metadata']
    rpath_cluster_details = taxonomy_metadata['cluster_to_cluster_annotation_membership_pivoted']['files']['csv']['relative_path']
    file_cluster_details = os.path.join(paths['Reference'], rpath_cluster_details)
    cluster_details = pd.read_csv(file_cluster_details, keep_default_na=False)
    cluster_details.set_index('cluster_alias', inplace=True)

    rpath_cluster_colors = taxonomy_metadata['cluster_to_cluster_annotation_membership_color']['files']['csv']['relative_path']
    file_cluster_colors = os.path.join(paths['Reference'], rpath_cluster_colors)
    cluster_colors = pd.read_csv(file_cluster_colors)
    cluster_colors.set_index('cluster_alias', inplace=True)

    rpath_roi_meta = metadata['region_of_interest_metadata']['files']['csv']['relative_path']
    file_roi_meta = os.path.join(paths['Reference'], rpath_roi_meta)
    roi = pd.read_csv(file_roi_meta)
    roi.set_index('acronym', inplace=True)
    roi.rename(columns={'order':'region_of_interest_order', 'color_hex_triplet':'region_of_interest_color'}, inplace=True)

    cell_extended = cell.join(cluster_details, on='cluster_alias')
    cell_extended = cell_extended.join(cluster_colors, on='cluster_alias')
    cell_extended = cell_extended.join(roi[['region_of_interest_order', 'region_of_interest_color']], on='region_of_interest_acronym')
    logger.info("Metadata loaded and joined.")

    data_keys = [i for i in manifest['directory_listing'].keys() if ('-10Xv' in i) and ('expression_matrices' in manifest['directory_listing'][i]['directories'].keys())]
    cell_annotation_index = cell_extended.index
    all_concatenated_data_batches = []
    for key in data_keys:
        logger.info(f"\nProcessing key: {key}")
        data_type, dataset_batch, dataset_name_for_path = manifest['directory_listing'][key]['directories']['expression_matrices']['relative_path'].split('/')
        os.makedirs(os.path.join(paths['Output']['Reference'], dataset_batch, dataset_name_for_path), exist_ok=True)
        dataset_path_for_key = os.path.join(paths['Reference'], data_type, dataset_batch, dataset_name_for_path)
        logger.info(f"Dataset path for key {key}: {dataset_path_for_key}")
        if not os.path.isdir(dataset_path_for_key):
            logger.warning(f"Dataset path {dataset_path_for_key} does not exist or is not a directory. Skipping.")
            continue
        files_in_dataset = os.listdir(dataset_path_for_key)
        tasks_for_executor = []
        for file_name_loop in files_in_dataset:
            if 'log' in file_name_loop.lower(): # Make check case-insensitive
                continue
            if not 'WMB' in file_name_loop: # Assuming 'WMB' check is still relevant
                continue
            if not file_name_loop.endswith('.h5ad'): # Process only .h5ad files
                continue
            tasks_for_executor.append(
                (file_name_loop, dataset_path_for_key, cell_annotation_index, cell_extended, WeightMat, design_name, paths['Output']['Reference'], dataset_batch, dataset_name_for_path)
            )
        if not tasks_for_executor:
            logger.warning(f"No valid .h5ad files found to process for key {key} in {dataset_path_for_key}")
            continue
        current_batch_processed_data = []
        
        # Log memory usage before starting batch
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage before batch {dataset_batch}: {memory_mb:.1f} MB")
        except:
            pass
            
        with ThreadPoolExecutor(max_workers=1) as executor:
            logger.info(f"Submitting {len(tasks_for_executor)} files for processing using 5 threads for batch {dataset_batch}...")
            future_to_task_args = {executor.submit(process_single_file, *task_args): task_args for task_args in tasks_for_executor}
            for i, future in tqdm(enumerate(as_completed(future_to_task_args)), total=len(future_to_task_args), desc=f"Processing files in batch {dataset_batch}"):
                task_args_done = future_to_task_args[future]
                file_name_done = task_args_done[0]
                logger.info(f"Thread finished for file: {file_name_done} ({i+1}/{len(tasks_for_executor)})")
                try:
                    result_out_data = future.result()
                    if result_out_data is not None:
                        current_batch_processed_data.append(result_out_data)
                except Exception as exc:
                    logger.error(f"File {file_name_done} generated an exception during future.result(): {exc}")
                    import traceback
                    traceback.print_exc()
            # Clean up futures and task args to free memory
            del future_to_task_args
            gc.collect()
        if current_batch_processed_data:
            logger.info(f"Concatenating {len(current_batch_processed_data)} processed files for batch {dataset_batch}...")
            try:
                concatenated_data_for_batch = anndata.concat(current_batch_processed_data, index_unique='observations')
                batch_out_dir = os.path.join(paths['Output']['Reference'], dataset_batch)
                os.makedirs(batch_out_dir, exist_ok=True)
                out_path_batch_combined = os.path.join(batch_out_dir, dataset_batch + '_combined.h5ad')
                logger.info(f"Writing combined batch data to: {out_path_batch_combined}")
                concatenated_data_for_batch.write(out_path_batch_combined)
                all_concatenated_data_batches.append(concatenated_data_for_batch)
                logger.info(f"Finished processing and combining for batch {dataset_batch}")
                # Clean up batch data to free memory
                del current_batch_processed_data, concatenated_data_for_batch
                gc.collect()
            except Exception as e_concat:
                logger.error(f"Error concatenating batch {dataset_batch}: {e_concat}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"No data to concatenate for batch {dataset_batch}.")
        logger.info(' ')
    if all_concatenated_data_batches:
        logger.info("\nConcatenating all processed batches...")
        try:
            final_all_concatenated_data = anndata.concat(all_concatenated_data_batches, index_unique='observations') # Or 'raise'
            final_output_path = os.path.join(paths['Output']['Reference'], f"{design_name}.h5ad")
            logger.info(f"Writing final concatenated data to: {final_output_path}")
            final_all_concatenated_data.write(final_output_path)
            logger.info("All processing complete. Final file written.")
            del final_all_concatenated_data # Free memory
        except Exception as e_final_concat:
            logger.error(f"Error during final concatenation: {e_final_concat}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("No data was processed and concatenated across all batches. Final file not written.")

    logger.info("Freeing up memory...")
    del all_concatenated_data_batches, cell_extended, cell, cluster_details, cluster_colors, roi, manifest
else:
    fname = os.path.join(paths['Output']['Reference'], f"{design_name}.h5ad")
    logger.info(f"Combined reference file {fname} already exists. Skipping processing.")

""" Build Simulation """
""" Chunk Data """
def chunk_data(fname):
    adata = anndata.read_h5ad(fname,backed='r')
    n_cells = adata.shape[0]
    n_chunks = int(np.ceil(n_cells / 50000))
    for i in trange(n_chunks):
        chunk_fname = fname.replace('.h5ad', f'_chunk{i}.h5ad')
        logger.info(f"Checking chunk {i}... {chunk_fname}")
        # Check if file exists and has reasonable size (> 1KB) before trying to read
        if os.path.exists(chunk_fname):
            try:
                # Check file size to ensure it's not empty/corrupted
                file_size = os.path.getsize(chunk_fname)
                if file_size < 1024:  # Less than 1KB, likely corrupted
                    logger.info(f"Chunk {i} exists but is too small ({file_size} bytes), recreating it.")
                    os.remove(chunk_fname)  # Remove corrupted file
                else:
                    # Try to read with timeout using ThreadPoolExecutor
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(anndata.read_h5ad, chunk_fname, backed='r')
                        try:
                            logger.info(f"Chunk {i} exists trying to read...")
                            temp_adata = future.result(timeout=60)  # 60 second timeout
                            logger.info(f"Chunk {i} already exists and is valid, skipping.")
                            continue
                        except TimeoutError:
                            logger.info(f"Chunk {i} exists but reading timed out, recreating it.")
                            os.remove(chunk_fname)
                        except Exception as e:
                            logger.info(f"Chunk {i} exists but is corrupted ({e}), recreating it.")
                            os.remove(chunk_fname)
            except Exception as e:
                logger.info(f"Error checking chunk {i} file: {e}, will recreate it.")
                if os.path.exists(chunk_fname):
                    try:
                        os.remove(chunk_fname)
                    except:
                        pass
        
        logger.info(f"Creating chunk {i}...")
        start_idx = i * 50000
        end_idx = min((i + 1) * 50000, n_cells)
        chunk_adata = anndata.AnnData(
            X=adata.X[start_idx:end_idx],
            obs=adata.obs.iloc[start_idx:end_idx],
            var=adata.var,
        )
        chunk_adata.obsm['X_CCF'] = adata.obsm['X_CCF'][start_idx:end_idx]
        logger.info(f"Writing chunk {i} to {chunk_fname}")
        chunk_adata.write(chunk_fname)
        del chunk_adata
for i in ['anterior','posterior']:
    chunk_data(os.path.join(paths['Simulation'], f"WB_imputation_animal1_coronal_{i}.h5ad"))

def process_chunk_item(chunk_h5ad_path, WeightMat_main, ccf_x_min_val, ccf_x_max_val):
    """Processes a single data chunk file."""
    try:
        adata = anndata.read_h5ad(chunk_h5ad_path)
        ccf_x_coords = np.array(adata.obsm['X_CCF'])[:, 0] / 1000.0
        mask = (adata.obs['high_quality_transfer']) & (ccf_x_coords > ccf_x_min_val) & (ccf_x_coords < ccf_x_max_val)
        if not np.any(mask):
            return None
        adata = adata[mask, :].copy()
        adata_genes = adata.var['gene_name']
        shared_genes = sorted(list(set(adata_genes).intersection(set(WeightMat_main.index))))
        if not shared_genes:
            logger.warning(f"No shared genes for {os.path.basename(chunk_h5ad_path)}")
            return None
        # Prepare var for gene ID mapping
        var_temp = adata.var.copy()
        var_temp['gene_id_col'] = var_temp.index # Store original var index (e.g. Ensembl)
        var_temp = var_temp.set_index('gene_name', drop=False) # Set gene_name as index
        var_temp_shared = var_temp.loc[shared_genes]
        adata_var_indices_ordered = var_temp_shared['gene_id_col'].values
        adata = adata[:, adata_var_indices_ordered].copy()
        WeightMat_chunk_specific = WeightMat_main.loc[shared_genes]
        E = torch.tensor(WeightMat_chunk_specific.values, dtype=torch.float32)
        if hasattr(adata.X, "toarray"):
            X_data = adata.X.toarray()
        else:
            X_data = adata.X
        X = torch.tensor(X_data, dtype=torch.float32)
        y_str = np.array(adata.obs['subclass_transfer'])
        ccf_coords = adata.obsm['X_CCF']
        P = X.mm(E)
        projected_obs_df = pd.DataFrame(index=adata.obs.index)
        projected_obs_df['subclass'] = y_str
        projected_obs_df['ccf_x'] = ccf_coords[:, 0] / 1000.0
        projected_obs_df['ccf_y'] = ccf_coords[:, 1] / 1000.0
        projected_obs_df['ccf_z'] = ccf_coords[:, 2] / 1000.0
        num_readouts = WeightMat_chunk_specific.shape[1]
        projected_var_df = pd.DataFrame(index=[f"readout{i}" for i in range(num_readouts)])
        projected_var_df['readout'] = [f"readout{i}" for i in range(num_readouts)]
        projected_var_df['hybe'] = [f"hybe{i}" for i in range(num_readouts)]
        projected_var_df['channel'] = [f"FarRed" for i in range(num_readouts)]
        adata = anndata.AnnData(X=P.numpy(), obs=projected_obs_df.copy(), var=projected_var_df.copy())
        # Clean up intermediate variables (but keep P for the return)
        # del adata, WeightMat_chunk_specific, E, X, X_data, ccf_coords, P, projected_obs_df, projected_var_df
        gc.collect()
        return adata
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(chunk_h5ad_path)}: {e}")
        return None

# Main processing logic
output_file = os.path.join(paths['Output']['Simulation'], f"{design_name}.h5ad")
if not os.path.exists(output_file):
    all_projected_adatas = []
    chunk_file_paths_to_process = [os.path.join(paths['Simulation'], i) for i in os.listdir(paths['Simulation']) if ('chunk' in i)]
    if not chunk_file_paths_to_process:
        logger.warning("No chunk files found to process.")
    else:
        logger.info(f"Found {len(chunk_file_paths_to_process)} chunk files to process")
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_chunk_item, cfp, WeightMat, ccf_x_min, ccf_x_max) for cfp in chunk_file_paths_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                result = future.result()
                if result is not None:
                    all_projected_adatas.append(result)
            # Clean up futures to free memory
            del futures
            gc.collect()
        if all_projected_adatas:
            final_projected_adata = anndata.concat(all_projected_adatas, axis=0, join='outer', merge='same')
            final_projected_adata.write_h5ad(output_file)
            logger.info(f"Processing complete. Final AnnData shape: {final_projected_adata.shape}")
            del final_projected_adata, all_projected_adatas, WeightMat
        else:
            logger.warning("No data was successfully processed from chunks.")
            raise ValueError("No data was successfully processed from chunks.")
else:
    logger.info(f"Output file {output_file} already exists. Skipping.")

""" Test Simulation """

def harmonize(M1, M2, pvmin=0, pvmax=100, num_quantiles=1000):
    """
    Harmonizes M2 to M1 using quantile alignment, handling NaNs and edge cases.

    Args:
        M1: Reference data matrix (numpy array).
        M2: Measured data matrix to be aligned (numpy array).
        pvmin: Lower percentile bound for outlier removal (default: 0).
        pvmax: Upper percentile bound for outlier removal (default: 100).
        num_quantiles: Number of quantiles to use for alignment (default: 1000).
                       Higher values give finer alignment but can be more sensitive
                       to noise.  A good starting point is 1000, but adjust as needed.

    Returns:
        M2_aligned: The aligned measured data matrix.
    """

    if M1.shape[1] != M2.shape[1]:
        logger.error("Matrices must have the same number of columns (dimensions).")
        raise ValueError("Matrices must have the same number of columns (dimensions).")

    M2_aligned = np.zeros_like(M2)

    for i in range(M1.shape[1]):
        # 1. Handle NaNs and Outliers (Robust Percentile Calculation)
        m1_col = M1[:, i]
        m2_col = M2[:, i]

        m1_valid = m1_col[~np.isnan(m1_col)]
        m2_valid = m2_col[~np.isnan(m2_col)]

        if len(m1_valid) < 2 or len(m2_valid) < 2:  # Not enough valid data
            # Fallback:  Shift M2's median to match M1's median
            M2_aligned[:, i] = m2_col - np.nanmedian(m2_col) + np.nanmedian(m1_col)
            continue

        vmin1, vmax1 = np.percentile(m1_valid, [pvmin, pvmax])
        vmin2, vmax2 = np.percentile(m2_valid, [pvmin, pvmax])

        m1_filtered = m1_valid[(m1_valid >= vmin1) & (m1_valid <= vmax1)]
        m2_filtered = m2_valid[(m2_valid >= vmin2) & (m2_valid <= vmax2)]

        if len(m1_filtered) < 2 or len(m2_filtered) < 2:  # Not enough data after filtering
            M2_aligned[:, i] = m2_col - np.nanmedian(m2_col) + np.nanmedian(m1_col)
            continue
        
        # 2. Quantile Normalization (using interpolation)
        # Create quantiles *within the filtered range*
        quantiles = np.linspace(0, 1, num_quantiles)
        m1_quantiles = np.quantile(m1_filtered, quantiles)
        m2_quantiles = np.quantile(m2_filtered, quantiles)

        # Handle duplicate quantiles (important for robustness!)
        m1_quantiles, unique_indices_m1 = np.unique(m1_quantiles, return_index=True)
        m2_quantiles = m2_quantiles[unique_indices_m1] #keep quantiles the same length

        m2_quantiles, unique_indices_m2 = np.unique(m2_quantiles, return_index = True)
        m1_quantiles = m1_quantiles[unique_indices_m2]


        # Create an interpolation function:  m2_values -> m1_values
        interp_func = interp1d(
            m2_quantiles,
            m1_quantiles,
            kind='linear',  # Or 'cubic' for smoother interpolation
            bounds_error=False,  # Allow extrapolation outside the quantile range
            fill_value=(m1_quantiles[0], m1_quantiles[-1])  # Extrapolate with min/max
        )

        # 3. Apply the Transformation to the *original* M2 data (including NaNs)
        M2_aligned[:, i] = interp_func(m2_col)

        # 4. Handle NaNs in the result (if any)
        nan_indices = np.isnan(M2_aligned[:, i])
        if np.any(nan_indices):  # If there are still NaNs, replace with M1's median
             M2_aligned[nan_indices, i] = np.nanmedian(m1_col)

    return M2_aligned

def normalize_fishdata_robust_regression(X):
    """
    Regression of the "sum" out of each basis using a robust estimate of the sum 
    so that "high" cells in a few bits won't skew that sum estimate too mucn

    Approach works in following steps: 
    1. Use n-1 other basis to predict each basis
    2. Do a RANSAC regression for each bit vs the other estimate to identify "outlier" cells, i.e. cells whos response
        is more extreme than expected based on the other basis sum. 
    3. Replace the values for the outlier cells/basis with the predictions from other basis to calcualte the sum. 
    4. Divide by sum and adjust for scale. 
    """

    # step 1: cross prediction matrix P
    P = np.zeros_like(X)
    
    for target_col in range(X.shape[1]):
        # Prepare the features (X) and target (y) for the current target column
        F = np.delete(X, target_col, axis=1)
        y = X[:, target_col]
        linear_model = LinearRegression().fit(F,y)
        
        # Predict the target column using the trained model
        P[:, target_col] = linear_model.predict(F)

    # Step 2: fit RANSAC regression to find outliers
    inliners = np.zeros_like(X)
    common = P.mean(axis=1).reshape(-1,1)
    for i in range(X.shape[1]):
        f = X[:,i]
        # Step 1: Initial Linear Regression to estimate residuals
        init_reg = LinearRegression().fit(common,f)
        std_residuals = np.std(f - init_reg.predict(common))
        ransac = RANSACRegressor(LinearRegression(), 
                                residual_threshold = std_residuals, 
                                random_state=42)
        ransac.fit(common, f)
        inliners[:,i] = ransac.inlier_mask_

    # Step 3: replace outliers with cross-predictions
    Xrobust = X*inliners + P*(1-inliners)

    # Step 4: Normalize by dividing by sum and rescaling
    Nrm = X/Xrobust.mean(axis=1).reshape(-1,1)*Xrobust.mean()

    return Nrm

class KDESpatialPriors(object):
    def __init__(self,
    ref='/scratchdata2/MouseBrainAtlases/MouseBrainAtlases_V0/Allen/',
    ref_levels=['class', 'subclass'],neuron=None,kernel = (0.25,0.1,0.1),
    border=1,binsize=0.1,bins=None,gates=None,types=None,symetric=False):
        self.out_path = f"/u/home/z/zeh/rwollman/data/KDE_kernel_{kernel[0]}_{kernel[1]}_{kernel[2]}_border_{border}_binsize_{binsize}_level_{ref_levels[-1]}_neuron_{neuron}_symetric_{symetric}.pkl"
        logger.info(f"KDESpatialPriors: {self.out_path}")
        self.symetric = symetric
        if os.path.exists(self.out_path):
            temp = pickle.load(open(self.out_path,'rb'))
            self.typedata = temp.typedata
            self.types = temp.types
            self.ref_levels = temp.ref_levels
            self.neuron = temp.neuron
            self.kernel = temp.kernel
            self.bins = temp.bins
            self.gates = temp.gates
            self.border = temp.border
            self.binsize = temp.binsize
            self.ref = temp.ref
            self.converters = temp.converters
        else:
            if isinstance(ref,str):
                self.ref = anndata.read_h5ad(f"{ref}/Layer/cell_layer.h5ad")
                # TissueGraph.TissueMultiGraph(basepath = ref, input_df = None, redo = False).Layers[0].adata
            else:
                self.ref = ref

            self.ref_levels = ref_levels
            self.neuron = neuron
            self.kernel = kernel
            self.bins = bins
            self.gates = gates
            self.border = border
            self.binsize = binsize
            self.types = types
            self.typedata = None

    def train(self,dim_labels=['x_ccf','y_ccf','z_ccf']):
        """ check if types in self """
        if isinstance(self.typedata,type(None)):
            binsize = self.binsize
            border = self.border

            if self.symetric:
                dim = [i for i in dim_labels if 'z' in i][0]
                center = 5.71 #ccf_z'
                adata = self.ref.copy()
                # labels = dim_labels
                # labels.extend(self.ref_levels)
                # adata.obs = adata.obs[labels].copy()
                flipped_adata = adata.copy()
                flipped_adata.obs[dim] = center + (-1*(adata.obs[dim] - center))
                adata = anndata.concat([adata,flipped_adata])
                self.ref = adata


            XYZ = np.array(self.ref.obs[dim_labels])

            if isinstance(self.gates,type(None)):
                self.gates = []
                self.bins = []
                for dim in range(3):
                    vmin  = binsize*int((np.min(XYZ[:,dim])-border)/binsize)
                    vmax = binsize*int((np.max(XYZ[:,dim])+border)/binsize)
                    g = np.linspace(vmin,vmax,int((vmax-vmin)/binsize)+1)
                    self.gates.append(g)
                    self.bins.append(g[:-1]+binsize/2)
            bins = self.bins
            gates = self.gates

            labels = np.array(self.ref.obs[self.ref_levels[-1]])
            types = np.unique(labels)
            if isinstance(self.types,type(None)):
                if isinstance(self.neuron,bool):
                    if self.neuron:
                        print('Using Only Neurons')
                        types = np.array([i for i in types if not 'NN' in i])
                    else:
                        print('Using Only Non Neurons')
                        types = np.array([i for i in types if 'NN' in i])
                    # print(f" Using these Types only {types}")
            else:
                types = self.types
            self.types = types
            typedata = np.zeros([bins[0].shape[0],bins[1].shape[0],bins[2].shape[0],types.shape[0]],dtype=np.float16)
            for i in trange(types.shape[0],desc='Calculating Spatial KDE'):
                label = types[i]
                m = labels==label
                if np.sum(m)==0:
                    continue
                hist, edges = np.histogramdd(XYZ[m,:], bins=gates)
                # stk = gaussian_filter(hist,(0.5/binsize,0.25/binsize,0.25/binsize))
                # stk = gaussian_filter(hist,(0.25/binsize,0.1/binsize,0.1/binsize))
                stk = gaussian_filter(hist,(i/binsize for i in self.kernel))
                typedata[:,:,:,i] = stk
            density = np.sum(typedata,axis=-1,keepdims=True)
            density[density==0] = 1
            typedata = typedata/density
            self.typedata = typedata

            self.converters = {}
            for level in self.ref_levels:
                if level==self.ref_levels[-1]:
                    continue
                self.converters[level] = dict(zip(self.ref.obs[self.ref_levels[-1]],self.ref.obs[level]))
            pickle.dump(self,open(self.out_path,'wb'))

    def convert_priors(self,priors,level):
        converter = self.converters[level]
        types = np.unique([item for key,item in converter.items()])
        updated_priors = np.zeros([priors.shape[0],types.shape[0]])
        for i,label in enumerate(types):
            m = np.array([converter[key]==label for key in self.types])
            updated_priors[:,i] = np.sum(priors[:,m],axis=1)
        return updated_priors,types
        
    def classify(self,measured,level='subclass',dim_labels=['ccf_x','ccf_y','ccf_z']):
        XYZ = np.array(measured.obs[dim_labels])
        XYZ_coordinates = XYZ.copy()
        for dim in range(3):
            XYZ_coordinates[:,dim] = (XYZ_coordinates[:,dim]-self.bins[dim][0])/(self.bins[dim][1]-self.bins[dim][0])
        XYZ_coordinates = XYZ_coordinates.astype(int)

        priors = self.typedata[XYZ_coordinates[:,0],XYZ_coordinates[:,1],XYZ_coordinates[:,2],:]
        types = self.types

        if level!=self.ref_levels[-1]:
            priors,types = self.convert_priors(priors,level)
        return priors,types

class SingleCellAlignmentLeveragingExpectations(): 
    def __init__(self,measured,complete_reference='allen_wmb_tree',ref_level='subclass',verbose=True,visualize=False):
        self.verbose = verbose
        self.complete_reference = complete_reference
        self.measured = measured.copy()
        self.ref_level = ref_level
        self.visualize = visualize
        self.model = LogisticRegression(max_iter=1000) 
        self.likelihood_only = False
        self.prior_only = False
        self.n_models = 5

    def update_user(self,message):
        logger.info(f"SCALE: {message}")

    def unsupervised_clustering(self):
        self.update_user("Unsupervised Clustering is not up to date")
        raise ValueError("Unsupervised Clustering is not up to date")
        # has a tendency to oversplit large cell types
        self.update_user("Running unsupervised clustering")
        X = self.measured.layers['classification_space'].copy()
        G,knn = tmgu.build_knn_graph(X,metric='correlation')
        TypeVec = G.community_leiden(resolution=10,objective_function='modularity').membership
        unsupervised_labels = np.array(TypeVec).astype(str)
        logger.info(f"Number of unique unsupervised labels: {np.unique(unsupervised_labels).shape[0]}")
        unsupervised_labels,colors = merge_labels_correlation(X,unsupervised_labels)
        cts = np.array(np.unique(unsupervised_labels))
        # Generate colors evenly spaced across the jet colormap
        cmap = plt.get_cmap('gist_ncar')
        colors = [cmap(i / cts.shape[0]) for i in range(cts.shape[0])]
        np.random.shuffle(colors)
        # colors = np.random.choice(np.array(list(mcolors.XKCD_COLORS.keys())),cts.shape[0],replace=False)
        pallette = dict(zip(cts, colors))
        unsupervised_colors = np.array(pd.Series(unsupervised_labels).map(pallette))
        
        self.measured.obs['leiden'] = unsupervised_labels
        self.measured.obs['leiden_color'] = unsupervised_colors
        self.visualize_measured('leiden_color','Unsupervised Clustering')

    def calculate_spatial_priors(self):
        self.update_user("Loading Spatial Priors Class")
        kdesp = KDESpatialPriors(ref_levels=[self.ref_level],neuron=None,kernel=(0.25,0.1,0.1))
        kdesp.train()
        self.update_user("Calculating Spatial priors")
        priors = {}
        priors,types = kdesp.classify(self.measured, level=self.ref_level,dim_labels=['ccf_x','ccf_y','ccf_z'])
        priors[np.sum(priors,axis=1)==0,:] = 1 # if all zeros make it uniform
        priors = {'columns':types,'indexes':np.array(self.measured.obs.index),'matrix':priors.astype(np.float32)}
        self.priors = priors

    def load_reference(self):
        # if isinstance(self.complete_reference,str):
        #     self.update_user("Loading Reference Data")
        #     self.complete_reference = anndata.read(pathu.get_path(self.complete_reference, check=True))
        
        shared_var = list(self.complete_reference.var.index.intersection(self.measured.var.index))
        self.reference = self.complete_reference[:,np.isin(self.complete_reference.var.index,shared_var)].copy()
        self.measured = self.measured[:,np.isin(self.measured.var.index,shared_var)]

        # Filter and reindex the reference and measured data to ensure the order matches
        self.reference = self.reference[:, shared_var].copy()
        self.measured = self.measured[:, shared_var].copy()

        # add neuron annotation to reference
        self.unique_labels = np.array(self.reference.obs[self.ref_level].unique())
        self.neuron_labels = np.array([i for i in self.unique_labels if not 'NN' in i])
        self.non_neuron_labels = np.array([i for i in self.unique_labels if 'NN' in i])
        converter = {True:'Neuron',False:'Non_Neuron'}
        self.neuron_converter = {ct:converter[ct in self.neuron_labels] for ct in self.unique_labels}
        self.neuron_color_converter = {'Neuron':'k','Non_Neuron':'r'}
        self.reference.obs['neuron'] = self.reference.obs[self.ref_level].map(self.neuron_converter)
        self.reference.obs['neuron_color'] = self.reference.obs['neuron'].map(self.neuron_color_converter)
        self.ref_level_color_converter = dict(zip(self.reference.obs[self.ref_level],self.reference.obs[f"{self.ref_level}_color"]))

        # Normalize Measured Data
        self.update_user("Normalizing Measured Data")
        self.measured.X = self.measured.layers['raw'].copy()
        self.measured.layers['normalized'] = normalize_fishdata_robust_regression(self.measured.X.copy()) #Fix Staining Efficiency
        # self.measured.layers['normalized'] = basicu.image_coordinate_correction(self.measured.layers['normalized'].copy(),np.array(self.measured.obs[["image_x","image_y"]])) # Fix Flat field

        self.reference_dict = {}
        self.reference_info_dict = {}
        for iteration in range(self.n_models):
            self.reference_info_dict[iteration] = {}
            self.update_user(f"Resampling Reference Data {iteration}")
            # Balance Reference to Section Area
            Nbases = self.measured.shape[1]
            idxes = []
            # total_cells = np.min([self.measured.shape[0],500000])
            total_cells = 500000
            weights = np.mean(self.priors['matrix'],axis=0)
            weights = weights/weights.sum()
            for i,label in enumerate(self.priors['columns']):
                n_cells = int(total_cells*weights[i])
                if n_cells>10:
                    m = self.reference.obs[self.ref_level]==label
                    temp = np.array(self.reference.obs[m].index)
                    if temp.shape[0]>0:
                        if np.sum(m)>n_cells:
                            idxes.extend(list(np.random.choice(temp,n_cells,replace=False)),)
                        else:
                            idxes.extend(list(np.random.choice(temp,n_cells)))
                # else:
                #     self.update_user(f"Removing {label} from Reference too few cells {n_cells}")
            reference = self.reference[idxes,:].copy()
            # Perform Normalization
            reference.layers['raw'] = reference.X.copy()
            self.update_user(f"Normalizing Reference Data cell wise {iteration}")
            reference.layers['normalized'] = normalize_fishdata_robust_regression(reference.X.copy())
            reference.layers['classification_space'] = reference.layers['normalized'].copy()
            self.update_user(f"Normalizing Reference Data bit wise {iteration}")
            medians = np.zeros(reference.layers['classification_space'].shape[1])
            stds = np.zeros(reference.layers['classification_space'].shape[1])
            for i in range(reference.layers['classification_space'].shape[1]):
                vmin,vmax = np.percentile(reference.layers['classification_space'][:,i],[5,95])
                ref_mask = (reference.layers['classification_space'][:,i]>vmin)&(reference.layers['classification_space'][:,i]<vmax)
                medians[i] = np.median(reference.layers['classification_space'][ref_mask,i])
                stds[i] = np.std(reference.layers['classification_space'][ref_mask,i])
                reference.layers['classification_space'][:,i] = (reference.layers['classification_space'][:,i]-medians[i])/stds[i]
            self.reference_info_dict[iteration]['medians'] = medians
            self.reference_info_dict[iteration]['stds'] = stds

            self.reference_dict[iteration] = reference.copy()

        self.reference = reference.copy()
        self.ref_medians = medians
        self.ref_stds = stds

        self.update_user("Building Reference tree")
        # self.feature_tree_dict = {}
        # self.feature_tree_dict['labels'] = np.array(self.reference.obs.index.copy())
        # self.feature_tree_dict['tree'] = NNDescent(self.reference.layers['classification_space'], metric='euclidean', n_neighbors=15,n_trees=10,verbose=self.verbose,random_state=42)

    def merge_predictions(self):
        consensus_labels = []
        freq = {i+1:0 for i in range(self.n_models)}
        for cell in range(len(self.model_predictions[0])):
            labels = [self.model_predictions[i][cell] for i in range(self.n_models)]
            if len(set(labels)) == 1:
                freq[len(labels)]+=1
                consensus_labels.append(labels[0])
            elif len(set(labels)) == len(labels):
                freq[1]+=1
                consensus_labels.append(np.random.choice(labels))
            else:
                l = pd.DataFrame(labels).value_counts()
                consensus_labels.append(l.index[0][0])
                freq[l[0]]+=1
        return consensus_labels

    def supervised_neuron_annotation(self):
        self.model_predictions = {}
        for iteration in range(self.n_models):
            self.reference = self.reference_dict[iteration].copy()
            self.measured.layers['classification_space'] = harmonize(self.reference.layers['classification_space'].copy(),self.measured.layers['normalized'].copy())
            self.model.fit(self.reference.layers['classification_space'],self.reference.obs['neuron'])
            self.model_predictions[iteration] = self.model.predict(self.measured.layers['classification_space'])

        self.measured.obs['neuron'] = self.merge_predictions()
        self.measured.obs['neuron_color'] = self.measured.obs['neuron'].map(self.neuron_color_converter)
        self.visualize_layers('classification_space',measured_color='leiden_color',reference_color=f"{self.ref_level}_color",reference_layer = 'classification_space')

        # Update Priors
        self.update_user("Updating Priors")
        prior_matrix = self.priors['matrix'].copy()
        neuron_idxs = [idx for idx,ct in enumerate(self.priors['columns']) if not 'NN' in ct]
        m = self.measured.obs['neuron']=='Non_Neuron'
        nn_priors = self.priors['matrix'][m,:].copy()
        nn_priors[:,neuron_idxs] = 0
        prior_matrix[m,:] = nn_priors

        non_neuron_idxs = [idx for idx,ct in enumerate(self.priors['columns']) if 'NN' in ct]
        m = self.measured.obs['neuron']=='Neuron'
        n_priors = self.priors['matrix'][m,:].copy()
        n_priors[:,non_neuron_idxs] = 0
        prior_matrix[m,:] = n_priors

        self.priors['matrix'] = prior_matrix.copy()
        self.visualize_measured('neuron_color','Neuron Annotation')
        self.visualize_layers('classification_space',measured_color='neuron_color',reference_color='neuron_color',reference_layer = 'classification_space')


    def unsupervised_neuron_annotation(self):
        self.update_user("Unsupervised Neuron Annotation is not up to date")
        raise ValueError("Unsupervised Neuron Annotation is not up to date")
        # Initial harmonization
        self.update_user("Performing Initial harmonization")
        # self.measured.layers['classification_space'] = basicu.zscore_matching(self.reference.layers['classification_space'].copy(),self.measured.layers['normalized'].copy())
        self.measured.layers['classification_space'] = harmonize(self.reference.layers['classification_space'].copy(),self.measured.layers['normalized'].copy())
        self.visualize_layers('classification_space',measured_color='leiden_color',reference_color=f"{self.ref_level}_color",reference_layer = 'classification_space')

        self.update_user('Generateing Unsupervised Vectors')
        cts = self.measured.obs['leiden'].unique()
        X = np.zeros([len(cts),self.measured.shape[1]])
        for i,ct in enumerate(cts):
            m = self.measured.obs['leiden']==ct
            X[i,:] = np.median(self.measured.layers['classification_space'][m,:],axis=0)
        self.update_user("Querying Reference tree")
        neighbors,distances = self.feature_tree_dict['tree'].query(X,k=25)
        numerical_converter = {'Neuron':0,'Non_Neuron':1}
        referse_numerical_converter = {item:key for key,item in numerical_converter.items()}
        reference_labels = np.array(self.reference.obs.loc[self.feature_tree_dict['labels'],'neuron'].map(numerical_converter).values)
        prediction = np.array(pd.Series(1*(np.mean(reference_labels[neighbors],axis=1)>0.5)).map(referse_numerical_converter))
        converter = dict(zip(cts,prediction))

        self.measured.obs['neuron'] = self.measured.obs['leiden'].map(converter)
        self.measured.obs['neuron_color'] = self.measured.obs['neuron'].map(self.neuron_color_converter)

        # self.update_user("Querying Reference tree")
        # neighbors,distances = self.feature_tree_dict['tree'].query(self.measured.layers['classification_space'],k=25)
        # self.update_user("Calculating Neuron Annotation")
        # numerical_converter = {'Neuron':0,'Non_Neuron':1}
        # referse_numerical_converter = {item:key for key,item in numerical_converter.items()}
        # reference_labels = np.array(self.reference.obs.loc[self.feature_tree_dict['labels'],'neuron'].map(numerical_converter).values)
        # prediction = np.array(pd.Series(1*(np.mean(reference_labels[neighbors],axis=1)>0.5)).map(referse_numerical_converter))
        # self.measured.obs['neuron'] = prediction
        # self.measured.obs['neuron_color'] = self.measured.obs['neuron'].map(self.neuron_color_converter)

        # Update Priors
        self.update_user("Updating Priors")
        prior_matrix = self.priors['matrix'].copy()
        neuron_idxs = [idx for idx,ct in enumerate(self.priors['columns']) if not 'NN' in ct]
        m = self.measured.obs['neuron']=='Non_Neuron'
        nn_priors = self.priors['matrix'][m,:].copy()
        nn_priors[:,neuron_idxs] = 0
        prior_matrix[m,:] = nn_priors

        non_neuron_idxs = [idx for idx,ct in enumerate(self.priors['columns']) if 'NN' in ct]
        m = self.measured.obs['neuron']=='Neuron'
        n_priors = self.priors['matrix'][m,:].copy()
        n_priors[:,non_neuron_idxs] = 0
        prior_matrix[m,:] = n_priors

        self.priors['matrix'] = prior_matrix.copy()
        self.visualize_measured('neuron_color','Neuron Annotation')
        self.visualize_layers('classification_space',measured_color='neuron_color',reference_color='neuron_color',reference_layer = 'classification_space')


    def supervised_harmonization(self):
        backup_measured = self.measured.copy()
        for iteration in range(self.n_models):
            self.update_user(f"Performing Supervised Harmonization {iteration}")
            self.measured = backup_measured.copy()

            self.reference = self.reference_dict[iteration].copy()

            for ct in np.unique(self.measured.obs['neuron']):
                m = self.measured.obs['neuron']==ct
                ref_m = self.reference.obs['neuron']==ct
                if np.sum(m)>0:
                    self.measured.layers['classification_space'][m,:] = harmonize(self.reference.layers['classification_space'][ref_m,:],self.measured.layers['normalized'][m,:])

            self.update_user("Creating Dendrogram")
            X = np.array(self.reference.layers['classification_space'])
            y = np.array(self.reference.obs[self.ref_level])
            unique_labels = np.unique(y)
            averaged_data = {label: np.median(X[y == label],axis=0) for label in unique_labels}
            df = pd.DataFrame(averaged_data)
            correlations = df.corr()
            condensed_distance_matrix = squareform(np.abs(correlations-1), checks=False)
            self.Z = linkage(condensed_distance_matrix, method='ward')
            self.averagWeightMat = df.copy()
            clusters = fcluster(self.Z, t=self.Z[0,2], criterion='distance')
            column_clusters = {col: cluster for col, cluster in zip(self.averagWeightMat.columns, clusters)}

            dend = pd.DataFrame(index=self.averagWeightMat.columns,columns=range(self.Z.shape[0]))
            stop=False
            for n in range(self.Z.shape[0]):
                clusters = np.array(fcluster(self.Z, t=self.Z[-(n+1),2], criterion='distance'))
                if n==0:
                    dend[n] = clusters
                    column_clusters = {col: cluster for col, cluster in zip(self.averagWeightMat.columns, clusters)}
                    previous_labels = pd.Series(y).map(column_clusters)
                    continue
                clusters = clusters+dend[n-1].max()
                column_clusters = {col: cluster for col, cluster in zip(self.averagWeightMat.columns, clusters)}

                updated_labels = pd.Series(y).map(column_clusters)
                updated_label_counts = updated_labels.value_counts()
                previous_label_counts = previous_labels.value_counts()
                """ check for clusters with the same number of cells"""
                for idx in updated_label_counts.index:
                    count = updated_label_counts[idx]
                    for idx2 in previous_label_counts[previous_label_counts==count].index:
                        """ check if they are for the same base_types"""
                        if np.mean(np.array(dend[n-1]==idx2)==np.array(clusters==idx))==1:
                            clusters[clusters==idx] = idx2
                dend[n] = clusters
                column_clusters = {col: cluster for col, cluster in zip(self.averagWeightMat.columns, clusters)}
                previous_labels = pd.Series(y).map(column_clusters)
            """ Rename to have no gaps in names"""
            mapper = dict(zip(sorted(np.unique(dend)),range(len(np.unique(dend)))))
            for i in dend.columns:
                dend[i] = dend[i].map(mapper)
            self.dend = dend

            """ Remake Priors to use dend clusters """
            self.update_user("Remaking Priors")
            types = np.unique(np.array(self.dend).ravel())
            priors = np.zeros((self.measured.shape[0],types.shape[0])).astype(np.float32)
            updated_priors = {'columns':types,'indexes':np.array(self.measured.obs.index),'matrix':priors.astype(np.float32)}
            for i,cluster in enumerate(types):
                included_labels = np.array(self.dend.index)[(self.dend==cluster).max(axis=1)==1]
                updated_priors['matrix'][:,i] = np.sum(self.priors['matrix'][:,np.isin(self.priors['columns'],included_labels)],axis=1)

            self.dend_priors = updated_priors

            self.reference_features = X
            self.reference_labels_ref = y
            self.reference_cell_names = np.array([i.split('raise')[0] for i in self.reference.obs.index])

            self.measured_features = self.measured.layers['normalized'].copy()
            self.measured_labels = np.zeros(self.measured_features.shape[0])

            # self.measured_features = basicu.zscore_matching(self.reference_features,self.measured_features)
            # self.measured_features = harmonize(self.reference_features,self.measured_features)
            self.measured.layers['zscored'] = self.measured_features.copy()
            self.update_user("Harmonizing")
            # set up empty likelihoods and posteriors to match priors
            self.likelihoods = {'columns':self.dend_priors['columns'],'indexes':np.array(self.measured.obs.index),'matrix':np.zeros_like(self.dend_priors['matrix'])}
            self.posteriors = {'columns':self.dend_priors['columns'],'indexes':np.array(self.measured.obs.index),'matrix':np.zeros_like(self.dend_priors['matrix'])}
            self.measured_features = self.measured.layers['classification_space'].copy()
            completed_clusters = []

            clusters = np.array(sorted(np.unique(np.array(self.dend).ravel())))
            if self.verbose:
                iterable = tqdm(clusters,desc='Harmonizing')
            else:
                iterable = clusters
            for cluster in iterable:
                if cluster in self.dend[self.dend.columns[-1]].unique():
                    """ Reached the end of this branch"""
                    continue
                mask = self.measured_labels==cluster
                self.likelihoods['matrix'][mask,:] = 0
                if np.sum(mask)==0:
                    continue
                n = np.max(self.dend.columns[(self.dend==cluster).max(0)])
                if n==self.dend.columns[-1]:
                    """ Reached the end of this branch"""
                    continue
                mapper = dict(self.dend[n+1])
                next_clusters = self.dend[n+1][self.dend[n]==cluster].unique()
                self.reference_labels = np.array(pd.Series(self.reference_labels_ref).map(mapper))
                ref_m = np.isin(self.reference_labels, next_clusters) 
                self.model.fit(self.reference_features[ref_m,:],self.reference_labels[ref_m])

                likelihoods = self.model.predict_proba(self.measured_features[mask,:]).astype(np.float32)
                for idx,ct in enumerate(self.model.classes_):
                    jidx = np.where(self.likelihoods['columns']==ct)[0][0]
                    self.likelihoods['matrix'][mask,jidx] = likelihoods[:,idx]
                likelihoods = self.likelihoods['matrix'][mask,:].copy()
                likelihoods[:,np.isin(self.dend_priors['columns'],next_clusters)==False] = 0
                priors = self.dend_priors['matrix'][mask,:].copy()
                priors[:,np.isin(self.dend_priors['columns'],next_clusters)==False] = 0

                if self.likelihood_only:
                    posteriors = likelihoods
                elif self.prior_only:
                    posteriors = priors
                else:
                    posteriors = likelihoods*priors
                    posteriors[posteriors.max(1)==0,:] = likelihoods[posteriors.max(1)==0,:].copy()

                labels = self.dend_priors['columns'][np.argmax(posteriors,axis=1)]
                labels[posteriors.max(1)==0] = -1
                self.measured_labels[mask] = labels
                for cluster in np.unique(self.measured_labels[mask]):
                    m = self.measured_labels==cluster
                    ref_m = self.reference_labels==cluster
                    if np.sum(m)>0:
                        if cluster ==-1:
                            self.update_user(f"{np.sum(m)} cells 0 posterior")
                            continue
                        # self.measured_features[m,:] = basicu.zscore_matching(self.reference_features[ref_m,:],self.measured_features[m,:])
                        self.measured_features[m,:] = harmonize(self.reference_features[ref_m,:],self.measured_features[m,:])
                gc.collect()
            self.update_user(f"Mapping labels to {self.ref_level}")
            mapper = dict(self.dend[self.dend.columns[-1]])
            reverse_mapper = {v:k for k,v in mapper.items()}
            self.model_predictions[iteration] = np.array(pd.Series(self.measured_labels).map(reverse_mapper))

        self.update_user(f"Merging labels to {self.ref_level}")
        self.measured_labels = self.merge_predictions()
        self.measured.obs[self.ref_level] = self.measured_labels
        self.measured.obs[f"{self.ref_level}_color"] = self.measured.obs[self.ref_level].map(self.ref_level_color_converter)
        self.measured_features = self.measured.layers['normalized'].copy()
        for ct in np.unique(self.measured_labels):
            m = self.measured.obs[self.ref_level]==ct
            ref_m = self.reference.obs[self.ref_level]==ct
            if np.sum(m)>0:
                # self.measured_features[m,:] = basicu.zscore_matching(self.reference.layers['normalized'][ref_m,:],self.measured_features[m,:])
                self.measured_features[m,:] = harmonize(self.reference.layers['classification_space'][ref_m,:],self.measured_features[m,:])

        self.visualize_measured(f"{self.ref_level}_color",'Supervised Annotation')
        self.measured.layers['harmonized_classification_space'] = self.measured_features.copy()
        self.measured.layers['harmonized'] = (self.measured_features.copy()*self.ref_stds)+self.ref_medians
        
        # self.visualize_layers('harmonized',measured_color='leiden_color',reference_color=f"{self.ref_level}_color",reference_layer = 'normalized')
        self.visualize_layers('harmonized',measured_color=f"{self.ref_level}_color",reference_color=f"{self.ref_level}_color",reference_layer = 'normalized')

    def determine_neighbors(self):
        self.update_user("Determining Neighbors")
        
        self.update_user("Querying Reference tree")
        neighbors,distances = self.feature_tree_dict['tree'].query(self.measured.layers['harmonized_classification_space'],k=15)
        X = np.zeros_like(self.measured.layers['harmonized_classification_space'])
        for i in range(neighbors.shape[1]):
            self.measured.obs[f"reference_neighbor_{i}"] = np.array(self.feature_tree_dict['labels'][neighbors[:,i]])
            X = X+np.array(self.reference.layers['raw'][neighbors[:,i],:])
        X = X/neighbors.shape[1]
        self.measured.layers['imputed'] = X.copy()

        self.visualize_layers('imputed',measured_color=f"{self.ref_level}_color",reference_color=f"{self.ref_level}_color",reference_layer = 'raw')

    def run(self):
        self.calculate_spatial_priors()
        self.load_reference()
        #self.unsupervised_clustering()
        self.supervised_neuron_annotation()
        self.supervised_harmonization()
        self.determine_neighbors()
        return self.measured

    def visualize_measured(self,color,title):
        if self.visualize:
            adata = self.measured.copy()
            x = adata.obs['ccf_z']
            y = adata.obs['ccf_y']
            c = np.array(adata.obs[color])
            fig,ax  = plt.subplots(1,1,figsize=[5,5])
            ax.scatter(x,y,s=0.1,c=c,marker='.')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.axis('off')
            plt.show()

    def visualize_layers(self,layer,measured_color='leiden_color',reference_color='',reference_layer = ''):
        if self.visualize:
            adata = self.measured.copy()
            ref_adata = self.reference.copy()
            if reference_color == '':
                if measured_color in ref_adata.obs.columns:
                    reference_color = measured_color
                else:
                    reference_color = f"{self.ref_level}_color"
            if reference_layer == '':
                if layer in ref_adata.layers.keys():
                    reference_layer = layer
                else:
                    reference_layer = 'classification_space'

            reference_features = self.reference.layers[reference_layer].copy()
            measured_features = self.measured.layers[layer].copy()
            for bit1 in np.arange(0,self.reference.shape[1],2):
                bit2 = bit1+1
                
                percentile_min = 0.1
                percentile_max = 99.9
                # Calculate percentiles for reference data
                ref_x_min, ref_x_max = np.percentile(reference_features[:, bit1], [percentile_min, percentile_max])
                ref_y_min, ref_y_max = np.percentile(reference_features[:, bit2], [percentile_min, percentile_max])

                # Calculate percentiles for measured data
                meas_x_min, meas_x_max = np.percentile(measured_features[:, bit1], [percentile_min, percentile_max])
                meas_y_min, meas_y_max = np.percentile(measured_features[:, bit2], [percentile_min, percentile_max])

                # meas_x_min = np.max([meas_x_min,1])
                # meas_y_min = np.max([meas_y_min,1])
                meas_x_min = ref_x_min
                meas_y_min = ref_y_min
                meas_x_max = ref_x_max
                meas_y_max = ref_y_max

                # Clip the values and count how many points are above the limits
                ref_x_clipped = np.clip(reference_features[:, bit1], ref_x_min, ref_x_max)
                ref_y_clipped = np.clip(reference_features[:, bit2], ref_y_min, ref_y_max)
                meas_x_clipped = np.clip(measured_features[:, bit1], meas_x_min, meas_x_max)
                meas_y_clipped = np.clip(measured_features[:, bit2], meas_y_min, meas_y_max)

                ref_order = np.argsort(reference_features[:, bit1])
                meas_order = np.argsort(measured_features[:, bit1])
                ref_order = np.random.choice(ref_order, meas_order.shape[0])
                np.random.shuffle(ref_order)
                np.random.shuffle(meas_order)
                
                x_bins = np.linspace(ref_x_min, ref_x_max, 100)
                y_bins = np.linspace(ref_y_min, ref_y_max, 100)

                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                axs = axs.ravel()

                # Calculate density for reference data
                ref_density, ref_x_edges, ref_y_edges = np.histogram2d(ref_x_clipped[ref_order], ref_y_clipped[ref_order], bins=(x_bins, y_bins))
                # vmin, vmax = np.percentile(ref_density[ref_density > 0], [1, 99])
                # Plot density for reference data
                ref_density = np.flipud(ref_density.T)
                ref_density = np.log10(ref_density + 1)
                vmin, vmax = np.percentile(ref_density, [1, 99])
                im = axs[0].imshow(ref_density, cmap='jet', vmin=vmin, vmax=vmax)#, extent=[ref_x_min, ref_x_max, ref_y_min, ref_y_max])
                cbar = plt.colorbar(im, ax=axs[0])
                cbar.set_label('log10(Density)')
                axs[0].set_title('Reference Density')
                axs[0].axis('off')
                # axs[0].set_xlim(ref_x_min, ref_x_max)
                # axs[0].set_ylim(ref_y_min, ref_y_max)

                # Calculate density for measured data
                meas_density, meas_x_edges, meas_y_edges = np.histogram2d(meas_x_clipped[meas_order], meas_y_clipped[meas_order], bins=(x_bins, y_bins))
                # vmin, vmax = np.percentile(meas_density[meas_density > 0], [1, 99])
                # Plot density for measured data
                meas_density = np.flipud(meas_density.T)
                meas_density = np.log10(meas_density + 1)
                vmin, vmax = np.percentile(meas_density, [1, 99])
                im = axs[1].imshow(meas_density, cmap='jet', vmin=vmin, vmax=vmax)#, extent=[meas_x_min, meas_x_max, meas_y_min, meas_y_max])
                cbar = plt.colorbar(im, ax=axs[1])
                cbar.set_label('log10(Density)')
                axs[1].set_title('Measured Density')
                axs[1].axis('off')
                # axs[1].set_xlim(meas_x_min, meas_x_max)
                # axs[1].set_ylim(meas_y_min, meas_y_max)

                # Plot reference data
                axs[2].scatter(ref_x_clipped[ref_order], ref_y_clipped[ref_order], s=1, c=np.array(ref_adata.obs[reference_color])[ref_order], marker='.')
                axs[2].set_xlim(ref_x_min, ref_x_max)
                axs[2].set_ylim(ref_y_min, ref_y_max)
                axs[2].set_title('Reference')
                axs[2].set_xlabel(f"{self.reference.var.index[bit1]}")
                axs[2].set_ylabel(f"{self.reference.var.index[bit2]}")

                # Plot measured data
                axs[3].scatter(meas_x_clipped[meas_order], meas_y_clipped[meas_order], s=1, c=np.array(adata.obs[measured_color])[meas_order], marker='.')
                axs[3].set_xlim(meas_x_min, meas_x_max)
                axs[3].set_ylim(meas_y_min, meas_y_max)
                axs[3].set_title('Measured')

                plt.tight_layout()
                plt.show()

""" Load Simulation Data """
adata = anndata.read_h5ad(os.path.join(paths['Output']['Simulation'],f"{design_name}.h5ad"))
adata = adata[(adata.obs['ccf_x']>ccf_x_min) &(adata.obs['ccf_x']<ccf_x_max)].copy()
adata.obs['true_subclass'] = adata.obs['subclass'].copy()
adata.layers['raw'] = adata.X.copy()
complete_reference = anndata.read_h5ad(os.path.join(paths['Output']['Reference'],f"{design_name}.h5ad"))

""" Decode """
np.random.seed(42)
self = SingleCellAlignmentLeveragingExpectations(adata,complete_reference=complete_reference,visualize=False,verbose=False)
self.likelihood_only = False
self.calculate_spatial_priors()
self.load_reference()
self.model = LogisticRegression(max_iter=1000,random_state=42) 
self.supervised_neuron_annotation()
self.supervised_harmonization()
adata = self.measured.copy()
remove_index = {ct:ct[4:] for ct in adata.obs['subclass'].unique()}
adata.obs['predicted_subclass'] = adata.obs['subclass'].map(remove_index)
adata.write(os.path.join(paths['Output']['Results'],f"{design_name}.h5ad"))
accuracy = np.mean(np.array(adata.obs['predicted_subclass'].values)==np.array(adata.obs['true_subclass'].values))
logger.info(f"Final accuracy: {accuracy}")

""" Report Results """
results = {'accuracy': accuracy}
# save results to json
with open(os.path.join(paths['Output']['Results'],f"{design_name}.json"),'w') as f:
    json.dump(results,f)






