
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import anndata
import os

"""
To Run qsub /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/sub_python_script.sh /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/data_format_lymph.py
"""

base_path = '/u/home/z/zeh/rwollman/data/Mouse_Lymph/Mouse_LN_Datasets'
output_path = f"{base_path}/Training_data/"

# Combine all datasets
cell_type_label = 'celltype'
shared_genes = ''
loaded_data = {}
for fname in os.listdir(base_path):
    print(f"Loading {fname}")
    if not fname.endswith('.h5ad'):
        continue
    adata = anndata.read_h5ad(os.path.join(base_path, fname))
    if isinstance(shared_genes, str):
        shared_genes = adata.var_names
    else:
        shared_genes = np.intersect1d(shared_genes, adata.var_names)
    loaded_data[fname] = adata
    print(fname,shared_genes.shape)
full_adata = ''
for fname,adata in loaded_data.items():
    print(f"Merging {fname}")
    dataset_name = fname.split('.')[0]
    adata = adata[:,shared_genes].copy()
    adata.obs_names = [f"{dataset_name}_{i}" for i in adata.obs.index]
    try:
        adata.X = np.array(adata.X)
        adata.X = adata.X.todense()
    except:
        pass
    if not 'celltype' in adata.obs:
        adata.obs['celltype'] = adata.obs['Cell_Type']
        adata.obs['dataset'] = fname.split('.')[0]
    if isinstance(full_adata, str):
        full_adata = adata
    else:
        full_adata = anndata.concat([full_adata, adata])
    print(fname,full_adata.shape)

del loaded_data
# Clean up cell type labels
cell_type_mapping = {
    # --- T Cell Subtypes ---
    # CD4+ T cells (Using 'CD4+ T cell' as the standard)
    'T_cell_CD4': 'CD4+ T cell',
    # CD8+ T cells (Using 'CD8+ T cell' as the standard)
    'T_cell_CD8': 'CD8+ T cell',
    # Gamma Delta T cells (Using 'T_cell_gd' as the standard from the list)
    'GD T Cell': 'T_cell_gd',
    # --- B Cell Subtypes ---
    # Germinal Center B cells (Using 'GC B cell' as the standard)
    'B_cell_GC': 'GC B cell',
    # Plasma Cells (Using 'Plasma cell' for clarity, removing underscore)
    'Plasma_cell': 'Plasma cell',
    'PC': 'Plasma cell',
    # --- NK Cells ---
    # Natural Killer cells (Using 'NK cell' for clarity, removing underscore)
    'NK_cell': 'NK cell',
    'NK': 'NK cell',
}
print(f"Mapping cell types")
cell_type_labels = np.array(full_adata.obs[cell_type_label])
for old_label, new_label in cell_type_mapping.items():
    cell_type_labels[full_adata.obs[cell_type_label] == old_label] = new_label
full_adata.obs[cell_type_label] = cell_type_labels
data_genes = full_adata.var.index.tolist()
cell_types = np.array(full_adata.obs[cell_type_label].unique())
cell_types = np.array(cell_types[cell_types != 'doublet'])
print(cell_types.shape[0])

# Scale X to sum of 100k per cell
print(f"Scaling X to sum of 100k per cell")
full_adata.obs['library_size'] = full_adata.X.sum(axis=1)
full_adata.obs['scaling_factor'] = 100000 / np.array(full_adata.obs['library_size'])[:,None]

# Make Gene Constraints
print(f"Making gene constraints")
csv_file = '/u/project/rwollman/data/mm10_probes_Oct28_2022.csv' # Gene Names not in ensemble format
df = pd.read_csv(csv_file)
df.columns = [str(col) for col in df.columns]
probe_genes = df['gname'].tolist()
shared_genes = sorted(list(set(data_genes) & set(probe_genes)))
filtered_df = df[df['gname'].isin(shared_genes)]
full_adata = full_adata[:,shared_genes]
probe_count_converter = {gene:cc for gene,cc in df['gname'].value_counts().items()}
constraints = np.array([probe_count_converter[gene] for gene in shared_genes])
constraints = np.clip(constraints,0,100)
gene_m = constraints>0
print(f"Number of genes with constraints: {np.sum(gene_m)}, % of genes: {np.mean(gene_m)*100}")
constraints_df = pd.DataFrame(constraints,index=shared_genes,columns=['constraints'])

#
""" Class Balance """
print(f"Making class balance")
tn = 100000
n = int(tn/cell_types.shape[0])
idxs = []
for j,ct in enumerate(cell_types):
    print(f"Class: {ct} ({j+1}/{cell_types.shape[0]})")
    m = full_adata.obs[cell_type_label]==ct
    if m.sum() > n:
        idxs.extend(np.random.choice(np.where(m)[0], n, replace = False))
    else:
        idxs.extend(np.random.choice(np.where(m)[0], n, replace = True))

chosen_cells = full_adata.obs.index[idxs]
full_adata = full_adata[chosen_cells].copy()
chosen_cell_metadata = full_adata.obs#.loc[chosen_cells]

# Convert str labels to integers
print(f"Converting str labels to integers")
categorical_converter = {k:i for i,k in enumerate(cell_types)}

# Make test train split
test_size = 0.2
random_state = 42
print(f"Making test train split")
X = torch.tensor(full_adata.X.toarray(),dtype=torch.float32)
X = X * torch.tensor(np.array(full_adata.obs['scaling_factor'])[:,None],dtype=torch.float32)
y = torch.tensor(np.array(full_adata.obs[cell_type_label].map(categorical_converter).values))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
print(f"Saving data")
if not os.path.exists(output_path):
    os.mkdir(output_path)
print(f"Saving Cell Metadata")
chosen_cell_metadata.to_csv(f"{output_path}/cell_metadata.csv")
print(f"Saving categorical converter")
pd.DataFrame(categorical_converter.values(),index=categorical_converter.keys(),columns=['label']).to_csv(f"{output_path}/categorical_converter.csv")
print(f"Saving constraints")
constraints_df.to_csv(f"{output_path}/constraints.csv")
print(f"Saving train data")
torch.save(X_train,f"{output_path}/X_train.pt")
torch.save(y_train,f"{output_path}/y_train.pt")
print(f"Saving test data")
torch.save(X_test,f"{output_path}/X_test.pt")
torch.save(y_test,f"{output_path}/y_test.pt")

print(f"Unique y test labels: {torch.unique(y_test)}")
print(f"Unique y train labels: {torch.unique(y_train)}")
print(f"Done")

