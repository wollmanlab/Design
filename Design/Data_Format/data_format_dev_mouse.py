
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import anndata
import os

"""
To Run qsub /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/sub_python_script.sh /u/home/z/zeh/rwollman/zeh/Repos/Design/Design/data_format_dev_mouse.py
"""

base_path = '/u/home/z/zeh/rwollman/data/Dev_Mouse'
output_path = f"{base_path}/Training_data/"

# Combine all datasets
cell_type_label = 'author_cell_type'
full_adata = anndata.read_h5ad('/u/home/z/zeh/rwollman/data/Dev_Mouse/a5a85963-8004-41a1-8eb5-ca65266d89c3.h5ad',backed='r')

full_adata.var['Ensembl_gene_id'] = full_adata.var_names
full_adata.var_names = full_adata.var['gene_short_name'].values
data_genes = full_adata.var_names
print(f"Shared genes: {data_genes}")

print(f"Mapping cell types")
cell_types = np.array(full_adata.obs[cell_type_label].unique())
print(cell_types.shape[0])

# Scale X to sum of 100k per cell
print(f"Scaling X to sum of 100k per cell")
# full_adata.obs['library_size'] = sc.pp.calculate_qc_metrics(full_adata)[0]['total_counts']#full_adata.X.sum(axis=1)
# full_adata.obs['scaling_factor'] = 100000 / np.array(full_adata.obs['library_size'])[:,None]

# Make Gene Constraints
print(f"Making gene constraints")
csv_file = '/scratchdata1/ExternalData/mm10_probes_Oct28_2022.csv' # Gene Names not in ensemble format
df = pd.read_csv(csv_file)
df.columns = [str(col) for col in df.columns]
probe_genes = df['gname'].tolist()
shared_genes = sorted(list(set(data_genes) & set(probe_genes)))
filtered_df = df[df['gname'].isin(shared_genes)]
# full_adata = full_adata[:,shared_genes]
probe_count_converter = {gene:cc for gene,cc in df['gname'].value_counts().items()}
constraints = np.array([probe_count_converter[gene] for gene in shared_genes])
constraints = np.clip(constraints,0,100)
gene_m = constraints>0
print(f"Number of genes with constraints: {np.sum(gene_m)}, % of genes: {np.mean(gene_m)*100}")
constraints_df = pd.DataFrame(constraints,index=shared_genes,columns=['constraints'])

""" Class Balance """
print(f"Making class balance")
tn = 500000
n = int(tn/cell_types.shape[0])
idxs = []
for j,ct in enumerate(cell_types):
    # print(f"Class: {ct} ({j+1}/{cell_types.shape[0]})")
    m = full_adata.obs[cell_type_label]==ct
    if m.sum() > n:
        idxs.extend(np.random.choice(np.where(m)[0], n, replace = False))
    else:
        idxs.extend(np.random.choice(np.where(m)[0], n, replace = True))

chosen_cells = full_adata.obs.index[idxs]

chunk = full_adata[chosen_cells].X

library_size = chunk.sum(axis=1)
scaling_factor = 100000 / library_size

chosen_cell_metadata = full_adata[chosen_cells].obs.copy()#.loc[chosen_cells]
chosen_cell_metadata['library_size'] = library_size
chosen_cell_metadata['scaling_factor'] = scaling_factor

m = np.isin(full_adata.var_names,shared_genes)
ordered_shared_genes = full_adata.var_names[m]
constraints_df = constraints_df.loc[ordered_shared_genes]

# Convert str labels to integers
print(f"Converting str labels to integers")
categorical_converter = {k:i for i,k in enumerate(cell_types)}

# Make test train split
test_size = 0.2
random_state = 42
print(f"Making test train split")
del full_adata
import gc
gc.collect()
X = torch.tensor(chunk[:,m].toarray(),dtype=torch.float32)
del chunk
X = X * torch.tensor(np.array(chosen_cell_metadata['scaling_factor'])[:,None],dtype=torch.float32)
y = torch.tensor(np.array(chosen_cell_metadata[cell_type_label].map(categorical_converter).values))
print(f"Splitting data")
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