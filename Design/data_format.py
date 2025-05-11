# %%
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import anndata
import os

# %%
data_path = '/u/project/rwollman/data/Allen_WMB_2024Mar06'

# %%
available_cells = np.array(pd.read_csv(f"{data_path}/metadata/WMB-10X/20230830/available_cells.csv",index_col=0).index)
shared_genes = np.array(pd.read_csv(f"{data_path}/metadata/WMB-10X/20230830/shared_genes.csv",index_col=0).index)
cell_metadata = pd.read_csv(f"{data_path}/metadata/WMB-10X/20230830/updated_cell_metadata.csv",index_col=0)
ensemble_mapper = pd.read_csv(f"{data_path}/metadata/WMB-10X/20230830/ensemble_mapper.csv",index_col=0)
ensemble_mapper = dict(zip(ensemble_mapper.index, ensemble_mapper['gene_symbol']))
reverse_ensemble_mapper = {i:j for j, i in ensemble_mapper.items()}
cell_metadata = cell_metadata.loc[available_cells]

cell_type_label = 'subclass'
region_label = 'region_of_interest_acronym'


cell_types = np.array(cell_metadata[cell_type_label].unique())
region_types = np.array(cell_metadata[region_label].unique())
print(cell_types.shape[0],region_types.shape[0])
cell_metadata

# %%
csv_file = '/u/project/rwollman/data/mm10_probes_Oct28_2022.csv'
df = pd.read_csv(csv_file)
gene_list = shared_genes#genes['gene_symbol'].tolist()
gene_set = set(gene_list)
try:
  filtered_df = df[df['gene'].isin(gene_set)]
except KeyError:
  #handling the case that gene column not called gene
  df.columns = [str(col) for col in df.columns]   #make sure all the columns are string to be consistent with reading with csv files
  filtered_df = df[df.iloc[:, -4].isin(gene_set)] #use the original -4 column

df = filtered_df
vector_converter = {gene:cc for gene,cc in df['gname'].value_counts().items()}
mising_genes = []
for gene in gene_list:
    if gene not in vector_converter.keys():
        mising_genes.append(gene)
        vector_converter[gene] = 0
print(len(mising_genes),len(gene_list),df['gname'].unique().shape[0])
constraints = np.array([vector_converter[gene] for gene in gene_list])
constraints = np.clip(constraints,0,100)
gene_m = constraints>0
print(np.sum(gene_m))
genes = gene_list[gene_m]
genes_ensemble = np.array([reverse_ensemble_mapper[gene] for gene in genes])
constraints = constraints[gene_m]
constraints_df = pd.DataFrame(constraints,index=genes,columns=['constraints'])

# %%
constraints_df

# %%
""" Class Balance """
tn = 100000
n = int(tn/cell_types.shape[0])
idxs = []
for ct in cell_types:
    m = cell_metadata[cell_type_label]==ct
    if m.sum() > n:
        idxs.extend(np.random.choice(np.where(m)[0], n, replace = False))
    else:
        idxs.extend(np.random.choice(np.where(m)[0], n, replace = True))
chosen_cells = cell_metadata.index[idxs]
chosen_cell_metadata = cell_metadata.loc[chosen_cells]

# %%
X_train = ''
categorical_converter = {k:i for i,k in enumerate(cell_types)}
region_categorical_converter = {k:i for i,k in enumerate(region_types)}
for feature_matrix_label in chosen_cell_metadata['feature_matrix_label'].unique():
    m = chosen_cell_metadata['feature_matrix_label'] == feature_matrix_label
    temp_chosen_cells = chosen_cells[m]
    if '10Xv2' in feature_matrix_label:
        t = '10Xv2'
    elif '10Xv3' in feature_matrix_label:
        t = '10Xv3'
    adata_path = f"{data_path}/expression_matrices/WMB-{t}/20230630/{feature_matrix_label}-raw.h5ad"
    data = anndata.read_h5ad(adata_path,backed='r')
    temp_adata = data[temp_chosen_cells,:].to_memory()[:,genes_ensemble].copy()
    correction = torch.tensor(100000/np.array(temp_adata.obs['library_size']))
    X = torch.tensor(temp_adata.X.todense())*correction[:,None]
    y = torch.tensor(np.array(temp_adata.obs[cell_type_label].map(categorical_converter).values))
    r = torch.tensor(np.array(temp_adata.obs[region_label].map(region_categorical_converter).values))
    idxs = torch.tensor(np.arange(X.shape[0]))
    X_train_temp, X_test_temp, idxs_train_temp, idxs_test_temp = train_test_split(X,idxs, test_size = 0.2, random_state = 42)
    y_train_temp = y[idxs_train_temp]
    y_test_temp = y[idxs_test_temp]
    r_train_temp = r[idxs_train_temp]
    r_test_temp = r[idxs_test_temp]
    if isinstance(X_train, str):
        X_train = X_train_temp
        y_train = y_train_temp
        r_train = r_train_temp
        X_test = X_test_temp
        y_test = y_test_temp
        r_test = r_test_temp
    else:
        X_train = torch.cat([X_train,X_train_temp],0)
        y_train = torch.cat([y_train,y_train_temp],0)
        X_test = torch.cat([X_test,X_test_temp],0)
        y_test = torch.cat([y_test,y_test_temp],0)
        r_train = torch.cat([r_train,r_train_temp],0)
        r_test = torch.cat([r_test,r_test_temp],0)

# %%
X_train.shape,X_test.shape,y_train.shape,y_test.shape,r_train.shape,r_test.shape

# %%
output_path = f"{data_path}/Training_data/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
pd.DataFrame(categorical_converter.values(),index=categorical_converter.keys(),columns=['label']).to_csv(f"{output_path}/categorical_converter.csv")
pd.DataFrame(region_categorical_converter.values(),index=region_categorical_converter.keys(),columns=['label']).to_csv(f"{output_path}/region_categorical_converter.csv")
constraints_df.to_csv(f"{output_path}/constraints.csv")
torch.save(X_train,f"{output_path}/X_train.pt")
torch.save(y_train,f"{output_path}/y_train.pt")
torch.save(X_test,f"{output_path}/X_test.pt")
torch.save(y_test,f"{output_path}/y_test.pt")
torch.save(r_test,f"{output_path}/r_test.pt")
torch.save(r_train,f"{output_path}/r_train.pt")


