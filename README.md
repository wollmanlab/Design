# Design

Create Conda Environment
conda create -n "designer_3.12" python=3.12
conda activate designer_3.12
pip install torch
pip install scikit-learn
pip install anndata
pip install ipykernel

To Run 
First Format Data /Design/Notebooks/data_format.ipynb
Second Format Parameters /Design/Notebooks/parameters_format.ipynb
third run executable python /Design/EncodingDesigner.py "path/to/parameters/file"