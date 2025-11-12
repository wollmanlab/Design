# Design

## Create Conda Environment

To set up the necessary environment, follow these steps:

1.  **Create the Conda environment:**
    ```bash
    conda create -n "designer_3.12" python=3.12
    ```

2.  **Activate the Conda environment:**
    ```bash
    conda activate designer_3.12
    ```

3.  **Install required packages:**
    ```bash
    pip install torch
    pip install scikit-learn
    pip install anndata
    pip install ipykernel
    pip install matplotlib
    pip install seaborn
    pip install scanpy
    ```

## To Run Single Jobs

Follow these steps to execute the designer:

1.  **Format Data:**
    Open and run the Jupyter Notebook located at `/Design/Notebooks/data_format.ipynb`.
2.  **Format Parameters:**
    Open and run the Jupyter Notebook located at `/Design/Notebooks/parameters_format.ipynb`.
3.  **Run Executable:**
    Execute the main script using the following command in your terminal, replacing `"path/to/parameters/file"` with the actual path to your parameters file:
    ```bash
    python /Design/CIPHER.py "path/to/parameters/file"
    ```

## To Run Multiple Jobs

1. **Format Reference Scripts:**
    * `data_format.py` - Formats the data for the reference.
        * **Parameters to change:** `data_path`, `output_path`, `csv_file`
        ```bash
        conda activate designer_3.12
        python ./Design/data_format.py
        ```
    * `create_type_tree.py` - Creates the type tree for the reference.

2. **Run Scripts:**
    * `create_parameter_file.py` - Creates parameter files and automatically submits jobs.
        * **Parameters to change:** `base_dir`, `parameter_variants`
        ```bash
        conda activate designer_3.12
        python ./Design/create_parameter_file.py [Run#]
        ```
        * **Run number:** You can specify a run number (e.g., `Run0`, `Run1`) or let the script automatically find the next available run number.
        * This script now automatically calls `sub_multi_param_file_optimization.sh` after creating the parameter files, so you only need to run one command.
    * `sub_multi_param_file_optimization.sh` - This script has a dual purpose:
        1. Creates the jobs.
        2. Executes the jobs by running `CIPHER.py`.
        * **Parameters to change:** `OPT_DIR`, `CODE_DIR`, replace Run0 
        * **Note:** This script is now automatically called by `create_parameter_file.py`, but can still be run manually if needed:
        ```bash
        ./Design/sub_multi_param_file_optimization.sh Run0
        ```





