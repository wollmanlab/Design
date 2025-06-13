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
    python /Design/EncodingDesigner.py "path/to/parameters/file"
    ```

## To Run Multiple Jobs

1. **Format Reference Scripts:**
    * `data_format.py` - Formats the data for the reference.
        * **Parameters to change:** `data_path`, `output_path`, `csv_file`
        ```bash
        conda activate designer_3.12
        python /Design/data_format.py
        ```
    * `create_type_tree.py` - Creates the type tree for the reference.

2. **Run Scripts:**
    * `create_parameter_file.py` - Creates parameter files for `multi_param_file_optimization.sh`.
        * **Parameters to change:** `base_dir`, `parameter_variants`
        ```bash
        conda activate designer_3.12
        python /Design/create_parameter_file.py
        ```
    * `sub_multi_param_file_optimization.sh` - This script has a dual purpose:
        1. Creates the jobs.
        2. Executes the jobs by running `EncodingDesigner.py`.
        * **Parameters to change:** `OPT_DIR`, `CODE_DIR`
        ```bash
        /Design/sub_multi_param_file_optimization.sh
        ```





