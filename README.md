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

## To Run

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