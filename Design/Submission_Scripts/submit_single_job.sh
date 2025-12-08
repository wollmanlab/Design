#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o job_logs/job_log.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=24:00:00,h_data=16G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1

# Get the current username
CURRENT_USER=$(whoami)
echo "User identified as ${CURRENT_USER}"

# User specific variables
if [ "$CURRENT_USER" = "rwollman" ]; then
    CODE_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Design/Design"
    CONDA_PATH="/u/home/r/rwollman/miniconda3/etc/profile.d/conda.sh"
elif [ "$CURRENT_USER" = "zeh" ]; then
    CODE_DIR="/u/home/z/zeh/rwollman/zeh/Repos/Design/Design"
    CONDA_PATH="/u/home/z/zeh/miniconda3/etc/profile.d/conda.sh"
else
    echo "Using default paths for user ${CURRENT_USER}"
    CODE_DIR="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/rwollman/${CURRENT_USER}/Repos/Design/Design"
    CONDA_PATH="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
fi

echo "CODE_DIR: ${CODE_DIR}"
echo "CONDA_PATH: ${CONDA_PATH}"

# Check if conda.sh exists
if [ ! -f "$CONDA_PATH" ]; then
    echo "Error: Conda initialization script not found at ${CONDA_PATH}"
    exit 1
fi

# Check if a parameter file argument was provided
if [ $# -ge 1 ]; then
    PARAM_FILE="$1"
else
    echo "Error: You must provide a parameter file path as an argument."
    echo "Usage: $0 /path/to/parameter/file.csv"
    exit 1
fi

# Check if the parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
    echo "Error: Parameter file not found: $PARAM_FILE"
    exit 1
fi

# load the job environment:
. /u/local/Modules/default/init/modules.sh
module load conda

# Initialize conda for bash shell
source "${CONDA_PATH}"
conda activate designer_3.12

# Verify conda environment
echo "Using conda environment: $(which python)"
echo "Python version: $(python --version)"

echo "===== RUNNING SINGLE JOB ====="
echo "Parameter file: $PARAM_FILE"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"

# Run the calculation
python -u "${CODE_DIR}/CIPHER.py" "$PARAM_FILE"

EXIT_CODE=$?
echo "Job completed with exit code $EXIT_CODE"
echo "End time: $(date)"
exit $EXIT_CODE 