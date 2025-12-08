#!/bin/bash
# Usage: ./submit_all_single_jobs_in_run.sh Run1_Figures

if [ $# -ne 1 ]; then
    echo "Usage: $0 <RunDirName>"
    exit 1
fi

RUN_DIR_NAME="$1"

# User specific variables
CURRENT_USER=$(whoami)
USER_FIRST_LETTER=${CURRENT_USER:0:1}
if [ "$CURRENT_USER" = "rwollman" ]; then
    BASE_RUNS_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Runs"
    CODE_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Design/Design"
    CONDA_PATH="/u/home/r/rwollman/miniconda3/etc/profile.d/conda.sh"
elif [ "$CURRENT_USER" = "zeh" ]; then
    BASE_RUNS_DIR="/u/home/z/zeh/project-rwollman/Projects/Design/Runs"
    CODE_DIR="/u/home/z/zeh/rwollman/zeh/Repos/Design/Design"
    CONDA_PATH="/u/home/z/zeh/miniconda3/etc/profile.d/conda.sh"
else
    echo "Using default paths for user ${CURRENT_USER}"
    BASE_RUNS_DIR="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/rwollman/${CURRENT_USER}/Projects/Design/Runs"
    CODE_DIR="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/rwollman/${CURRENT_USER}/Repos/Design/Design"
    CONDA_PATH="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
fi

RUN_DIR="${BASE_RUNS_DIR}/${RUN_DIR_NAME}"
DESIGN_RESULTS_DIR="${RUN_DIR}/design_results"

if [ ! -d "$DESIGN_RESULTS_DIR" ]; then
    echo "Error: design_results directory not found: $DESIGN_RESULTS_DIR"
    exit 1
fi

PARAM_FILES=()
while IFS= read -r -d '' file; do
    PARAM_FILES+=("$file")
done < <(find "$DESIGN_RESULTS_DIR" -type f -name "used_user_parameters.csv" -print0)

if [ ${#PARAM_FILES[@]} -eq 0 ]; then
    echo "No parameter files found in $DESIGN_RESULTS_DIR"
    exit 1
fi

echo "Found ${#PARAM_FILES[@]} parameter files. Submitting jobs..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for param_file in "${PARAM_FILES[@]}"; do
    echo "Submitting: $param_file"
    qsub "${SCRIPT_DIR}/submit_single_job.sh" "$param_file"
done

echo "All jobs submitted." 