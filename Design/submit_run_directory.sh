#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o job_logs/job_log.$JOB_ID.$TASK_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=24:00:00,h_data=16G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
#$ -t 1-N  # This will be replaced with actual number of files

# Get the current username
CURRENT_USER=$(whoami)
echo "User identified as ${CURRENT_USER}"

# Get first letter of username for path
USER_FIRST_LETTER=${CURRENT_USER:0:1}

# User specific variables
if [ "$CURRENT_USER" = "rwollman" ]; then
    CODE_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Design/Design"
    BASE_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Runs"
    CONDA_PATH="/u/home/r/rwollman/miniconda3/etc/profile.d/conda.sh"
elif [ "$CURRENT_USER" = "zeh" ]; then
    CODE_DIR="/u/home/z/zeh/rwollman/zeh/Repos/Design/Design"
    BASE_DIR="/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs"
    CONDA_PATH="/u/home/z/zeh/miniconda3/etc/profile.d/conda.sh"
else
    echo "Using default paths for user ${CURRENT_USER}"
    CODE_DIR="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/rwollman/${CURRENT_USER}/Repos/Design/Design"
    BASE_DIR="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/rwollman/${CURRENT_USER}/Projects/Design/Runs"
    CONDA_PATH="/u/home/${USER_FIRST_LETTER}/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
fi

echo "BASE_DIR: ${BASE_DIR}"
echo "CODE_DIR: ${CODE_DIR}"
echo "CONDA_PATH: ${CONDA_PATH}"

# Check if conda.sh exists
if [ ! -f "$CONDA_PATH" ]; then
    echo "Error: Conda initialization script not found at ${CONDA_PATH}"
    exit 1
fi

# Check if a run directory argument was provided
if [ $# -ge 1 ]; then
    RUN_DIR="${BASE_DIR}/$1"
else
    echo "Error: You must provide a run directory name as an argument."
    echo "Usage: $0 <run_directory_name>"
    echo "Example: $0 Run0"
    exit 1
fi

# Check if the run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

DESIGN_RESULTS_DIR="${RUN_DIR}/design_results"

# Check if design_results directory exists
if [ ! -d "$DESIGN_RESULTS_DIR" ]; then
    echo "Error: design_results directory not found: $DESIGN_RESULTS_DIR"
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

# Check if running as part of an array job (SGE_TASK_ID is set)
if [[ -n "$SGE_TASK_ID" ]]; then
    # WORKER MODE - Process a specific file
    LIST_FILE="${RUN_DIR}/.files_to_process_task_list.txt" # Path to the list file

    # Get the list of files from the pre-generated list
    if [[ ! -f "$LIST_FILE" ]]; then
        echo "ERROR: File list $LIST_FILE not found. This file should have been created by the submitter."
        exit 1
    fi
    mapfile -t FILES < "$LIST_FILE"
    
    # Check if SGE_TASK_ID is valid for the number of files in the list
    if [[ $SGE_TASK_ID -gt ${#FILES[@]} ]] || [[ $SGE_TASK_ID -lt 1 ]]; then
        echo "ERROR: SGE_TASK_ID $SGE_TASK_ID is out of range for the number of files listed (${#FILES[@]}) in $LIST_FILE."
        exit 1
    fi
    
    # Get the file for this task
    CURRENT_FILE="${FILES[$((SGE_TASK_ID-1))]}"

    if [[ -z "$CURRENT_FILE" ]]; then
        echo "ERROR: Failed to retrieve a filename for SGE_TASK_ID $SGE_TASK_ID from $LIST_FILE. The line might be empty or index out of bounds."
        echo "Total files in list: ${#FILES[@]}. Task index attempted: $((SGE_TASK_ID-1))."
        exit 1
    fi
    
    FILE_PATH="${DESIGN_RESULTS_DIR}/${CURRENT_FILE}/used_user_parameters.csv"
    
    echo "===== RUNNING AS WORKER: Processing file: ${CURRENT_FILE} ====="
    echo "Task ID: $SGE_TASK_ID"
    echo "Hostname: $(hostname)"
    echo "Start time: $(date)"
    echo "Parameter file: $FILE_PATH"
    
    if [[ ! -f "$FILE_PATH" ]]; then
        echo "ERROR: Parameter file $FILE_PATH not found"
        exit 1
    fi

    # Run the calculation with the parameter file
    python -u "${CODE_DIR}/EncodingDesigner.py" "$FILE_PATH" 
    
    EXIT_CODE=$?
    echo "Job completed with exit code $EXIT_CODE"
    echo "End time: $(date)"
    exit $EXIT_CODE
    
else
    # SUBMITTER MODE - Submit jobs as an array
    echo "===== RUNNING AS SUBMITTER ====="
    echo "Processing run directory: $RUN_DIR"
    echo "Design results directory: $DESIGN_RESULTS_DIR"
    
    # Create a list of all subdirectories in design_results that contain used_user_parameters.csv
    echo "Finding parameter files to process..."
    
    # Define the persistent list file path
    LIST_FILE="${RUN_DIR}/.files_to_process_task_list.txt"
    
    # Find all subdirectories that contain used_user_parameters.csv
    find "$DESIGN_RESULTS_DIR" -name "used_user_parameters.csv" -type f | while read -r param_file; do
        # Get the directory name (relative to design_results)
        dir_path=$(dirname "$param_file")
        dir_name=$(basename "$dir_path")
        echo "$dir_name" >> "$LIST_FILE"
    done
    
    # Read the generated list into an array to count files
    mapfile -t FILES < "$LIST_FILE"
    
    # Check if any files were found
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No parameter files found in $DESIGN_RESULTS_DIR. Exiting."
        rm -f "$LIST_FILE" # Clean up the list file if no files were found
        exit 1
    fi
    
    echo "Found ${#FILES[@]} parameter files to process"
    
    # Get the first file to read n_cpu from it
    FIRST_FILE_TO_PROCESS="${FILES[0]}"
    FIRST_FILE_PATH="${DESIGN_RESULTS_DIR}/${FIRST_FILE_TO_PROCESS}/used_user_parameters.csv"
    
    N_CPU=1 # Default value if not found or invalid
    if [[ -f "$FIRST_FILE_PATH" ]]; then
        # Extract the value after "n_cpu,"
        N_CPU_VALUE_FROM_FILE=$(grep '^n_cpu,' "$FIRST_FILE_PATH" | cut -d',' -f2)
        # Check if the extracted value is a non-empty number
        if [[ -n "$N_CPU_VALUE_FROM_FILE" ]] && [[ "$N_CPU_VALUE_FROM_FILE" =~ ^[0-9]+$ ]]; then
            N_CPU=$N_CPU_VALUE_FROM_FILE
            echo "Successfully read n_cpu=${N_CPU} from ${FIRST_FILE_PATH} for SGE job."
        else
            echo "Warning: Could not read a valid n_cpu value from ${FIRST_FILE_PATH} (found: '${N_CPU_VALUE_FROM_FILE}'). Using default n_cpu=${N_CPU}."
        fi
    else
        echo "Warning: First parameter file ${FIRST_FILE_PATH} not found. Using default n_cpu=${N_CPU}."
    fi
    
    # Create the job array submission command
    # Replace the array size, the RUN_DIR path for logs, and the number of CPUs for -pe shared
    mkdir -p "${RUN_DIR}/job_logs"
    sed -e "s/-t 1-N/-t 1-${#FILES[@]}/" \
        -e "s|job_logs/job_log.\$JOB_ID.\$TASK_ID|${RUN_DIR}/job_logs/job_log.\$JOB_ID.\$TASK_ID|" \
        -e "s/^#\$ -pe shared [0-9][0-9]*/#\$ -pe shared ${N_CPU}/" \
        "$0" > "${0}.tmp"
    chmod +x "${0}.tmp"
    
    # Submit the job array
    qsub "${0}.tmp" "$1"
    
    # Clean up
    rm "${0}.tmp"
    
    echo "Submitted ${#FILES[@]} jobs for run directory: $1"
fi 