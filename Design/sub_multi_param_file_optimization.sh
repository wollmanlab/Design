#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o job_logs/job_log.$JOB_ID.$TASK_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=6:00:00,h_data=16G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
#$ -t 1-N  # This will be replaced with actual number of files

# Define paths first
OPT_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Runs"
# Check if a directory argument was provided
if [ $# -ge 1 ]; then
    OPT_DIR="${OPT_DIR}/$1"
else
    echo "Error: You must provide a run directory name as an argument."
    echo "Usage: $0 /path/to/run/directory"
    exit 1
fi

TODO_JOBS_DIR="${OPT_DIR}/params_files_to_scan"
FINISHED_JOBS_DIR="${OPT_DIR}/params_files_scanned"
CODE_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Design/Design"
OUTPUT_DIR="${OPT_DIR}/design_results"

# load the job environment:
. /u/local/Modules/default/init/modules.sh
module load conda

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
conda activate designer_3.12

# Dual purpose script:
# 1. When run directly: submits an array job to process multipe design files
# 2. When run as part of array job: processes a specific design job

# Get git version information
GIT_REPO_DIR=$(dirname "$CODE_DIR")
cd "$GIT_REPO_DIR"
COMMIT_HASH=$(git rev-parse HEAD)
SHORT_COMMIT_HASH=$(git rev-parse --short=7 HEAD)
BRANCH_NAME_RAW=$(git rev-parse --abbrev-ref HEAD)
# Check if repository is dirty (has uncommitted changes)
IS_DIRTY_OUTPUT=$(git status --porcelain)
if [ -n "$IS_DIRTY_OUTPUT" ]; then
  IS_DIRTY="true"
else
  IS_DIRTY="false"
fi
# Change back to original directory
cd - > /dev/null

# Check if running as part of an array job (SGE_TASK_ID is set)
if [[ -n "$SGE_TASK_ID" ]]; then
    # WORKER MODE - Process a specific file
    LIST_FILE="${OPT_DIR}/.files_to_process_task_list.txt" # Path to the list file

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
    
    FILE_PATH="${TODO_JOBS_DIR}/${CURRENT_FILE}"
    
    echo "===== RUNNING AS WORKER: Processing file: ${CURRENT_FILE} ====="
    echo "Task ID: $SGE_TASK_ID"
    echo "Hostname: $(hostname)"
    echo "Start time: $(date)"
    
    if [[ ! -f "$FILE_PATH" ]]; then
        echo "ERROR: File $FILE_PATH not found"
        exit 1
    fi

    # Update the output directory in the parameter file
    # Extract the filename without extension
    FILENAME=$(basename "$CURRENT_FILE" .csv)

    # add to FILENAME the git information
    # Add git information to the parameter file
    echo "repo_path:$GIT_REPO_DIR" >> "$FILE_PATH"
    echo "commit_hash:$COMMIT_HASH" >> "$FILE_PATH"
    echo "short_commit_hash:$SHORT_COMMIT_HASH" >> "$FILE_PATH"
    echo "branch_name:$BRANCH_NAME_RAW" >> "$FILE_PATH"
    echo "is_dirty:$IS_DIRTY" >> "$FILE_PATH"
    
    # Create a full output directory path that includes the filename
    FULL_OUTPUT_DIR="${OUTPUT_DIR}/${FILENAME}"
    
    # Create the directory if it doesn't exist
    mkdir -p "$FULL_OUTPUT_DIR"
    sed -i "s|^output,.*|output,${FULL_OUTPUT_DIR}|" "$FILE_PATH"
    
    # Run the calculation with the job file directly
    python -u "${CODE_DIR}/EncodingDesigner.py" "$FILE_PATH" 
    
    EXIT_CODE=$?
    echo "Job completed with exit code $EXIT_CODE"
    
    # If successful, move file to completed directory
    if [[ $EXIT_CODE -eq 0 ]]; then
        mv "$FILE_PATH" "${FINISHED_JOBS_DIR}/"
        echo "File moved to completed directory"
    fi
    
    echo "End time: $(date)"
    exit $EXIT_CODE
else
    # SUBMITTER MODE - Submit jobs as an array
    echo "===== RUNNING AS SUBMITTER ====="
    
    # Create a list of all files to process
    echo "Finding files to process..."
    # Define the persistent list file path
    LIST_FILE="${OPT_DIR}/.files_to_process_task_list.txt"
    
    # Create the list of files by listing the directory contents
    # This list will be used by all worker jobs
    ls "$TODO_JOBS_DIR" > "$LIST_FILE"
    
    # Read the generated list into an array to count files
    mapfile -t FILES < "$LIST_FILE"
    
    # Check if any files were found
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No files found in $TODO_JOBS_DIR. Exiting."
        rm -f "$LIST_FILE" # Clean up the list file if no files were found
        exit 1
    fi
    
    echo "Found ${#FILES[@]} files to process"
    
    # Get the first file to read n_cpu from it
    FIRST_FILE_TO_PROCESS="${FILES[0]}"
    FIRST_FILE_PATH="${TODO_JOBS_DIR}/${FIRST_FILE_TO_PROCESS}"
    
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
    # Replace the array size, the OPT_DIR path for logs, and the number of CPUs for -pe shared
    mkdir -p "${OPT_DIR}/job_logs"
    sed -e "s/-t 1-N/-t 1-${#FILES[@]}/" \
        -e "s|job_logs/job_log.\$JOB_ID.\$TASK_ID|${OPT_DIR}/job_logs/job_log.\$JOB_ID.\$TASK_ID|" \
        -e "s/^#\$ -pe shared [0-9][0-9]*/#\$ -pe shared ${N_CPU}/" \
        "$0" > "${0}.tmp"
    chmod +x "${0}.tmp"
    
    # Submit the job array
    qsub "${0}.tmp" "$1"
    
    # Clean up
    rm "${0}.tmp"
fi



