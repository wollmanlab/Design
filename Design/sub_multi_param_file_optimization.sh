#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o job_logs/job_log.$JOB_ID.$TASK_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=4:00:00,h_data=16G
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
conda activate designer_3.12

# Dual purpose script:
# 1. When run directly: submits an array job to process multipe design files
# 2. When run as part of array job: processes a specific design job

# Get git version information
GIT_REPO_DIR=$(dirname "$CODE_DIR")
COMMIT_HASH=$(git -C "$GIT_REPO_DIR" rev-parse HEAD)
SHORT_COMMIT_HASH=$(git -C "$GIT_REPO_DIR" rev-parse --short=7 HEAD)
BRANCH_NAME_RAW=$(git -C "$GIT_REPO_DIR" rev-parse --abbrev-ref HEAD)
# Check if repository is dirty (has uncommitted changes)
IS_DIRTY_OUTPUT=$(git -C "$GIT_REPO_DIR" status --porcelain)
if [ -n "$IS_DIRTY_OUTPUT" ]; then
  IS_DIRTY="true"
else
  IS_DIRTY="false"
fi

# Check if running as part of an array job (SGE_TASK_ID is set)
if [[ -n "$SGE_TASK_ID" ]]; then
    # WORKER MODE - Process a specific file
    # Get the list of files
    mapfile -t FILES < <(ls "$TODO_JOBS_DIR")
    
    # Get the file for this task
    CURRENT_FILE="${FILES[$((SGE_TASK_ID-1))]}"
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
    echo "repo_path:$REPO_PATH" >> "$FILE_PATH"
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
    mapfile -t FILES < <(ls "$TODO_JOBS_DIR")
    
    # Check if any files were found
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No files found in $TODO_JOBS_DIR. Exiting."
        exit 1
    fi
    
    echo "Found ${#FILES[@]} files to process"
    
    # Create the job array submission command
    # Replace both the array size and the OPT_DIR path
    mkdir -p "${OPT_DIR}/job_logs"
    sed -e "s/-t 1-N/-t 1-${#FILES[@]}/" \
        -e "s|job_logs/job_log.\$JOB_ID.\$TASK_ID|${OPT_DIR}/job_logs/job_log.\$JOB_ID.\$TASK_ID|" \
        "$0" > "${0}.tmp"    
    chmod +x "${0}.tmp"
    
    # Submit the job array
    qsub "${0}.tmp"
    
    # Clean up
    rm "${0}.tmp"
fi



