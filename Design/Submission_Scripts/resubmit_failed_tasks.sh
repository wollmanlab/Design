#!/bin/bash

# Script to automatically detect and resubmit failed tasks from a run directory
# Usage: ./resubmit_failed_tasks.sh <run_directory_name>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_directory_name>"
    echo "Example: $0 Run3"
    exit 1
fi

RUN_DIR="$1"
CURRENT_USER=$(whoami)

# User specific variables
if [ "$CURRENT_USER" = "rwollman" ]; then
    BASE_DIR="/u/home/r/rwollman/project-rwollman/atlas_design/Runs"
elif [ "$CURRENT_USER" = "zeh" ]; then
    BASE_DIR="/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs"
else
    BASE_DIR="/u/home/z/zeh/rwollman/zeh/Projects/Design/Runs"
fi

RUN_PATH="${BASE_DIR}/${RUN_DIR}"

if [ ! -d "$RUN_PATH" ]; then
    echo "Error: Run directory $RUN_PATH does not exist"
    exit 1
fi

echo "Checking for failed tasks in run directory: $RUN_PATH"

# Get the job ID from the job logs directory
JOB_LOGS_DIR="${RUN_PATH}/job_logs"
if [ ! -d "$JOB_LOGS_DIR" ]; then
    echo "Error: Job logs directory $JOB_LOGS_DIR does not exist"
    exit 1
fi

# Find the most recent job ID by looking at log files
JOB_ID=$(ls "$JOB_LOGS_DIR" | grep "job_log\." | head -1 | sed 's/job_log\.\([0-9]*\)\..*/\1/')

if [ -z "$JOB_ID" ]; then
    echo "Error: Could not find job ID from log files"
    exit 1
fi

echo "Found job ID: $JOB_ID"

# Check current job status and find failed tasks
echo "Checking job status..."
qstat -j "$JOB_ID" > /tmp/job_status_$$.txt 2>&1

# Extract failed task IDs
FAILED_TASKS=$(grep "job_state.*Eqw" /tmp/job_status_$$.txt | awk '{print $2}' | sed 's/://' | tr '\n' ' ')

if [ -z "$FAILED_TASKS" ]; then
    echo "No failed tasks found for job $JOB_ID"
    rm -f /tmp/job_status_$$.txt
    exit 0
fi

echo "Found failed tasks: $FAILED_TASKS"

# Get the task list file
TASK_LIST_FILE="${RUN_PATH}/.files_to_process_task_list.txt"
if [ ! -f "$TASK_LIST_FILE" ]; then
    echo "Error: Task list file $TASK_LIST_FILE not found"
    rm -f /tmp/job_status_$$.txt
    exit 1
fi

# Create a new job script for just the failed tasks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SCRIPT="${SCRIPT_DIR}/sub_multi_param_file_optimization.sh"
NEW_SCRIPT="${SCRIPT_DIR}/sub_multi_param_file_optimization_failed_tasks.sh"

# Copy the original script
cp "$ORIGINAL_SCRIPT" "$NEW_SCRIPT"

# Create a new task list with only the failed tasks
FAILED_TASK_LIST="${RUN_PATH}/.failed_tasks_list.txt"
for task_id in $FAILED_TASKS; do
    sed -n "${task_id}p" "$TASK_LIST_FILE" >> "$FAILED_TASK_LIST"
done

# Count the number of failed tasks
FAILED_COUNT=$(wc -l < "$FAILED_TASK_LIST")

echo "Creating resubmission script for $FAILED_COUNT failed tasks..."

# Get the first failed task file to read n_cpu from it
FIRST_FAILED_FILE=$(head -1 "$FAILED_TASK_LIST")
FIRST_FAILED_FILE_PATH="${RUN_PATH}/params_files_to_scan/${FIRST_FAILED_FILE}"

N_CPU=1 # Default value if not found or invalid
if [[ -f "$FIRST_FAILED_FILE_PATH" ]]; then
    # Extract the value after "n_cpu,"
    N_CPU_VALUE_FROM_FILE=$(grep '^n_cpu,' "$FIRST_FAILED_FILE_PATH" | cut -d',' -f2)
    # Check if the extracted value is a non-empty number
    if [[ -n "$N_CPU_VALUE_FROM_FILE" ]] && [[ "$N_CPU_VALUE_FROM_FILE" =~ ^[0-9]+$ ]]; then
        N_CPU=$N_CPU_VALUE_FROM_FILE
        echo "Successfully read n_cpu=${N_CPU} from ${FIRST_FAILED_FILE_PATH} for SGE job."
    else
        echo "Warning: Could not read a valid n_cpu value from ${FIRST_FAILED_FILE_PATH} (found: '${N_CPU_VALUE_FROM_FILE}'). Using default n_cpu=${N_CPU}."
    fi
else
    echo "Warning: First failed parameter file ${FIRST_FAILED_FILE_PATH} not found. Using default n_cpu=${N_CPU}."
fi

# Modify the script to use the failed tasks list and correct resources
sed -i "s|LIST_FILE=\"\${OPT_DIR}/.files_to_process_task_list.txt\"|LIST_FILE=\"\${OPT_DIR}/.failed_tasks_list.txt\"|" "$NEW_SCRIPT"
sed -i "s/-t 1-N/-t 1-$FAILED_COUNT/" "$NEW_SCRIPT"
sed -i "s/^#\$ -pe shared [0-9][0-9]*/#\$ -pe shared ${N_CPU}/" "$NEW_SCRIPT"

# Make the script executable
chmod +x "$NEW_SCRIPT"

# Delete the failed tasks from the original job
echo "Deleting failed tasks from original job..."
for task_id in $FAILED_TASKS; do
    qdel "${JOB_ID}.${task_id}"
done

# Submit the new job for failed tasks
echo "Submitting new job for failed tasks..."
qsub "$NEW_SCRIPT" "$RUN_DIR"

# Clean up
rm -f /tmp/job_status_$$.txt

echo "Resubmission complete!"
echo "New job submitted for $FAILED_COUNT failed tasks"
echo "You can monitor progress with: qstat -u $USER" 