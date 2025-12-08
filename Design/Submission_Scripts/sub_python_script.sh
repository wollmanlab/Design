#!/bin/bash
#$ -cwd
#$ -j y
#$ -o job_$JOB_NAME.$JOB_ID.out
#$ -l h_rt=1:00:00,h_data=64G
#$ -pe shared 1

# --- Script Arguments ---
# The first argument to this script will be the Python script to run.
PYTHON_SCRIPT_TO_RUN="$1"

# Check if a Python script name was provided
if [ -z "$PYTHON_SCRIPT_TO_RUN" ]; then
  echo "Error: No Python script specified."
  echo "Usage: qsub sub_python_script.sh <python_script_name_or_path> [python_script_args...]"
  exit 1
fi

# Shift the first argument, so $@ now contains only the arguments for the Python script
shift

# --- Environment Setup ---
echo "------------------------------------------------------------------------"
echo "Job $JOB_ID started on: $(hostname -s)"
echo "Job $JOB_ID started on: $(date)"
echo "Python script: $PYTHON_SCRIPT_TO_RUN"
echo "Python script arguments: $@"
echo "------------------------------------------------------------------------"

# Load required modules and activate conda environment
. /u/local/Modules/default/init/modules.sh
module load conda
conda activate designer_3.12

# --- Execute the Python Script ---
echo "Executing Python script..."
echo "/usr/bin/time -v python $PYTHON_SCRIPT_TO_RUN $@"
/usr/bin/time -v python "$PYTHON_SCRIPT_TO_RUN" "$@"

# --- Job Completion ---
EXIT_STATUS=$?
echo "------------------------------------------------------------------------"
echo "Job $JOB_ID finished with exit status: $EXIT_STATUS"
echo "Job $JOB_ID ended on: $(hostname -s)"
echo "Job $JOB_ID ended on: $(date)"
echo "------------------------------------------------------------------------"

exit $EXIT_STATUS