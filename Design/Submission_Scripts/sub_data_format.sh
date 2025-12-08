#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o job_data_format.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=1:00:00,h_data=64G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1

# Get the current username
CURRENT_USER=$(whoami)
echo "User identified as ${CURRENT_USER}"

# Get first letter of username for path
USER_FIRST_LETTER=${CURRENT_USER:0:1}

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

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load conda

# Initialize conda for bash shell
source "${CONDA_PATH}"
conda activate designer_3.12

# Verify conda environment
echo "Using conda environment: $(which python)"
echo "Python version: $(python --version)"

## substitute the command to run your code
## in the two lines below:
echo "/usr/bin/time -v python ${CODE_DIR}/Data_Format/data_format.py"
/usr/bin/time -v python "${CODE_DIR}/Data_Format/data_format.py"

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "