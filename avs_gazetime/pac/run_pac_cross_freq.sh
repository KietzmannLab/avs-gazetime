#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH -c 30
#SBATCH -p klab-cpu
#SBATCH --job-name=tarrifs
#SBATCH --error=pac_roi_%A_%a.err
#SBATCH --output=pac_roi_%A_%a.out
#SBATCH --array=0-29 # Default array size (6 ROIs × 5 subjects = 30 jobs)
#SBATCH --requeue
echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo
# Load required modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs
# Configuration
gazetime_path="/home/student/p/psulewski/AVS-GazeTime/avs_gazetime"
ch_types="stc"
# Default settings
OVERWRITE=false
# Process flags (use -- arguments)
while [[ $# -gt 0 ]]; do
case $1 in
--overwrite)
OVERWRITE=true
shift
 ;;
--rois)
shift
ROI_NAMES=($1)
shift
 ;;
--roidefs)
shift
ROI_DEFINITIONS=($1)
shift
 ;;
--subjects)
shift
SUBJECTS=($1)
shift
 ;;
 *)
# Legacy positional argument handling
if [ -z "${ROI_NAMES+x}" ]; then
# First positional argument is ROI_NAMES
ROI_NAMES=($1)
shift
else
# Remaining positional arguments are ROI_DEFINITIONS
ROI_DEFINITIONS+=("$1")
shift
fi
 ;;
esac
done
# ROI definitions - can be customized or passed as argument
if [ -z "${ROI_NAMES+x}" ]; then
# Default ROIs if none provided
ROI_NAMES=("HC" "FEF" "early" "dlPFC" "OFC" "infFC")
ROI_DEFINITIONS=("H" "FEF" "early" "8C,8Av,i6-8,s6-8,SFL,8BL,9p,9a,8Ad,p9-46v,a9-46v,46,9-46d" "47s,47m,a47r,11l,13l,a10p,p10p,10pp,10d,OFC,pOFC" "45,IFJp,IFJa,IFSp,IFSa,47l,p47r")
fi
# Subject range - default is all 5 subjects
if [ -z "${SUBJECTS+x}" ]; then
SUBJECTS=$(seq 1 5)
fi
NUM_SUBJECTS=$(echo $SUBJECTS | wc -w)
# Calculate total number of jobs (subjects * ROIs)
NUM_ROIS=${#ROI_NAMES[@]}
TOTAL_JOBS=$((NUM_SUBJECTS * NUM_ROIS))
echo "Processing $NUM_SUBJECTS subjects across $NUM_ROIS ROIs (total jobs: $TOTAL_JOBS)"
echo "ROIs to process: ${ROI_NAMES[*]}"
echo "Overwrite existing results: $OVERWRITE"
# Calculate subject and ROI index from array task ID
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-0}
SUBJECT_IDX=$((ARRAY_ID % NUM_SUBJECTS))
ROI_IDX=$((ARRAY_ID / NUM_SUBJECTS))
# Extract subject ID from SUBJECTS array
SUBJECT_ARRAY=($SUBJECTS)
SUBJECT_ID=${SUBJECT_ARRAY[$SUBJECT_IDX]}
# Get ROI name and definition
ROI_NAME=${ROI_NAMES[$ROI_IDX]}
ROI_DEF=${ROI_DEFINITIONS[$ROI_IDX]}
echo "==================================================="
echo "Array task $ARRAY_ID: Processing subject $SUBJECT_ID, ROI $ROI_NAME"
echo "ROI definition: $ROI_DEF"
if [ "$OVERWRITE" = true ]; then
echo "Overwrite mode: Will recompute existing results"
fi
echo "==================================================="
# Set environment variables for the Python script
export SUBJECT_ID_GAZETIME=$SUBJECT_ID
export CH_TYPE_GAZETIME=$ch_types
# Path to the cross-frequency PAC script
analysis_path="${gazetime_path}/pac/cross_frequency_pac.py"
# Build command with optional overwrite flag
command="python $analysis_path $ROI_NAME \"$ROI_DEF\""
if [ "$OVERWRITE" = true ]; then
command="$command --overwrite"
fi
echo "Running command: $command"
eval $command
echo "Completed processing for subject $SUBJECT_ID, ROI $ROI_NAME"
exit 0