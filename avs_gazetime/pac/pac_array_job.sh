#!/bin/bash
# ============================================================================
# PAC Analysis Array Job
# ============================================================================
# This script runs phase-amplitude coupling analysis as a SLURM array job.
#
# SUBMISSION INSTRUCTIONS:
#
# 1. Submit a single subject with 30 chunks:
#    sbatch --export=SUBJECT_ID=1 --array=0-29 pac_array_job.sh
#
# 2. Submit all 5 subjects at once:
#    for subj in {1..5}; do
#        sbatch --export=SUBJECT_ID=$subj,CHUNKS=100 --array=0-99 pac_array_job.sh
#    done
#
# 3. Customize number of chunks:
#    sbatch --export=SUBJECT_ID=1,CHUNKS=100 --array=0-99 pac_array_job.sh
#
# NOTE: The SLURM array indices (0-N) must match the number of chunks minus 1
# ============================================================================

#SBATCH --time=12:00:00 # Run time
#SBATCH --nodes=1
#SBATCH --mem=100G # Memory per array task
#SBATCH -c 20 # Cores per array task
#SBATCH -p klab-cpu
#SBATCH --job-name=wtpac
#SBATCH --error=error_pac_%A_%a.err
#SBATCH --output=output_pac_%A_%a.out
#SBATCH --array=0-29 # This will be overridden by command line
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo
cd /home/student/p/psulewski/avs-encoding

# Load required modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Parameters
CH_TYPE="stc"
gazetime_path="/home/student/p/psulewski/AVS-GazeTime/avs_gazetime"
analysis_path="${gazetime_path}/pac/pac_analysis.py"

# Get subject from command line or default to 1
SUBJECT=${SUBJECT_ID:-1}
export SUBJECT_ID_GAZETIME=$SUBJECT
export CH_TYPE_GAZETIME=$CH_TYPE

# Get total number of chunks from environment or default to array size + 1
TOTAL_CHUNKS=${CHUNKS:-$(( ${SLURM_ARRAY_TASK_COUNT:-30} + 1 ))}

echo "Running subject $SUBJECT_ID_GAZETIME with $CH_TYPE_GAZETIME channels"
echo "Processing chunk ${SLURM_ARRAY_TASK_ID} of ${TOTAL_CHUNKS}"

# Compose the command
chunk_str="${SLURM_ARRAY_TASK_ID}_${TOTAL_CHUNKS}"
command="python $analysis_path $chunk_str"
echo "Executing: $command"
eval $command