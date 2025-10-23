#!/bin/bash
# ============================================================================
# PAC Analysis Array Job
# ============================================================================
# This script runs phase-amplitude coupling analysis as a SLURM array job.
#
# SUBMISSION INSTRUCTIONS:
#
# 1. Submit a single subject with 200 chunks (default):
#    sbatch --export=SUBJECT_ID=1 pac_array_job.sh
#
# 2. Submit all 5 subjects at once with default 200 chunks:
#    for subj in {1..5}; do
#      sbatch --export=SUBJECT_ID=$subj pac_array_job.sh
#    done
#
# 3. Customize number of chunks for memory management:
#    sbatch --export=SUBJECT_ID=1,CHUNKS=100 --array=0-99 pac_array_job.sh
#
# MEMORY CONSIDERATIONS:
# - More chunks = less memory per job (but more jobs total)
# - Typical source space (STC): ~8000 channels total
# - For 200G memory limit: 30-100 chunks recommended (80-267 channels/chunk)
# - Each chunk pre-filters all its channels (theta + gamma) before PAC computation
# - Memory usage per chunk ≈ 3 × (n_epochs × n_channels_in_chunk × n_times × 8 bytes)
#
# NOTE: The SLURM array indices (0-N) must match the number of chunks minus 1
# ============================================================================
#SBATCH --time=1:10:00 # Run time
#SBATCH --nodes=1
#SBATCH --mem=60G # Memory per array task
#SBATCH -c 10 # Cores per array task
#SBATCH -p klab-cpu
#SBATCH --job-name=gude
#SBATCH --error=error_pac_%A_%a.err
#SBATCH --output=output_pac_%A_%a.out
#SBATCH --array=0-199 # Default: 200 chunks (indices 0-199)
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Parameters
CH_TYPE="stc"
gazetime_path="/home/student/p/psulewski/avs-gazetime/avs_gazetime"
analysis_path="${gazetime_path}/pac/pac_analysis.py"

# Get subject from command line or default to 1
SUBJECT=${SUBJECT_ID:-1}
export SUBJECT_ID_GAZETIME=$SUBJECT
export CH_TYPE_GAZETIME=$CH_TYPE

# Get total number of chunks - default to 200
TOTAL_CHUNKS=${CHUNKS:-200}

echo "Running subject $SUBJECT_ID_GAZETIME with $CH_TYPE_GAZETIME channels"
echo "Processing chunk ${SLURM_ARRAY_TASK_ID} of ${TOTAL_CHUNKS}"

# Validate array task ID is within range
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_CHUNKS} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) >= TOTAL_CHUNKS (${TOTAL_CHUNKS})"
    echo "Submit with: --array=0-$((TOTAL_CHUNKS - 1))"
    exit 1
fi

# Compose the command
chunk_str="${SLURM_ARRAY_TASK_ID}_${TOTAL_CHUNKS}"
command="python $analysis_path $chunk_str"

echo "Executing: $command"
eval $command