#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=450G
#SBATCH --cpus-per-task=20

#SBATCH -p klab-cpu
#SBATCH --job-name=memdecode
#SBATCH --error=error_memorability_decoder_%A_%a.err
#SBATCH --output=output_memorability_decoder_%A_%a.out
#SBATCH --array=1-5  # Run for subjects 1-5
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Channel types to process
ch_types="mag"

# Base paths
gazetime_path="/home/student/p/psulewski/avs-gazetime/avs_gazetime"

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

echo "==================================================="
echo "Running memorability decoding for subject $subject"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "==================================================="

# Loop over channel types
for ch_type in $ch_types
do
    # Set environment variables for this run
    export SUBJECT_ID_GAZETIME=$subject
    export CH_TYPE_GAZETIME=$ch_type
    
    echo "Processing subject $SUBJECT_ID_GAZETIME with $CH_TYPE_GAZETIME channels"
    
    # Run the memorability decoder
    python ${gazetime_path}/decoding/memorability_decoder.py
    
    echo "Completed $ch_type for subject $subject"
done

echo "Subject $subject processing complete"