#!/bin/bash
#SBATCH --time=12:00:00           # Run time
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=40
#SBATCH -p klab-cpu
#SBATCH --job-name=dynamics
#SBATCH --array=1-5               # Run 5 jobs in parallel (one per subject)
#SBATCH --error=error_dynamics_%A_%a.err   # %A = job ID, %a = array index
#SBATCH --output=output_dynamics_%A_%a.out
#SBATCH --requeue

echo "Running in shell: $SHELL"
echo "Processing subject $SLURM_ARRAY_TASK_ID as part of array job $SLURM_ARRAY_JOB_ID"

export NCCL_SOCKET_IFNAME=lo

cd /home/student/p/psulewski/avs-encoding
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# Channel types to process
ch_types="stc"

# Base paths
gazetime_path="/home/student/p/psulewski/AVS-GazeTime/avs_gazetime"

# Get subject ID from array task ID
subject=$SLURM_ARRAY_TASK_ID

# Loop over channel types
for ch_type in $ch_types
do
    # Set environment variables for this run
    export SUBJECT_ID_GAZETIME=$subject
    export CH_TYPE_GAZETIME=$ch_type
    
    echo "Running subject $SUBJECT_ID_GAZETIME with $CH_TYPE_GAZETIME channels"
    
    # Run the dynamics analysis
    python ${gazetime_path}/dynamics/dynamics_analysis.py
done

echo "Job completed: subject $subject processed"