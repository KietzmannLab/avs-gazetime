#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=32
#SBATCH -p klab-cpu
#SBATCH --job-name=resmem
#SBATCH --error=error_resmem_%A_%a.err
#SBATCH --output=output_resmem_%A_%a.out
#SBATCH --array=1-5%5  # Run all 5 subjects in parallel
#SBATCH --requeue

echo "Running memorability score computation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

export NCCL_SOCKET_IFNAME=lo

# Load required modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate /share/klab/psulewski/envs/thingsvision

# Set environment variables
export SUBJECT_ID_GAZETIME=$SLURM_ARRAY_TASK_ID
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print environment info
echo "Subject ID: $SUBJECT_ID_GAZETIME"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the memorability computation script
gazetime_path="/home/student/p/psulewski/AVS-GazeTime/avs_gazetime"
python "${gazetime_path}/memorability/compute_memorability_scores.py"

echo "Job completed at: $(date)"