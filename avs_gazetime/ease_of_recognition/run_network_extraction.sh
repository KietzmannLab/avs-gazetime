#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH -c 40
#SBATCH -p klab-cpu
#SBATCH --job-name=help
#SBATCH --error=nn_activations_%A_%a.err
#SBATCH --output=nn_activations_%A_%a.out
#SBATCH --array=1-30%4
#SBATCH --requeue

echo "Running in shell: $SHELL"
export NCCL_SOCKET_IFNAME=lo


# Load required modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate  /share/klab/psulewski/envs/thingsvision  # Assuming avs is the correct environment

# Get the subject ID from the SLURM array task ID
SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

# Get crop size from command line argument (default: 112)
CROP_SIZE=${1:-112}

# Validate crop size (only allow 100, 112, or 164)
if [[ "$CROP_SIZE" != "100" && "$CROP_SIZE" != "112" && "$CROP_SIZE" != "164" ]]; then
    echo "Error: Invalid crop size. Please use 100, 112, or 164 only."
    exit 1
fi

# Print information about the job
echo "Starting activation extraction job for subject ${SUBJECT_ID} with crop size ${CROP_SIZE}px"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Run the Python script with arguments
python /home/student/p/psulewski/AVS-GazeTime/avs_gazetime/ease-of-recognition/get_netwok_activations.py --subject_id ${SUBJECT_ID} --crop_size ${CROP_SIZE} --verbose

# Print job completion information
echo "Job completed at: $(date)"