#!/bin/bash
#SBATCH --time=48:00:00 # Run time
#SBATCH --nodes 1 
#SBATCH --mem 700G
#SBATCH -c 100
#SBATCH -p klab-cpu
#SBATCH --job-name hammertime
#SBATCH --error=error.o%j
#SBATCH --output=output.o%j
#SBATCH --requeue

echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo

cd /home/student/p/psulewski/avs-encoding
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate avs

# wich analyses to run?
# which analyses to run?
theta_mem=true

#ch_types="mag "
ch_types="stc "
n_chunks=30
# ch_types="grad "
# n_chunks=1

gazetime_path="/home/student/p/psulewski/AVS-GazeTime/avs_gazetime"

# Check if a subject ID was provided as an argument
if [ -z "$1" ]; then
    subjects=$(seq 1 5)
else
    subjects=$1
fi

# Loop over subjects
for i in $subjects
do
    # Loop over chtypes "grad" and "mag"
    for ch_type in $ch_types
    do  
        export SUBJECT_ID_GAZETIME=$i
        export CH_TYPE_GAZETIME=$ch_type
    
        
        echo "Running subject $SUBJECT_ID_GAZETIME with $CH_TYPE_GAZETIME channels and $SENSOR_SELECTION_GAZETIME sensor selection"

        # Check if the analysis is to be run
        if [ "$theta_mem" = true ] ; then
            analysis_path="${gazetime_path}/pac/pac_analysis.py"
            # if ch type stc we have to add another chunking system command. It works as follows:
            # 1_4 = the first of 4 chunks of channels, 3_4 the third of 4 chunks of channels. 4_6 the fourth of 6 chunks of channels....
            # iterate over the chunks
            if [ "$ch_type" = "stc" ]; then
                for chunk in $(seq 0 $n_chunks)
                do
                    # compose the command
                    echo $chunk
                    echo $n_chunks
                    chunk_str="$chunk"_"$n_chunks"
                    echo $chunk_str
                    command="python $analysis_path $chunk_str"
                    echo $command
                    eval $command
                
                done
            else
                command="python $analysis_path"
                echo $command
                eval $command 
            fi
        fi
    done
done