#!/bin/bash
#SBATCH --job-name=crowd-mig
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/JobGPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobGPU%j.err # error file (%j = job ID)
#SBATCH --time=20:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=60
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH -A axs@v100

# Environment variables to import: (all str names)


echo "PARAMETERS:"
echo $CONFIG
echo $PROJECTNAME
echo $NUM_RUNS

module purge

source $HOME/.bashrc
conda activate coltra-rl

wandb offline

srun bash run_training.sh $CONFIG $PROJECTNAME $NUM_RUNS
