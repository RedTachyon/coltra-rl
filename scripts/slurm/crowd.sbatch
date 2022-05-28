#!/bin/bash
#SBATCH --job-name=crowd-onejob
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/JobGPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobGPU%j.err # error file (%j = job ID)
#SBATCH --time=05:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH -A ibc@gpu

# Environment variables to import: (all str names)
# OBSERVER
# DYNAMICS
# MODEL

OBSERVER="Egocentric"
DYNAMICS="PolarVelocity"
MODEL="relation"
PROJECTNAME="DCSRL-big-debug"
EXTRA_CONFIG='{"environment.mode": "Circle", "num_agents": 12}'

module purge
module load pytorch-gpu/py3/1.8.1

source $HOME/.bashrc
conda activate coltra

wandb offline

srun bash run_training.sh $OBSERVER $DYNAMICS $MODEL $PROJECTNAME $EXTRA_CONFIG
