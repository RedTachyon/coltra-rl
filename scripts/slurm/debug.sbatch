#!/bin/bash
#SBATCH --job-name=crowd-onejob
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/JobGPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobGPU%j.err # error file (%j = job ID)
#SBATCH --time=00:01:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH -A ibc@v100

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

srun bash run_debug.sh $OBSERVER $DYNAMICS $MODEL $PROJECTNAME $EXTRA_CONFIG
