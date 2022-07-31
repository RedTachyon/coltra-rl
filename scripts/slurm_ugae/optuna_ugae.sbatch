#!/bin/bash
#SBATCH --job-name=ugae-onejob
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_cpu-t3
#SBATCH --output=logs/JobCPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobCPU%j.err # error file (%j = job ID)
#SBATCH --time=10:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --hint=nomultithread
##SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
##SBATCH --exclusive
#SBATCH -A ibc@cpu

# Environment variables imported:
# CONFIG -- path to the config yaml
# OPTUNA -- name of the optuna database
# WANDB -- name of the wandb project

module purge
module load pytorch-gpu/py3/1.8.1

source $HOME/.bashrc
conda activate coltra

wandb offline

srun bash run_optuna_ugae.sh $CONFIG $OPTUNA $WANDB
