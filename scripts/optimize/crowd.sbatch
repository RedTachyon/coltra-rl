#!/bin/bash
#SBATCH --job-name=crowd-onejob
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/JobGPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobGPU%j.err # error file (%j = job ID)
#SBATCH --time=03:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
##SBATCH --exclusive
#SBATCH -A nbk@gpu

# Environment variables imported:
# CONFIG -- path to the config yaml
# OPTUNA -- name of the optuna database
# WANDB -- name of the wandb project

module purge
module load pytorch-gpu/py3/1.8.1



no_proxy=admin,localhost,127.0.0.1

source $HOME/.bashrc
conda activate coltra

wandb offline

set -x
unset HTTP_PROXY
unset HTTPS_PROXY

srun bash run_optuna.sh $CONFIG $OPTUNA $WANDB
