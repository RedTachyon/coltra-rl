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


module purge
module load pytorch-gpu/py3/1.8.1


COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts
no_proxy=admin,localhost,127.0.0.1

source $HOME/.bashrc
conda activate coltra

wandb offline

set -x
unset HTTP_PROXY
unset HTTPS_PROXY

srun python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 0 -n 1 -o test
