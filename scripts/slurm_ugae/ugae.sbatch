#!/bin/bash
#SBATCH --job-name=ugae-onejob
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/JobGPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobGPU%j.err # error file (%j = job ID)
#SBATCH --time=05:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --hint=nomultithread
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH -A ibc@v100

# Environment variables to import: (all str names)
# OBSERVER
# DYNAMICS
# MODEL

# OBSERVER="Egocentric"
# DYNAMICS="PolarAcceleration"
# MODEL="relation"
# PROJECTNAME="DCSRL-big-debug"
# EXTRA_CONFIG='{"environment.mode":"Circle","environment.num_agents":12}' # CAREFUL - no spaces

echo "PARAMETERS:"
echo $ENV_ID
echo $PROJECTNAME
echo $EXTRA_CONFIG
echo $NUM_RUNS

module purge

source $HOME/.bashrc
conda activate coltra

wandb offline

srun bash run_training.sh $ENV_ID $PROJECTNAME $EXTRA_CONFIG $NUM_RUNS
