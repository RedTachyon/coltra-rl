#!/bin/bash
#SBATCH --job-name=crowd-echo
##SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/JobGPU%j.out # output file (%j = job ID)
#SBATCH --error=logs/JobGPU%j.err # error file (%j = job ID)
#SBATCH --time=00:01:00 # maximum allocation time "(HH:MM:SS)"
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
echo $OBSERVER
echo $DYNAMICS
echo $MODEL
echo $PROJECTNAME
echo $EXTRA_CONFIG
