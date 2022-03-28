#!/usr/bin/bash
CONFIG=${1:-configs/base.yaml}
OPTUNA=${2:-optuna}
WANDB=${3:-jeanzay-sweep}

echo "CONFIG: $CONFIG"
echo "OPTUNA: $OPTUNA"
echo "WANDB: $WANDB"

python optuna_setup.py -n $OPTUNA

LASTID1=$(sbatch --export=CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB crowd.sbatch)
LASTID1=$(echo $LASTID1 | awk 'NF{ print $NF }')

echo "Launched $LASTID1"
LASTID2=$(sbatch --export=CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB crowd.sbatch)
LASTID2=$(echo $LASTID2 | awk 'NF{ print $NF }')

echo "Launched $LASTID2"

for i in {2..15}
do
    LASTID1=$(sbatch --dependency=afterok:$LASTID1 --export=CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB crowd.sbatch)
    LASTID1=$(echo $LASTID1 | awk 'NF{ print $NF }')
    echo "$i Queued $LASTID1"

    LASTID2=$(sbatch --dependency=afterok:$LASTID2 --export=CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB crowd.sbatch)
    LASTID2=$(echo $LASTID2 | awk 'NF{ print $NF }')

    echo "$i Queued $LASTID2"
done
