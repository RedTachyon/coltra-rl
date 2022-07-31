#!/usr/bin/bash
CONFIG=${1:-configs/base.yaml}
OPTUNA=${2:-optuna}
WANDB=${3:-jeanzay-sweep}
TOTAL=${4:-15}

echo "CONFIG: $CONFIG"
echo "OPTUNA: $OPTUNA"
echo "WANDB: $WANDB"

python optuna_setup.py -n $OPTUNA

LASTID1=$(sbatch --export=ALL,CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB optuna_ugae.sbatch)
LASTID1=$(echo $LASTID1 | awk 'NF{ print $NF }')

echo "Launched $LASTID1"
LASTID2=$(sbatch --export=ALL,CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB optuna_ugae.sbatch)
LASTID2=$(echo $LASTID2 | awk 'NF{ print $NF }')

echo "Launched $LASTID2"

for i in $(seq 2 $TOTAL)
do
    LASTID1=$(sbatch --dependency=afterok:$LASTID1 --export=ALL,CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB optuna_ugae.sbatch)
    LASTID1=$(echo $LASTID1 | awk 'NF{ print $NF }')
    echo "$i Queued $LASTID1"

    LASTID2=$(sbatch --dependency=afterok:$LASTID2 --export=ALL,CONFIG=$CONFIG,OPTUNA=$OPTUNA,WANDB=$WANDB optuna_ugae.sbatch)
    LASTID2=$(echo $LASTID2 | awk 'NF{ print $NF }')

    echo "$i Queued $LASTID2"
done
