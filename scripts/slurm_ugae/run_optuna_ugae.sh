WORK=/gpfswork/rech/nbk/utu66tc
COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts
export WANDB_CACHE_DIR=$WORK/wandb-cache

CONFIG=${1:-invalid}
OPTUNA=${2:-optuna}
WANDB=${3:-invalid}

python $PROJECT/slurm_ugae/optuna_ugae.py -c $CONFIG -n 1 -o $OPTUNA -wp $WANDB #&
#sleep 1
#python $PROJECT/optimize/optuna_ugae.py -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
#sleep 1
#python $PROJECT/optimize/optuna_ugae.py -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
#sleep 1
#python $PROJECT/optimize/optuna_ugae.py -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
#sleep 1
#python $PROJECT/optimize/optuna_ugae.py -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
#wait
