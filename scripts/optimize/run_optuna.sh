COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts

CONFIG=${1:-config/base.yaml}
OPTUNA=${2:-optuna}
WANDB=${3:-jeanzay-sweep}

python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 0 -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 10 -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 20 -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 30 -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 40 -n 1 -o $OPTUNA -c $CONFIG -wp $WANDB &
wait
