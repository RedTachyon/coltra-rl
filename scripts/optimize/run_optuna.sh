COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts

OPTUNA=${1:-optuna}

python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 0 -n 1 -o $OPTUNA &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 10 -n 1 -o $OPTUNA &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 20 -n 1 -o $OPTUNA &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 30 -n 1 -o $OPTUNA &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 40 -n 1 -o $OPTUNA &
wait
