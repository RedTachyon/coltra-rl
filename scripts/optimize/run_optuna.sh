COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts


python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 0 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 10 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 20 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 30 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 40 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 50 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 60 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 70 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 80 -n 1 -o test &
sleep 1
python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 90 -n 1 -o test &
sleep 1
wait
