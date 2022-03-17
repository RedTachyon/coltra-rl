COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts


srun python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 0 -n 1 -o test &
sleep 5
srun python $PROJECT/optimize/optuna_crowd.py -e $COLTRA/builds/crowd-v5/crowd.x86_64 -w 10 -n 1 -o test &
wait
