CUDA_VISIBLE_DEVICES=$1 python optuna_crowd.py -e ../builds/crowd-v5/crowd.x86_64 -w $2 -n 25 -o optuna
