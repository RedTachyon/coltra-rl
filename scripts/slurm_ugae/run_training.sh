WORK=/gpfswork/rech/nbk/utu66tc
COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts
export WANDB_CACHE_DIR=$WORK/wandb-cache


#OBSERVER=${1:-invalid}
#DYNAMICS=${2:-invalid}
#MODEL=${3:-invalid}
ENV_NAME=${1:-invalid}
PROJECTNAME=${2:-debug}
EXTRA_CONFIG=${3:-invalid}
NUM_RUNS=${4:-invalid}



#for i in {0..$NUM_RUNS}
for ((i=0; i<NUM_RUNS; i++))
do
    CUDA_VISIBLE_DEVICES=$((i/2)) python $PROJECT/slurm_ugae/train_gym.py -e "$ENV_NAME" -p "$PROJECTNAME" -i 2000 -s $i -norme -tf -ec $EXTRA_CONFIG -tb $WORK &
    sleep 1
done
wait

