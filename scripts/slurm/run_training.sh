WORK=/gpfswork/rech/nbk/utu66tc
COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts
export WANDB_CACHE_DIR=$WORK/wandb-cache

OBSERVER=${1:-invalid}
DYNAMICS=${2:-invalid}
MODEL=${3:-invalid}
PROJECTNAME=${4:-debug}
EXTRA_CONFIG=${5:-invalid}
NUM_RUNS=${6:-invalid}

#for i in {0..$NUM_RUNS}
for ((i=0; i<NUM_RUNS; i++))
do
    CUDA_VISIBLE_DEVICES=$((i/2)) python $PROJECT/slurm/train_crowd_slurm.py -e $COLTRA/builds/crowd-v6a/crowd.x86_64 -w $((i*10)) -o "$OBSERVER" -d "$DYNAMICS" -m "$MODEL" -p "$PROJECTNAME" -ec $EXTRA_CONFIG -tb $WORK &
    sleep 1
done
wait

