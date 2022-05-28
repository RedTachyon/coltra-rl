WORK=/gpfswork/rech/nbk/utu66tc
COLTRA=$WORK/coltra-rl
PROJECT=$COLTRA/scripts
export WANDB_CACHE_DIR=$WORK/wandb-cache

OBSERVER=${1:-invalid}
DYNAMICS=${2:-invalid}
MODEL=${3:-invalid}
PROJECTNAME=${4:-debug}
EXTRA_CONFIG=${5:-invalid}

i=0

CUDA_VISIBLE_DEVICES=$((i/3)) python $PROJECT/slurm/train_crowd.py -e $COLTRA/builds/crowd-v6a/crowd.86_64 -w $((i*10)) -o "$OBSERVER" -d "$DYNAMICS" -m "$MODEL" -p "$PROJECTNAME" -ec $EXTRA_CONFIG &
