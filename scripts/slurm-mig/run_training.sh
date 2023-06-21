WORK=/gpfswork/rech/axs/utu66tc
COLTRA=$WORK/projects/coltra-rl
PROJECT=$COLTRA/scripts
CURDIR=$PROJECT/slurm-mig
export WANDB_CACHE_DIR=$WORK/wandb-cache

CONFIG=${1:-invalid}
PROJECTNAME=${2:-debug}
NUM_RUNS=${3:-invalid}

#for i in {0..$NUM_RUNS}
for ((i=0; i<NUM_RUNS; i++))
do
    CUDA_VISIBLE_DEVICES=$((i/2)) python $CURDIR/train_crowd_slurm.py -c "$CONFIG" -i 1000 -e $COLTRA/builds/crowd-vR2a/crowd.x86_64 -p "$PROJECTNAME" -w $((i*10)) &
    sleep 1
done
wait

