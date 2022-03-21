#!/usr/bin/bash

LASTID1=$(sbatch crowd.sbatch)
LASTID1=$(echo $LASTID1 | awk 'NF{ print $NF }')

echo "Launched $LASTID1"
LASTID2=$(sbatch crowd.sbatch)
LASTID2=$(echo $LASTID2 | awk 'NF{ print $NF }')

echo "Launched $LASTID2"

for i in {2..15}
do
    LASTID1=$(sbatch --dependency=afterok:$LASTID1 crowd.sbatch)
    LASTID1=$(echo $LASTID1 | awk 'NF{ print $NF }')
    echo "$i Queued $LASTID1"

    LASTID2=$(sbatch --dependency=afterok:$LASTID2 crowd.sbatch)
    LASTID2=$(echo $LASTID2 | awk 'NF{ print $NF }')

    echo "$i Queued $LASTID2"
done
