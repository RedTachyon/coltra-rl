#!/usr/bin/bash

LASTID1=$(sbatch crowd.sbatch)
echo "Launched $LASTID1"
LASTID2=$(sbatch crowd.sbatch)
echo "Launched $LASTID2"

for i in {2..15}
do
    LASTID1=$(sbatch --dependency=afterok:$LASTID1 crowd.sbatch)
    echo "$i Queued $LASTID1"

    LASTID2=$(sbatch --dependency=afterok:$LASTID2 crowd.sbatch)
    echo "$i Queued $LASTID2"
done
