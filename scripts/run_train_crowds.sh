#!/bin/bash

# Check if a worker_id argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <worker_id>"
    exit 1
fi

worker_id=$1
config_dir="./vR0-configs/rewards"

switch_flag=0

# Iterate over each file in the specified directory
for file in "$config_dir"/*; do
	if [ $switch_flag -eq 0 ]; then
		current_worker_id=$worker_id
		switch_flag=1
	else
		current_worker_id=$(($worker_id + 5))
		switch_flag=0
	fi

	python train_crowd.py -c "$file" -i 1000 -e ../builds/crowd-vR1a/crowd.x86_64 -p crowd-reward-choke -w "$current_worker_id" -u
done
