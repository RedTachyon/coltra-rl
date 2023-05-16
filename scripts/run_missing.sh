#!/bin/bash

# Check if a worker_id argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <worker_id>"
    exit 1
fi

worker_id=$1
config_dir="./vR0-configs/rewards"

file_list=("kwiat-speeding.yaml" "kwiat-sq.yaml" "kwiat.yaml" "v-matching.yaml")

# Iterate over each file in the specified directory
for filename in "${file_list[@]}"; do
	file="$config_dir/$filename"
	python train_crowd.py -c "$file" -i 1000 -e ../builds/crowd-vR1a/crowd.x86_64 -p crowd-reward-circ-6 -w "$worker_id" -u
done
