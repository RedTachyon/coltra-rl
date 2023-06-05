#!/bin/bash

# Check if all arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <config_dir> <python_p_arg> <worker_id>"
    exit 1
fi

config_dir=$1
python_p_arg=$2
worker_id=$3

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

	python train_crowd.py -c "$file" -i 1000 -e ../builds/crowd-vR2a/crowd.x86_64 -p "$python_p_arg" -w "$current_worker_id" -u
done
