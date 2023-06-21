#!/bin/bash

# Check if all arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <config_dir> <python_p_arg> <worker_id> <file_range>"
    exit 1
fi

config_dir=$1
python_p_arg=$2
worker_id=$3
file_range=$4

switch_flag=0

# Define the start and end indices for the file range
start_index=${file_range%-*}
end_index=${file_range#*-}

if (( start_index < 0 || start_index > 9 || end_index < 0 || end_index > 9 || start_index > end_index )); then
    echo "Invalid file range. The range should be between 0 and 9 (inclusive) and start_index should be less than or equal to end_index."
    exit 1
fi

echo "Processing files in the range ${start_index}-${end_index}..."

# Iterate over the range of indices
for ((i=start_index; i<=end_index; i++)); do
    file="${config_dir}/${i}-*.yaml"

    # Get the name of the file
    file_name=$(ls $file)

    # Check if the file exists
    if [ ! -f "$file_name" ]; then
        echo "File $file not found."
        continue
    fi

    if [ $switch_flag -eq 0 ]; then
        current_worker_id=$worker_id
        switch_flag=1
    else
        current_worker_id=$(($worker_id + 5))
        switch_flag=0
    fi

    python train_crowd.py -c "$file_name" -i 1000 -e ../builds/crowd-vR2a/crowd.x86_64 -p "$python_p_arg" -w "$current_worker_id" -u
done
