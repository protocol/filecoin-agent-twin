#!/bin/bash

# This script is used to run agentfil experiments in parallel.
# It requires GNU parallel to be installed.
. `which env_parallel.bash`

num_jobs=4
root_output_dir="$HOME/agentfil/exp"
mkdir -p $root_output_dir

start_date="2023-02-01"
end_date="2026-02-01"

experiment_names=(
    "nagents-5_type-greedy_optimism-3" \
    "nagents-5_type-greedy_optimism-4" \
    "nagents-25_type-greedy_optimism-3" \
    "nagents-25_type-greedy_optimism-4" \
    "nagents-125_type-greedy_optimism-3" \
    "nagents-125_type-greedy_optimism-4" \
)
experiment_output_dirs=(
    "$root_output_dir/nagents-5_type-greedy_optimism-3" \
    "$root_output_dir/nagents-5_type-greedy_optimism-4" \
    "$root_output_dir/nagents-25_type-greedy_optimism-3" \
    "$root_output_dir/nagents-25_type-greedy_optimism-4" \
    "$root_output_dir/nagents-125_type-greedy_optimism-3" \
    "$root_output_dir/nagents-125_type-greedy_optimism-4" \
)

experiment_runner() {
    experiment_id=$1
    echo "Experiment id: $experiment_id"
    name=${experiment_names[$experiment_id]}
    echo "Running experiment $name"

    output_dir=${experiment_output_dirs[$experiment_id]}
    mkdir -p $output_dir
    
    python3 run_experiment.py \
        --start-date $start_date \
        --end-date $end_date \
        --experiment-name $name \
        --output-dir $output_dir
}
export start_date end_date experiment_names experiment_output_dirs
export -f experiment_runner 

num_experiments=${#experiment_names[@]}
run_indices=($(seq 0 1 $((num_experiments-1))))
#echo $num_experiments ${run_indices[@]}
env_parallel -j $num_jobs experiment_runner ::: ${run_indices[@]}
