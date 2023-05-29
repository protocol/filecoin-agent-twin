#!/bin/bash

# This script is used to run agentfil experiments in parallel.
# It requires GNU parallel to be installed.
. `which env_parallel.bash`

run_file=${1:-"None"}  # input filename which contains experiment configs to run
auth_config=${2:-"None"} # JSON file containing bearer token for data downloads
num_jobs=${3:-7}
start_date=${4:-"2023-05-20"}
end_date=${5:-"2033-12-31"}
OVERWRITE_EXISTING=${5:-0} # if 1, reruns experiments even if output already exists
                           # if 0, skips experiments if output already exists
root_output_dir=${6:-"$HOME/agentfil/exp"}
mkdir -p $root_output_dir

# read input file into array
if [ "$run_file" = "None" ]; then
  echo "First argument to script should be the run file!"
  return -1
fi
declare -a experiment_names
let i=0
while IFS=$'\n' read -r line_data; do
    l="${line_data}"
    if [[ ${l::1} == "#" || "$l" == "" ]]; then
        continue
    fi
    experiment_names[i]="$l"
    ((++i))
done < $run_file

experiment_runner() {
    experiment_id=$1
    echo "Experiment id: $experiment_id"
    name=${experiment_names[$experiment_id]}
    echo "Running experiment: $name"

    output_dir="$root_output_dir/$name"
    if [ $OVERWRITE_EXISTING -eq 0 ] && [ -f "$output_dir/filecoin_df.csv" ]; then
        echo "Skipping experiment $name because output already exists"
        return
    fi
    mkdir -p $output_dir
    
    python3 run_experiment.py \
        --auth-config $auth_config \
        --experiment-name $name \
        --output-dir $output_dir \
        --start-date $start_date \
        --end-date $end_date >> $output_dir/log.txt 2>&1
    echo "Finished experiment $name" >> $output_dir/log.txt
}
export start_date end_date experiment_names experiment_output_dirs
export -f experiment_runner 

num_experiments=${#experiment_names[@]}
run_indices=($(seq 0 1 $((num_experiments-1))))

env_parallel -j $num_jobs experiment_runner ::: ${run_indices[@]}
