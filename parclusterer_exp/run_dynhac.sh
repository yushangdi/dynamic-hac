# Copyright 2024 The Approximate Hac Experiments Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

export EXP_ROOT=$(pwd) # set EXP_ROOT to current directory
export PARLAY_NUM_THREADS=1

datasets=("mnist" "aloi" "ilsvrc_small") #mnist   "imagenet" 
num_batches=(70000 108000 50000) #70000     100000 
weights=(0.0001 0.01 0.01)
k=50

first_batch="0.99"
epsilons=("0.1" "1" "0")

bazel build //parclusterer_exp/benchmark:cut_dendrogram
bazel build //parclusterer_exp/benchmark:parhac_main

for i in {0..2}; do
  dataset=${datasets[i]}
  num_batch=${num_batches[i]}
  weight=${weights[i]}
  store_batch_size=$((num_batch / 1000))

  input_data="$EXP_ROOT/data/${dataset}/${dataset}.scale.permuted.fvecs"
  ground_truth="$EXP_ROOT/data/${dataset}/${dataset}.scale.permuted_label.bin"
  clustering="$EXP_ROOT/results/dynamic_hac_tail/${dataset}"
  output_file="$EXP_ROOT/results/results_dyn/tail_${dataset}"
  # output_knn="benchmark/result/knn/${dataset}/knn_${dataset}"
  mkdir -p ${output_file}

  for epsilon in ${epsilons[@]}; do
    command="python3 parclusterer_exp/benchmark/run_experiment.py \
    --input_data=${input_data} \
    --ground_truth=${ground_truth} \
    --clustering=${clustering} \
    --output_file=${output_file} \
    --num_batch=${num_batch} \
    --store_batch_size=${store_batch_size} \
    --k=${k} \
    --weight=${weight} --epsilon=${epsilon}\
    --method=dynamic_hac \
    --first_batch_ratio=${first_batch}"


    echo ${command} # Print the command for verification
    eval ${command}  # Execute the command
    echo 
  done
done
