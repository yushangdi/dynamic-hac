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

# use_output_knn="False"
# mnist
weight=0.0001
# aloi, "ilsvrc_small"
# weight=0.01

datasets=("mnist") #mnist "aloi"    "imagenet" "ilsvrc_small"
num_batches=(70000) #70000 108000    100000 50000

first_batch="0.99"
epsilon="0.1"


for i in {0..0}; do
  dataset=${datasets[i]}
  num_batch=${num_batches[i]}
  store_batch_size=$((num_batch / 1000))

  input_data="benchmark/${dataset}/${dataset}.scale.permuted.fvecs"
  ground_truth="benchmark/${dataset}/${dataset}.scale.permuted_label.csv"
  clustering="benchmark/result/dynamic_hac_tail/${dataset}"
  output_file="$EXP_ROOT/benchmark/results_dyn/tail_${dataset}"
  # output_knn="benchmark/result/knn/${dataset}/knn_${dataset}"
  k=50

  command="bazel run benchmark:run_experiment -- \
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

  bazel build //experimental/users/shangdi/parclusterer_exp/benchmark:parhac_main

  mkdir -p ${output_file}

  echo $command  # Print the command for verification
  eval $command  # Execute the command
done
