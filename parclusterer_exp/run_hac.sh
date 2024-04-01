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



# dataset="iris"
# num_batch=2
# use_output_knn="True"

weight=0.0001
k=50
num_batch=100
use_output_knn="True"

datasets=("mnist") #mnist "aloi" "imagenet" "ilsvrc_small"

for dataset in ${datasets[@]}
do
  input_data="benchmark/${dataset}/${dataset}.scale.permuted.fvecs"
  ground_truth="benchmark/${dataset}/${dataset}.scale.permuted_label.csv"
  clustering="benchmark/result/parhac/${dataset}"
  output_file="$EXP_ROOT/benchmark/results_hac/${dataset}"
  output_knn="benchmark/result/knn/${dataset}/knn_${dataset}"

  command="bazel run benchmark:run_experiment -- \
  --input_data=${input_data} \
  --ground_truth=${ground_truth} \
  --clustering=${clustering} \
  --output_file=${output_file} \
  --num_batch=${num_batch} \
  --k=${k} \
  --weight=${weight} \
  --output_knn=${output_knn}\
  --use_output_knn=${use_output_knn}\
  --first_batch_ratio=0"

  bazel build //benchmark:parhac_main

  mkdir -p ${output_file}

  echo $command
  eval $command
done