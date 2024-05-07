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





# dataset="iris"
# num_batch=2
# use_output_knn="True"

export EXP_ROOT=$(pwd) # set EXP_ROOT to current directory
export PARLAY_NUM_THREADS=1

weight=0.0001
k=50
num_batch=100
use_output_knn="True"
run_hac="False"

datasets=("mnist") #mnist "aloi" "imagenet" "ilsvrc_small"

for dataset in ${datasets[@]}
do
  input_data="$EXP_ROOT/data/${dataset}/${dataset}.scale.permuted.fvecs"
  ground_truth="$EXP_ROOT/data/${dataset}/${dataset}.scale.permuted_label.bin"
  clustering="$EXP_ROOT/results/parhac/${dataset}"
  output_file="$EXP_ROOT/results/results_hac/${dataset}"
  output_knn="$EXP_ROOT/results/knn/${dataset}/knn_${dataset}"

  command="python3 parclusterer_exp/benchmark/run_experiment.py \
  --method=parhac \
  --run_hac=${run_hac} \
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

  bazel build //parclusterer_exp/benchmark:cut_dendrogram
  bazel build //parclusterer_exp/benchmark:parhac_main

  mkdir -p ${output_file}

  echo $command
  eval $command
done