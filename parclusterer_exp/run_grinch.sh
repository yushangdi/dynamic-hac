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

ratio=0
dataset=$1
echo ${dataset}
# dataset="mnist"
# dataset="test"
# dataset="aloi"
# dataset="ilsvrc_small"
# dataset="iris"

# ratio=0.99

base_dir="$EXP_ROOT/results/"

command="python3 parclusterer_exp/benchmark/grinch_main.py \
--log_file=$EXP_ROOT/
--dataset=${dataset} --eval_index_ratio=${ratio} --batch_num=100"

mkdir -p ${base_dir}/grinch_insertion
mkdir -p ${base_dir}/grinch_deletion

bazel build //parclusterer_exp/benchmark:cut_dendrogram

echo ${command}  # Print the command for verification
eval ${command}  # Execute the command