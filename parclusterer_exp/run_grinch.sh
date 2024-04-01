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

# dataset="mnist"
# dataset="test"
# dataset="aloi"
# dataset="imagenet"
dataset="ilsvrc_small"
# dataset="iris"

ratio=0.99

base_dir="parclusterer_exp"

command="bazel run -c opt //experimental/users/shangdi/parclusterer_exp/benchmark:grinch_main -- \
--log_file=${base_dir}/benchmark/
--dataset=${dataset} --eval_index_ratio=${ratio}"

mkdir -p ${base_dir}/benchmark/grinch_insertion
mkdir -p ${base_dir}/benchmark/grinch_deletion

echo $command  # Print the command for verification
eval $command  # Execute the command