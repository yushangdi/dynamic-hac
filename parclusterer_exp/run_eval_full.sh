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

cd parclusterer_exp
export EXP_ROOT=`pwd`

weight=0.0001
k=50

# method="dynamic_hac_deletion"
datasets=("mnist") #mnist "aloi"    "imagenet" "ilsvrc_small"
num_batches=(70000) #70000 108000    100000 50000

epsilon="0.1"
methods=("dynamic_hac" "dynamic_hac_deletion")

for i in {0..0}; do
    for method in ${methods[@]}; do
      dataset=${datasets[i]}
      num_batch=${num_batches[i]}

      input_data="benchmark/${dataset}/${dataset}.scale.permuted.fvecs"
      ground_truth="benchmark/${dataset}/${dataset}.scale.permuted_label.csv"

      # dynamic hac
      clustering="benchmark/result/dynamic_hac/${dataset}"
      output_file="$EXP_ROOT/benchmark/results_dyn/${dataset}"

      if [[ "$method" = "dynamic_hac_deletion" ]];
      then 
        clustering="benchmark/result/dynamic_hac_deletion/${dataset}"
        output_file="$EXP_ROOT/benchmark/results_dyn_deletion/${dataset}"
      fi


      command="bazel run -c opt benchmark:run_experiment -- \
      --run_hac=False \
      --input_data=${input_data} \
      --ground_truth=${ground_truth} \
      --num_batch=${num_batch} \
      --clustering=${clustering} \
      --output_file=${output_file} \
      --weight=${weight} --epsilon=${epsilon}\
      --method=${method}"

      mkdir -p ${output_file}

      echo $command  # Print the command for verification
      eval $command  # Execute the command
    done
done
