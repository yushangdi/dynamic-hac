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

use_output_knn="False"
weight=0.0001

# dataset="iris"
# num_batch=2
# use_output_knn="True"

datasets=("aloi" "imagenet" "ilsvrc_small") #mnist
num_batches=(108000 100000 50000) #70000

 bazel run benchmark:parhac_main -- \
 --input_data=benchmark/aloi/aloi.scale.permuted.fvecs \
 --output_clustering=benchmark/result/dynamic_hac_2/aloi_0.1_0.0001_108000 \
 --k=50 --max_degree=100 --l=100 --epsilon=0.1 --weight_threshold=0.0001 \
 --num_batch=108000 --pbbs_num_workers=1 --method=dynamic_hac --output_knn= \
 --use_output_knn=False \ 2>&1 | tee aloi_log_eps_0.1_weight_0.0001_108000.txt

   bazel run benchmark:parhac_main -- \
 --input_data=benchmark/ilsvrc_small/ilsvrc_small.scale.permuted.fvecs \
 --output_clustering=benchmark/result/dynamic_hac_2/ilsvrc_small_0.1_0.0001_108000 \
 --k=50 --max_degree=100 --l=100 --epsilon=0.1 --weight_threshold=0.0001 \
 --num_batch=50000 --pbbs_num_workers=1 --method=dynamic_hac \
 --use_output_knn=False  2>&1 | tee ilsvrc_small_log_eps_0.1_weight_0.0001_50000.txt


  bazel run benchmark:parhac_main -- \
 --input_data=benchmark/aloi/aloi.scale.permuted.fvecs \
 --output_clustering=benchmark/result/dynamic_hac_2/aloi_0.1_0.0001_10800 \
 --k=50 --max_degree=100 --l=100 --epsilon=0.1 --weight_threshold=0.0001 \
 --num_batch=10800 --pbbs_num_workers=1 --method=dynamic_hac --output_knn= \
 --use_output_knn=False \ 2>&1 | tee aloi_log_eps_0.1_weight_0.0001_10800.txt


  bazel run benchmark:parhac_main -- \
 --input_data=benchmark/imagenet/imagenet.scale.permuted.fvecs \
 --output_clustering=benchmark/result/dynamic_hac_2/imagenet_0.1_0.0001_108000 \
 --k=50 --max_degree=100 --l=100 --epsilon=0.1 --weight_threshold=0.0001 \
 --num_batch=100000 --pbbs_num_workers=1 --method=dynamic_hac \
 --use_output_knn=False  2>&1 | tee imagenet_log_eps_0.1_weight_0.0001_100000.txt


bazel run benchmark:evaluate_clustering -- \
 --clustering=benchmark/result/dynamic_hac_2/aloi_0.1_0.0001_108000 \
 --ground_truth=benchmark/aloi/aloi.scale.permuted_label.csv \
 --log_file=parclusterer_exp/aloi_tail.txt \
 --output_file=parclusterer_exp/aloi_tail


bazel run benchmark:process_result -- \
--input_file=parclusterer_exp/aloi_tail.txt \
--output_file=parclusterer_exp/aloi_tail.png


############# ALOI head

bazel run benchmark:process_result -- --input_file=parclusterer_exp/benchmark/results_dyn/aloi/head_log_eps_0.1_weight_0.0001_108000.txt \
--output_file=parclusterer_exp/benchmark/results_dyn/aloi/log_eps_0.1_weight_0.0001_108000.png

bazel run benchmark:evaluate_clustering  -- \
--log_file=parclusterer_exp/benchmark/results_dyn/aloi/head_log_eps_0.1_weight_0.0001_108000.txt \
--output_file=parclusterer_exp/benchmark/results_dyn/aloi/log_eps_0.1_weight_0.0001_108000.png \
--threshold=0.0001 -ground_truth=benchmark/aloi/aloi.scale.permuted_label.csv \
--clustering=benchmark/result/dynamic_hac/aloi_0.1_0.0001_108000


############# ILSVRC head

bazel run benchmark:process_result -- \
--input_file=parclusterer_exp/benchmark/results_dyn_deletion/ilsvrc_small/log_eps_0.1_weight_0.0001_50000.txt \
--output_file=parclusterer_exp/benchmark/results_dyn_deletion/ilsvrc_small/log_eps_0.1_weight_0.0001_50000.png

bazel run benchmark:evaluate_clustering  -- \
--log_file=parclusterer_exp/benchmark/results_dyn_deletion/ilsvrc_small/log_eps_0.1_weight_0.0001_50000.txt \
--output_file=parclusterer_exp/benchmark/results_dyn_deletion/ilsvrc_small/log_eps_0.1_weight_0.0001_50000.png \
--threshold=0.0001 -ground_truth=benchmark/ilsvrc_small/ilsvrc_small.scale.permuted_label.csv \
--clustering=benchmark/result/dynamic_hac_deletion/ilsvrc_small_0.1_0.0001_50000