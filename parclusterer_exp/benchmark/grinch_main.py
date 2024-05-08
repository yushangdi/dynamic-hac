# coding=utf-8
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

# Copyright 2024 Google LLC
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

"""Grinch evaluation."""

from collections.abc import Sequence
import time

from absl import app
from absl import flags
import evaluate_utils
import grinch
import numpy as np
import pandas as pd
import tqdm


_DATASET = flags.DEFINE_string("dataset", None, "Dataset")
_LOG_FILE = flags.DEFINE_string("log_file", None, "Log File directory")
_BATCH_NUM = flags.DEFINE_integer(
    "batch_num", 1000, "num. of dendroram to evaluate."
)
_EVAL_INDEX_RATIO = flags.DEFINE_float(
    "eval_index_ratio",
    0,
    "only store ARIs for index [x*n, n]. Deletion stops at this index.",
)

base_dir = "./"


def fvecs_read(f):
  a = np.frombuffer(f.read(), dtype="int32")
  d = a[0]
  return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def get_test_data():
  return np.array(
      [(0, 1), (0, 1.1), (-1, -2), (-1, -2.1), (0.02, 1.02), (-0.8, -2.3)]
  )


def get_data(dataset_subdir):
  with open(
      base_dir + dataset_subdir,
      mode="rb",
  ) as f:
    return fvecs_read(f)


dataset_subdirs = {
    "mnist": "data/mnist/mnist.scale.permuted.fvecs",
    "aloi": "data/aloi/aloi.scale.permuted.fvecs",
    "imagenet": "data/imagenet/imagenet.scale.permuted.fvecs",
    "iris": "data/iris/iris.scale.permuted.fvecs",
    "ilvrc_small": "data/ilvrc_small/ilvrc_small.scale.permuted.fvecs",
}


def save_and_flush(output_file, text):
  output_file.write(text)
  output_file.flush()


def append_to_string(flush_string, text):
  flush_string += text + "\n"
  return flush_string


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dataset_name = _DATASET.value

  if dataset_name in dataset_subdirs:
    dataset = get_data(dataset_subdirs[dataset_name])
  else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

  # normalize
  dataset = dataset / np.linalg.norm(dataset, axis=-1, keepdims=True)

  ground_truth_file = (
      f"{base_dir}/data/{dataset_name}/{dataset_name}.scale.permuted_label.bin"
  )
  ground_truth = [1, 2, 3, 4, 5, 6]
  if dataset_name != "test":
    ground_truth = evaluate_utils.read_ground_truth(ground_truth_file)

  print(dataset.shape)
  print(dataset[:3, :5])
  print(ground_truth)
  n = dataset.shape[0]
  batch_num = _BATCH_NUM.value
  batch_size = max(1, n // batch_num)
  eval_index_ratio = _EVAL_INDEX_RATIO.value
  eval_index = int(eval_index_ratio * n)
  print("batch size: ", batch_size)
  print("eval_index: ", eval_index)

  output_filename = _LOG_FILE.value
  gr = grinch.GrinchWithDeletes(dataset, sim="l2")

  flush_string = ""
  insertion_stored_indices = []
  insertion_aris = []
  num_clusters = []
  insertion_times = []
  indices = []
  insertion_nmis = []
  nmis_num_clusters = []
  cutting_thresholds = []

  with open(
      output_filename + f"results/grinch_insertion/{dataset_name}_{batch_num}.log",
      "w",
      buffering=1,
  ) as output_file:

    for i in tqdm.tqdm(range(n), "grinch_insertion"):
      s = time.time()
      gr.insert(i)
      _, knn_time = gr.stats_string()
      round_time = time.time() - s

      # Append the statements to the output string
      flush_string = append_to_string(flush_string, "====== index: " + str(i))
      flush_string = append_to_string(
          flush_string, "KNN time: " + str(knn_time) + " seconds"
      )
      flush_string = append_to_string(
          flush_string,
          "Clustering time: " + str(round_time - knn_time) + " seconds",
      )
      flush_string = append_to_string(
          flush_string, "Round time: " + str(round_time) + " seconds"
      )
      indices.append(i)
      insertion_times.append(round_time)

      if i >= eval_index and (i % batch_size == 0 or i == n - 1):
        ari, num_cluster, best_nmi, best_num_cluster_nmi, threshold = (
            evaluate_utils.find_best_cut(gr, ground_truth, i)
        )
        insertion_stored_indices.append(i)
        insertion_aris.append(ari)
        num_clusters.append(num_cluster)
        insertion_nmis.append(best_nmi)
        nmis_num_clusters.append(best_num_cluster_nmi)
        cutting_thresholds.append(threshold)
        save_and_flush(output_file, flush_string)
        flush_string = ""

  df_dict = {
      "Index": insertion_stored_indices,
      "ARI": insertion_aris,
      "Num_Clusters": num_clusters,
      "NMI": insertion_nmis,
      "NMI_Num_Clusters": nmis_num_clusters,
      'Thresholds': cutting_thresholds
  }
  df = pd.DataFrame(df_dict)
  df.to_csv(base_dir + f"results/grinch_insertion/{dataset_name}_{batch_num}_nmi.csv")
  evaluate_utils.plot(
      insertion_stored_indices,
      insertion_nmis,
      num_clusters,
      base_dir + f"results/grinch_insertion/{dataset_name}_{batch_num}",
  )

  time_dict = {
      "Index": indices,
      "Round": insertion_times,
  }
  df = pd.DataFrame(time_dict)
  df.to_csv(base_dir + f"results/grinch_insertion/{dataset_name}_{batch_num}_time.csv")

  gr.clear_stats()
  flush_string = ""
  deletion_stored_indices = []
  deletion_aris = []
  num_clusters = []
  indices = []
  deletion_times = []
  deletion_nmis = []
  nmis_num_clusters = []
  cutting_thresholds = []
  with open(
      output_filename
      + f"results/grinch_deletion/{dataset_name}_{batch_num}.log",
      "w",
      buffering=1,
  ) as output_file:

    for i in tqdm.tqdm(
        range(n - 1, eval_index, -1), "grinch_deletion"
    ):
      s = time.time()
      gr.delete_point(i)
      _, knn_time = gr.stats_string()
      round_time = time.time() - s
      flush_string = append_to_string(
          flush_string, "====== index: " + str(n - i)
      )
      flush_string = append_to_string(flush_string, "removing: " + str(i))
      flush_string = append_to_string(
          flush_string, "KNN time: " + str(knn_time) + " seconds"
      )
      flush_string = append_to_string(
          flush_string,
          "Clustering time: " + str(round_time - knn_time) + " seconds",
      )
      flush_string = append_to_string(
          flush_string, "Round time: " + str(round_time) + " seconds"
      )
      indices.append(i)
      deletion_times.append(round_time)
      if i > eval_index and (i % batch_size == 0 or i == n - 1):
        ari, num_cluster, best_nmi, best_num_cluster_nmi, threshold = (
            evaluate_utils.find_best_cut(gr, ground_truth, i)
        )
        deletion_stored_indices.append(i)
        deletion_aris.append(ari)
        num_clusters.append(num_cluster)
        deletion_nmis.append(best_nmi)
        nmis_num_clusters.append(best_num_cluster_nmi)
        cutting_thresholds.append(threshold)
        save_and_flush(output_file, flush_string)
        flush_string = ""
  df_dict = {
      "Index": deletion_stored_indices,
      "ARI": deletion_aris,
      "Num_Clusters": num_clusters,
      "NMI": deletion_nmis,
      "NMI_Num_Clusters": nmis_num_clusters,
      'Thresholds': cutting_thresholds
  }
  df = pd.DataFrame(df_dict)
  df.to_csv(
      base_dir + f"results/grinch_deletion/{dataset_name}_{batch_num}_nmi.csv"
  )
  evaluate_utils.plot(
      deletion_stored_indices,
      deletion_nmis,
      num_clusters,
      base_dir + f"results/grinch_deletion/{dataset_name}_{batch_num}",
  )

  time_dict = {
      "Index": indices,
      "Round": deletion_times,
  }
  df = pd.DataFrame(time_dict)
  df.to_csv(
      base_dir + f"results/grinch_deletion/{dataset_name}_{batch_num}_time.csv"
  )


if __name__ == "__main__":
  flags.mark_flags_as_required(["dataset", "log_file"])
  app.run(main)
