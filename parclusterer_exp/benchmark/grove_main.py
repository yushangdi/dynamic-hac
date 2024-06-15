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

# Copyright 2024 Approximate Hac Experiments Authors
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

"""Run experiments on GraphGrove online SCC.
Results are stored to base_dir+results/results_grove/

"""

from collections.abc import Sequence
import struct
import time

from absl import app
from absl import flags
import evaluate_utils
import graphgrove
import graphgrove.vec_scc
# from graphgrove.scc import SCC
# from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
# from graphgrove.vec_scc import Cosine_SCC
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import sklearn.metrics.cluster as cluster_metrics
import tqdm

base_dir = "./"

_DATASET = flags.DEFINE_string(
    "dataset",
    default=None,
    help="Input data in fvecs format.",
)

_K = flags.DEFINE_integer("k", default=50, help="k.")

_EVAL_INDEX_RATIO = flags.DEFINE_float(
    "eval_index_ratio",
    0,
    "only store ARIs for index [x*n, n].",
)

_BATCH_NUM = flags.DEFINE_integer(
    "batch_num", 1000, "num. of dendroram to evaluate."
)


def fvecs_read(f):
  a = np.frombuffer(f.read(), dtype="int32")
  d = a[0]
  return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def read_edges(filename):
  edges = []
  with open(filename, "rb") as f:
    n, m = struct.unpack("QQ", f.read(16))

    for _ in range(m):
      int1, int2, double1 = struct.unpack(
          "iid", f.read(16)
      )  # iid for int, int, double
      edges.append([int1, int2, double1])
  return edges


def get_clusters(level, n):
  clustering = np.ones(n) * -1
  for i, node in enumerate(level.nodes):
    # node.set_descendants()
    descendants = node.descendants()
    clustering[descendants] = i
  return clustering


def find_best_cut(scc, ground_truth, i):
  """Find the best scc level when compared with ground_truth."""

  best_ari = 0
  best_num_cluster = 0
  num_true_cluster = len(np.unique(ground_truth[:i]))
  num_clusters = []
  best_nmi = 0
  best_num_cluster_nmi = 0
  for level in scc.levels:
    clustering = get_clusters(level, i)
    # print(clustering)
    ari = cluster_metrics.adjusted_rand_score(ground_truth[:i], clustering)
    nmi = cluster_metrics.normalized_mutual_info_score(
        ground_truth[:i], clustering
    )
    num_cluster = len(np.unique(clustering))
    num_clusters.append(num_cluster)
    if ari > best_ari:
      best_ari = ari
      best_num_cluster = num_cluster
    if nmi > best_nmi:
      best_nmi = nmi
      best_num_cluster_nmi = num_cluster
  # num_clusters = np.array(num_clusters)
  return best_ari, best_num_cluster, best_nmi, best_num_cluster_nmi


def get_data(dataset_subdir):
  with open(
      base_dir + dataset_subdir,
      mode="rb",
  ) as f:
    return fvecs_read(f)


dataset_subdirs = {
    "mnist": "data/mnist/mnist.scale.permuted.fvecs",
    "aloi": "data/aloi/aloi.scale.permuted.fvecs",
    "iris": "data/iris/iris.scale.permuted.fvecs",
    "ilsvrc_small": "data/ilsvrc_small/ilsvrc_small.scale.permuted.fvecs",
}


def main(argv):

  dataset_name = _DATASET.value
  if dataset_name in dataset_subdirs:
    points = get_data(dataset_subdirs[dataset_name])
  else:
    raise ValueError(f"Unknown dataset: {dataset_name}")
  # Normalize each vector to have a unit norm
  points = points / np.linalg.norm(points, axis=-1, keepdims=True)
  # add random noise to avoid duplicate entry
  noise = np.random.uniform(-1e-6, 1e-6, size=points.shape)
  points = (points + noise).astype(np.float32)
  n = len(points)

  ground_truth_file = f"{base_dir}/data/{dataset_name}/{dataset_name}.scale.permuted_label.bin"
  ground_truth = evaluate_utils.read_ground_truth(ground_truth_file)

  # edges = read_edges(
  #     "/result/knn/iris/knn_iris_150.bin"
  # )
  # edges = np.array(edges)

  batch_num = _BATCH_NUM.value
  batch_size = max(1, n // batch_num)
  eval_index_ratio = _EVAL_INDEX_RATIO.value
  eval_index = int(eval_index_ratio * n)
  print("batch size: ", batch_size)
  print("eval_index: ", eval_index)

  cores = 1
  num_rounds = 50
  k = _K.value
  thresholds = np.geomspace(1, 1e-8, num=num_rounds).astype(np.float32)
  # scc = SCC.init(thresholds, cores)  # , verbosity=1
  scc = graphgrove.vec_scc.Cosine_SCC(k=k, num_rounds=num_rounds, thresholds=thresholds, cores=1)
  # scc.insert_graph_mb(edges[:, 0], edges[:, 1], edges[:, 2])
  times = []
  indices = []
  ari_indices = []
  aris = []
  num_clusters = []
  nmis = []
  nmis_num_clusters = []
  insertion_stored_indices = []
  # batch insert first batch
  if eval_index > 0:
    scc.partial_fit(np.array(points[:eval_index, :]))
  for i in tqdm.tqdm(range(eval_index, len(points))):
    time_start = time.time()
    scc.partial_fit(np.array([points[i, :]]))
    time_end = time.time()
    times.append(time_end - time_start)
    indices.append(i)
    if i >= eval_index and (i % batch_size == 0 or i == n - 1):
      ari, num_cluster, best_nmi, best_num_cluster_nmi = find_best_cut(
          scc.scc, ground_truth, i + 1
      )
      aris.append(ari)
      num_clusters.append(num_cluster)
      insertion_stored_indices.append(i)
      nmis.append(best_nmi)
      nmis_num_clusters.append(best_num_cluster_nmi)

  df_dict = {
      "Index": insertion_stored_indices,
      "ARI": aris,
      "Num_Clusters": num_clusters,
      "NMI": nmis,
      "NMI_Num_Clusters": nmis_num_clusters,
  }
  df = pd.DataFrame(df_dict)
  df.to_csv(base_dir + f"results/results_grove/{dataset_name}_{batch_num}_nmi.csv")
  time_dict = {
      "Index": indices,
      "Round": times,
  }
  df = pd.DataFrame(time_dict)
  df.to_csv(base_dir + f"results/results_grove/{dataset_name}_{batch_num}_time.csv")

  # print(scc.levels)
  # for level in scc.levels:
  #   print(level.nodes)
  #   print([node.descendants() for node in level.nodes])


if __name__ == "__main__":
  app.run(main)
