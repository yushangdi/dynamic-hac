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

"""Util functions for evaluate clustering."""

from collections.abc import Sequence
import re
import os
import ctypes
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics.cluster as cluster_metrics
import tqdm


# # Load the shared library
# exp_root = os.environ.get('EXP_ROOT')
# cut_dendrogram = ctypes.CDLL(os.path.join(exp_root, "bazel-bin/parclusterer_exp/benchmark/cut_dendrogram.so"))

sys.path.append("/home/sy/dynamic-hac/bazel-bin/parclusterer_exp/benchmark")
import cut_dendrogram

class Error(Exception):
  """Base class for exceptions in this module."""


def read_file(pathname):
  """Returns the contents of a binary file."""
  try:
    with open(pathname, 'rb') as input_file:
      # Read the entire file contents
      file_content = input_file.read()
      return file_content
  except Exception as e:
    raise Error('Could not open file: ' + pathname) from e


def read_ground_truth(ground_truth_pathname):
  """read a vector from a binary file, the number type is int."""
  ground_truth_buffer = read_file(ground_truth_pathname)
  ground_truth = np.frombuffer(ground_truth_buffer, dtype=int)
  return ground_truth.astype(str)


def extract_indices(text):
  """extract the indices from text."""
  indices = []
  for block in text.split('======'):
    if 'index:' in block:
      s = re.search(r'index: (\d+)', block)
      if s:
        index = int(s.group(1))
        indices.append(index)
      else:
        print(block)
        raise Error('index not found')

  return indices


def extract_stored_clustering_indices(text):
  """extract the indices from text."""
  indices = []
  for block in text.split('======'):
    if 'Store index:' in block:
      s = re.search(r'Store index: (\d+)', block)
      if s:
        index = int(s.group(1))
        indices.append(index)

  return indices


def plot(
    indices,
    nmis,
    num_clusters,
    output_file,
):
  """plot the results."""
  plt.clf()
  plt.plot(indices, nmis)
  plt.scatter(indices, nmis)
  plt.title('Clustering NMI')
  plt.xlabel('# Nodes')
  plt.ylabel('NMI')
  plt.tight_layout()
  plt.savefig(output_file + '_nmis.png', format='png')

  plt.clf()
  plt.plot(indices, num_clusters)
  plt.scatter(indices, num_clusters)
  plt.title('# Clusters')
  plt.xlabel('# Nodes')
  plt.ylabel('# clusters')
  plt.tight_layout()
  plt.savefig(output_file + '_num_clusters.png', format='png')


def evaluate(
    ground_truth_file,
    log_text,
    clustering_file_base,
    output_file,
    threshold,
):
  """evaluete clustering for all indices in log_text."""
  ground_truth = read_ground_truth(ground_truth_file)
  print('ground truth:', ground_truth)
  indices = extract_stored_clustering_indices(log_text)
  assert len(indices) > 0
  if len(ground_truth) != indices[-1]:
    print('Warning: not all points are processed.')
  nmis = []
  num_clusters = []
  thresholds_used = []
  thresholds = list(np.logspace(-4, -1, 20))
  thresholds += list(np.logspace(-1, 0, 20))
  print(thresholds)
  for index in tqdm.tqdm(indices):
    dendrogram = cut_dendrogram.ReadDendrogram(
        clustering_file_base + '-' + str(index) + '-dendro.bin'
    )
    # TODO: find the best score
    best_threshold = 0
    best_nmi = 0
    best_num_cluster_nmi = 0
    for cut_threshold in thresholds:
      clustering = cut_dendrogram.CutDendrogramAt(dendrogram, cut_threshold)
      clustering_flatten = np.zeros(index)
      for cid, cluster in enumerate(clustering):
        for i in cluster:
          clustering_flatten[i] = cid

      nmi = cluster_metrics.normalized_mutual_info_score(
          ground_truth[:index], clustering_flatten
      )
      if nmi > best_nmi:
        best_nmi = nmi
        best_num_cluster_nmi = len(clustering)
        best_threshold = cut_threshold
    num_clusters.append(best_num_cluster_nmi)
    nmis.append(best_nmi)
    thresholds_used.append(best_threshold)

  df_dict = {
      'Index': indices,
      'NMI': nmis,
      'Num_Clusters': num_clusters,
      'Thresholds': thresholds_used
  }
  df = pd.DataFrame(df_dict)
  df.to_csv(output_file + '_nmi.csv')
  plot(indices, nmis, num_clusters, output_file)


def extract_times(text):
  """extract the times from text."""
  times = {}
  index = 0
  # Remove all lines starting with I1004 (the logs).
  text = re.sub(r'^I0114.*\n', '', text, flags=re.MULTILINE)
  for block in text.split('======'):
    if 'index:' in block:
      s = re.search(r'index: (\d+)', block)
      if s:
        index = int(s.group(1))
        # index = int(re.search(r"index: (\d+)", block).group(1))
        times[index] = {}
      else:
        print(block)
        raise Error('index not found')
    for line in block.splitlines():
      matches = re.search(
          r'(KNN|Clustering|Round) time: ([\d.e+-]+) seconds', line
      )
      if matches:
        times[index][matches.group(1)] = float(matches.group(2))
  return times


def create_dataframe(times):
  data = []
  # times in seconds
  for index, time_data in times.items():
    data.append({
        'Index': index,
        'KNN': time_data['KNN'],
        'Clustering': time_data['Clustering'],
        'Round': time_data['Round'],
    })
  return pd.DataFrame(data)


def plot_running_times(text, output_file):
  """plot the running times."""
  times = extract_times(text)
  df = create_dataframe(times)
  print(df.head())
  df.to_csv(output_file + '_running_times.csv')
  df = df.drop(['Round'], axis=1)

  # Create the plot
  plt.stackplot(
      df['Index'],
      df['KNN'],
      df['Clustering'],
      labels=['KNN', 'Clustering'],
      alpha=0.7,
  )

  plt.title('Clustering Time')
  plt.xlabel('# Nodes inserted')
  plt.ylabel('Time (seconds)')
  plt.legend()
  plt.savefig(output_file, format='png')


def extract_times_deletion(text):
  """extract the times from text."""
  times = {}
  index = 0
  # Remove all lines starting with I1004 (the logs).
  text = re.sub(r'^I0114.*\n', '', text, flags=re.MULTILINE)
  for block in text.split('======'):
    if 'index:' in block and 'early stop' not in block:
      s = re.search(r'index: (\d+)', block)
      if s:
        index = int(s.group(1))
        times[index] = {}
      else:
        raise Error('index not found')
      for line in block.splitlines():
        matches = re.search(r'Clustering time: ([\d.e+-]+) seconds', line)
        if matches:
          times[index]['Clustering'] = float(matches.group(1))
  return times


def plot_running_times_deletion(text, output_file):
  """plot the running times."""
  times = extract_times_deletion(text)
  data = []
  # times in seconds
  for index, time_data in times.items():
    data.append({
        'Index': index,
        'Clustering': time_data['Clustering'],
    })
  df = pd.DataFrame(data)
  print(df.head())
  df.to_csv(output_file + '_running_times.csv')

  # Create the plot
  plt.stackplot(
      df['Index'],
      df['Clustering'],
      labels=['Clustering'],
      alpha=0.7,
  )

  plt.title('Clustering Time')
  plt.xlabel('# Nodes deleted')
  plt.ylabel('Time (seconds)')
  plt.legend()
  plt.savefig(output_file, format='png')


def find_best_cut(gr, ground_truth, i):
  start = 0
  end = 1
  max_ari = 0
  best_threshold = 0
  ground_truth = ground_truth[:i]
  num_true_cluster = len(np.unique(ground_truth))
  best_num_clusters = 0
  best_nmi = 0
  best_num_cluster_nmi = 0
  while end - start > 1e-6:
    threshold = (start + end) / 2
    clustering = gr.flat_clustering(threshold)[:i]
    ari = cluster_metrics.adjusted_rand_score(ground_truth, clustering)
    nmi = cluster_metrics.normalized_mutual_info_score(
        ground_truth[:i], clustering
    )
    num_clusters = len(np.unique(clustering))

    if ari > max_ari:
      max_ari = ari
      best_threshold = threshold
      best_num_clusters = num_clusters

    if nmi > best_nmi:
      best_nmi = nmi
      best_num_cluster_nmi = num_clusters

    if num_clusters < num_true_cluster:
      start = threshold
    else:
      end = threshold

  print(best_nmi, best_num_cluster_nmi)
  return max_ari, best_num_clusters, best_nmi, best_num_cluster_nmi
