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

"""Reading files for plotting"""

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd

base_dir = "results/"
epsilon = 0.1


def ComputeSlidingWindow(df, col, window_size=100):
  df["Sliding_Window_Avg"] = (
      df[col].rolling(window=window_size, center=True, min_periods=1).mean()
  )


def get_hac_time(dataset):
  hac_batch_num = 100
  threshold = 0.01
  if dataset == "mnist":
    threshold = 0.0001
  hac_time = (
      base_dir
      + "results_hac/"
      + dataset
      + f"/fig_eps_{epsilon}_weight_{threshold}_{hac_batch_num}.png_running_times.csv"
  )
  return pd.read_csv(hac_time)


def get_hac_quality(dataset):
  hac_batch_num = 100
  threshold = 0.01
  if dataset == "mnist":
    threshold = 0.0001
  hac_quality = (
      base_dir
      + "results_hac/"
      + dataset
      + f"/fig_eps_{epsilon}_weight_{threshold}_{hac_batch_num}_nmi.csv"
  )
  return pd.read_csv(hac_quality)


def get_grinch_dfs(dataset):
  grinch_insertion_time = (
      base_dir + "grinch_insertion/" + dataset + f"_1000_time.csv"
  )
  grinch_insertion_quality = (
      base_dir + "grinch_insertion/" + dataset + f"_1000_nmi.csv"
  )

  grinch_deletion_time = (
      base_dir + "grinch_deletion/" + dataset + f"_1000_time.csv"
  )
  grinch_deletion_quality = (
      base_dir + "grinch_deletion/" + dataset + f"_1000_nmi.csv"
  )

  try:
    df_grinch_insertion_time = pd.read_csv(grinch_insertion_time)
    ComputeSlidingWindow(df_grinch_insertion_time, "Round")
    df_grinch_insertion_quality = pd.read_csv(grinch_insertion_quality)
  except Exception as e:
    print(e)
    df_grinch_insertion_time = pd.DataFrame(
        columns=["Index", "Clustering", "Round", "Sliding_Window_Avg"]
    )
    df_grinch_insertion_quality = pd.DataFrame(columns=["Index", "NMI"])
  try:
    df_grinch_deletion_time = pd.read_csv(grinch_deletion_time)
    ComputeSlidingWindow(df_grinch_deletion_time, "Round")
    df_grinch_deletion_quality = pd.read_csv(grinch_deletion_quality)
  except Exception as e:
    print(e)
    df_grinch_deletion_time = pd.DataFrame(
        columns=["Index", "Clustering", "Round", "Sliding_Window_Avg"]
    )
    df_grinch_deletion_quality = pd.DataFrame(columns=["Index", "NMI"])

  return (
      df_grinch_insertion_time,
      df_grinch_insertion_quality,
      df_grinch_deletion_time,
      df_grinch_deletion_quality,
  )


def get_grove_dfs(dataset):
  batch_num = 1000
  if dataset == "mnist":
    batch_num = 100
  grove_insertion_time = (
      base_dir + "results_grove/" + dataset + f"_{batch_num}_time.csv"
  )
  grove_insertion_quality = (
      base_dir + "results_grove/" + dataset + f"_{batch_num}_nmi.csv"
  )

  df_grove_insertion_time = pd.read_csv(grove_insertion_time)
  ComputeSlidingWindow(df_grove_insertion_time, "Round")
  df_grove_insertion_quality = pd.read_csv(grove_insertion_quality)

  return (
      df_grove_insertion_time,
      df_grove_insertion_quality,
  )


def get_tail_time(dataset, epsilon, n, insertion):
  threshold = 0.01
  if dataset == "mnist":
    threshold = 0.0001
  if insertion:
    time_file = (
        base_dir
        + f"results_dyn/tail_{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}.png_running_times.csv"
    )
  else:
    time_file = (
        base_dir
        + f"results_dyn_deletion/tail_{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}.png_running_times.csv"
    )
  df_time = pd.read_csv(time_file)
  return df_time


def get_tail_quality(dataset, epsilon, n, insertion):
  threshold = 0.01
  if dataset == "mnist":
    threshold = 0.0001
  if insertion:
    quality_file = (
        base_dir
        + f"results_dyn/tail_{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}_nmi.csv"
    )
    df_quality = pd.read_csv(quality_file)
    return df_quality
  else:
    quality_file = (
        base_dir
        + f"results_dyn_deletion/tail_{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}_nmi.csv"
    )
    df_quality = pd.read_csv(quality_file)
    return df_quality


def get_full_time(dataset, epsilon, n, insertion):
  threshold = 0.01
  if dataset == "mnist":
    threshold = 0.0001
  if insertion:
    time_file = (
        base_dir
        + f"results_dyn/{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}.png_running_times.csv"
    )
  else:
    time_file = (
        base_dir
        + f"results_dyn_deletion/{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}.png_running_times.csv"
    )
  df_time = pd.read_csv(time_file)
  return df_time


def get_full_quality(dataset, epsilon, n, insertion):
  threshold = 0.01
  if dataset == "mnist":
    threshold = 0.0001
  if insertion:
    quality_file = (
        base_dir
        + f"results_dyn/{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}_nmi.csv"
    )
    df_quality = pd.read_csv(quality_file)
    return df_quality
  else:
    quality_file = (
        base_dir
        + f"results_dyn_deletion/{dataset}/"
        + f"fig_eps_{epsilon}_weight_{threshold}_{n}_nmi.csv"
    )
    df_quality = pd.read_csv(quality_file)
    return df_quality


def extract_times_and_num_dirty(text):
  """extract the times from text."""
  times = []
  num_dirty_edges = []
  # Remove all lines starting with I1004 (the logs).
  for block in text.split("======"):
    matches = re.search(r"Clustering time: ([\d.e-]+) seconds", block)
    matches_edge = re.search(r"Num. Dirty Edges: ([\d]+)", block)
    if matches and matches_edge:
      times.append(float(matches.group(1)))
      num_dirty_edges.append(int(matches_edge.group(1)))
      # times[index]['Clustering'] = float(matches.group(1))
  return times, num_dirty_edges


def get_df_times_dirty_edges():
  threshold = 0.0001
  log_file = (
      base_dir
      + "results_dyn/mnist"
      + f"/log_eps_{epsilon}_weight_{threshold}_70000.txt"
  )

  with open(log_file, "r") as f:
    times, num_dirty = extract_times_and_num_dirty(f.read())
  return times, num_dirty


def extract_rounds(text):
  """extract the times from text."""
  num_rounds = []
  indices = []
  # Remove all lines starting with I1004 (the logs).
  for block in text.split("======"):
    if "index:" in block:
      s = re.search(r"index: (\d+)", block)
      matches = re.search(r"Num. Rounds: ([\d]+)", block)
      if s and matches:
        index = int(s.group(1))
        indices.append(index)
        num_rounds.append(int(matches.group(1)))
  return indices, num_rounds


def get_df_num_rounds():
  threshold = 0.0001
  log_file = (
      base_dir
      + "results_dyn/mnist"
      + f"/log_eps_{epsilon}_weight_{threshold}_70000.txt"
  )

  with open(log_file, "r") as f:
    indices, num_rounds = extract_rounds(f.read())
  return indices, num_rounds


def get_df_value(df, col, i):
  return float(df[df["Index"] == i][col].iloc[0])


def get_bar_dataframe(dataset, n):
  """get the dataframe for bar plot comparing different methods"""
  epsilon = "0.1"
  df_hac_time = get_hac_time(dataset)
  df_hac_quality = get_hac_quality(dataset)
  # if dataset == "mnist":
  #   df_dyn_insertion_time = get_full_time(dataset, epsilon, n, True)
  #   df_dyn_insertion_quality = get_full_quality(dataset, epsilon, n, True)
  #   df_dyn_deletion_time = get_full_time(dataset, epsilon, n, False)
  # else:
  df_dyn_insertion_time = get_tail_time(dataset, epsilon, n, True)
  ComputeSlidingWindow(df_dyn_insertion_time, "Round")
  df_dyn_insertion_quality = get_tail_quality(dataset, epsilon, n, True)
  df_dyn_deletion_time = get_tail_time(dataset, epsilon, n, False)
  ComputeSlidingWindow(df_dyn_deletion_time, "Clustering")
  # df_dyn_deletion_quality = get_tail_quality(dataset, epsilon, n, False)
  (
      df_grinch_insertion_time,
      df_grinch_insertion_quality,
      df_grinch_deletion_time,
      df_grinch_deletion_quality,
  ) = get_grinch_dfs(dataset)
  (
      df_grove_insertion_time,
      df_grove_insertion_quality,
  ) = get_grove_dfs(dataset)

  # insertion
  methods = ["Static HAC", "DynHAC", "GRINCH", "Grove"]
  times = [0, 0, 0, 0]
  nmis = [0, 0, 0, 0]
  times[0] = get_df_value(df_hac_time, "Clustering", n)
  times[1] = get_df_value(df_dyn_insertion_time, "Sliding_Window_Avg", n)
  times[2] = get_df_value(df_grinch_insertion_time, "Sliding_Window_Avg", n - 1)
  times[3] = get_df_value(df_grove_insertion_time, "Sliding_Window_Avg", n - 1)
  nmis[0] = get_df_value(df_hac_quality, "NMI", n)
  nmis[1] = get_df_value(df_dyn_insertion_quality, "NMI", n)
  nmis[2] = get_df_value(df_grinch_insertion_quality, "NMI", n - 1)
  nmis[3] = get_df_value(df_grove_insertion_quality, "NMI", n - 1)
  speedups = [times[0] / i for i in times]  # speedup over static hac

  df_ins = pd.DataFrame({
      "Dataset": dataset.upper(),  # .replace("_", "\_")
      "Algorithm": methods,
      "Clustering": times,
      "NMI": nmis,
      "Speedup": speedups,
  })

  # deletion
  times[1] = get_df_value(df_dyn_deletion_time, "Clustering", 1)
  times[2] = get_df_value(df_grinch_deletion_time, "Round", n - 1)
  speedups = [times[0] / i for i in times]  # speedup over static hac

  df_del = pd.DataFrame({
      "Dataset": dataset.upper(),  # .replace("_", "\_")
      "Algorithm": methods[:3],
      "Clustering": times[:3],
      "NMI": nmis[:3],
      "Speedup": speedups[:3],
  })

  return df_ins, df_del


def get_df_average(df, col):
  return float(df[col].mean())


def get_bar_dataframe_epsilon(dataset, n):
  """get the dataframe for bar plot comparing different epsilon"""
  epsilons = ["0.0", "0.1", "1.0"]
  insertion_times = [0, 0, 0]
  insertion_nmis = [0, 0, 0]
  deletion_times = [0, 0, 0]
  deletion_nmis = [0, 0, 0]
  for i, epsilon in enumerate(epsilons):
    df_dyn_insertion_time = get_tail_time(dataset, epsilon, n, True)
    df_dyn_insertion_time = df_dyn_insertion_time.iloc[1:, :]
    df_dyn_insertion_quality = get_tail_quality(dataset, epsilon, n, True)
    df_dyn_deletion_time = get_tail_time(dataset, epsilon, n, False)
    df_dyn_deletion_quality = get_tail_quality(dataset, epsilon, n, False)

    insertion_times[i] = get_df_average(df_dyn_insertion_time, "Clustering")
    insertion_nmis[i] = get_df_average(df_dyn_insertion_quality, "NMI")
    deletion_times[i] = get_df_average(df_dyn_deletion_time, "Clustering")
    deletion_nmis[i] = get_df_average(df_dyn_deletion_quality, "NMI")

  df_ins = pd.DataFrame({
      "Dataset": dataset.upper(),  # .replace("_", "\_")
      "epsilon": epsilons,
      "Clustering": insertion_times,
      "NMI": insertion_nmis,
      "Speedup": [insertion_times[0] / i for i in insertion_times],
  })

  df_del = pd.DataFrame({
      "Dataset": dataset.upper(),  # .replace("_", "\_")
      "epsilon": epsilons,
      "Clustering": deletion_times,
      "NMI": deletion_nmis,
      "Speedup": [deletion_times[0] / i for i in deletion_times],
  })

  return df_ins, df_del
