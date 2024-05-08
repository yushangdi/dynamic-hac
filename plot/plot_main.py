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

"""Make plots."""

from collections.abc import Sequence
import re

from absl import app
import jinja2  # dependency for pringing latex table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_io
import seaborn as sns

base_dir = "./"


def plot_quality(
    dataset,
    df_hac_quality,
    df_dyn_insertion_quality,
    df_dyn_deletion_quality,
    df_grinch_inserton_quality,
    df_grinch_deletion_quality,
    df_grove_insertion_quality,
):
  plt.clf()

  # Create a figure with two subplots (1 row, 2 columns)
  fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
  markersize = 60
  linewidth = 10

  # Plot HAC in both subplots
  for ax in axs:
    sns.lineplot(
        x="Index",
        y="NMI",
        data=df_hac_quality,
        label="Static HAC",
        color="blue",
        ax=ax,
        linewidth=linewidth,
    )

  # Plot Ins in the top subplot
  sns.scatterplot(
      x="Index",
      y="NMI",
      data=df_dyn_insertion_quality,
      label="DynHAC",
      color="orange",
      ax=axs[0],
      s=markersize,
      zorder=10,
  )
  sns.lineplot(
      x="Index",
      y="NMI",
      data=df_grinch_inserton_quality,
      label="Grinch",
      color="red",
      ax=axs[0],
      linewidth=linewidth,
  )

  sns.lineplot(
      x="Index",
      y="NMI",
      data=df_grove_insertion_quality,
      label="Grove",
      color="pink",
      ax=axs[0],
      linewidth=linewidth,
  )

  # Plot Del in the bottom subplot
  sns.scatterplot(
      x="Index",
      y="NMI",
      data=df_dyn_deletion_quality,
      label="DynHAC",
      color="orange",
      ax=axs[1],
      s=markersize,
      zorder=10,
  )
  sns.lineplot(
      x="Index",
      y="NMI",
      data=df_grinch_deletion_quality,
      label="Grinch",
      color="red",
      ax=axs[1],
      linewidth=linewidth,
  )

  font_size = 16  # Adjust the font size as needed

  # Set labels and legend
  axs[0].set_xlabel("")  # Remove x-label from top subplot
  axs[0].set_title("Insertions", fontsize=font_size)
  axs[1].set_title("Deletions", fontsize=font_size)
  axs[1].set_xlabel("# Points ", fontsize=font_size)
  # Increase font size for axis ticks

  for ax in axs:
    ax.set_ylabel("NMI", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.get_legend().remove()

  # Adjust layout
  plt.subplots_adjust(top=0.9)
  min_x = df_dyn_insertion_quality["Index"].min()
  max_x = df_dyn_insertion_quality["Index"].max()
  plt.xlim(min_x, max_x)
  plt.tight_layout()
  plt.savefig(base_dir + "plots/" + dataset + "_quality.png")
  plt.clf()


def plot_time(
    dataset,
    df_hac_time,
    df_dyn_insertion_time,
    df_dyn_deletion_time,
    df_grinch_inserton_time,
    df_grinch_deletion_time,
    df_grove_insertion_time,
):
  plt.clf()
  min_x = df_dyn_insertion_time["Index"].min()
  max_x = df_dyn_insertion_time["Index"].max()

  linewidth = 10
  fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=False)
  for ax in axs:
    sns.lineplot(
        x="Index",
        y="Clustering",
        data=df_hac_time,
        label="Static HAC",
        ax=ax,
        linewidth=linewidth,
    )
  sns.scatterplot(
      x="Index",
      y="Clustering",
      data=df_dyn_insertion_time,
      label="DynHAC",
      color="orange",
      s=10,
      ax=axs[0],
  )
  sns.lineplot(
      x="Index",
      y="Sliding_Window_Avg",
      data=df_dyn_insertion_time,
      color="orange",
      linewidth=3,
      ax=axs[0],
  )
  sns.scatterplot(
      x="Index",
      y="Clustering",
      data=df_dyn_deletion_time,
      label="DynHAC",
      color="orange",
      s=10,
      ax=axs[1],
  )
  sns.lineplot(
      x="Index",
      y="Sliding_Window_Avg",
      data=df_dyn_deletion_time,
      color="orange",
      linewidth=3,
      ax=axs[1],
  )

  sns.scatterplot(
      x="Index",
      y="Round",
      data=df_grinch_inserton_time,
      label="Grinch",
      color="red",
      s=10,
      ax=axs[0],
  )
  sns.lineplot(
      x="Index",
      y="Sliding_Window_Avg",
      data=df_grinch_inserton_time,
      color="red",
      linewidth=3,
      ax=axs[0],
  )
  sns.scatterplot(
      x="Index",
      y="Round",
      data=df_grinch_deletion_time,
      label="Grinch",
      color="red",
      s=10,
      ax=axs[1],
  )
  sns.lineplot(
      x="Index",
      y="Sliding_Window_Avg",
      data=df_grinch_deletion_time,
      color="red",
      linewidth=3,
      ax=axs[1],
  )
  sns.scatterplot(
      x="Index",
      y="Round",
      data=df_grove_insertion_time,
      label="Grove",
      color="pink",
      s=10,
      ax=axs[0],
  )
  sns.lineplot(
      x="Index",
      y="Sliding_Window_Avg",
      data=df_grove_insertion_time,
      color="pink",
      linewidth=3,
      ax=axs[0],
  )

  font_size = 16  # Adjust the font size as needed

  # Set labels and legend
  axs[0].set_xlabel("")  # Remove x-label from top subplot
  axs[0].set_title("Insertions", fontsize=font_size)
  axs[1].set_title("Deletions", fontsize=font_size)
  axs[1].set_xlabel("# Points ", fontsize=font_size)
  # Increase font size for axis ticks

  for ax in axs:
    ax.set_yscale("log")
    ax.set_ylabel("Clustering Time (s)", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)

  axs[0].legend(
      loc="upper center",
      bbox_to_anchor=(0.5, 1.3),
      prop={"size": font_size},
      markerscale=5,
      ncol=4,
  )
  axs[1].get_legend().remove()

  plt.xlim(min_x, max_x)
  plt.tight_layout()

  plt.savefig(base_dir + "plots/" + dataset + "_time.png")
  plt.clf()


def plot_mnist():

  dataset = "mnist"
  n = 70000
  data_sizes = {"mnist": 70000}

  epsilon = "0.1"

  df_hac_time = plot_io.get_hac_time(dataset)
  df_hac_quality = plot_io.get_hac_quality(dataset)
  df_dyn_insertion_time = plot_io.get_full_time(dataset, epsilon, n, True)
  plot_io.ComputeSlidingWindow(df_dyn_insertion_time, "Round")
  df_dyn_insertion_quality = plot_io.get_full_quality(dataset, epsilon, n, True)
  df_dyn_deletion_time = plot_io.get_full_time(dataset, epsilon, n, False)
  df_dyn_deletion_time["Index"] = 70000 - df_dyn_deletion_time["Index"]
  plot_io.ComputeSlidingWindow(df_dyn_deletion_time, "Clustering")
  df_dyn_deletion_quality = plot_io.get_full_quality(dataset, epsilon, n, False)
  (
      df_grinch_inserton_time,
      df_grinch_inserton_quality,
      df_grinch_deletion_time,
      df_grinch_deletion_quality,
  ) = plot_io.get_grinch_dfs(dataset)
  # only for mnist, wrong postprocessing was run
  # df_grinch_deletion_time["Index"] = n - df_grinch_deletion_time["Index"]

  (
      df_grove_insertion_time,
      df_grove_insertion_quality,
  ) = plot_io.get_grove_dfs(dataset)

  plot_quality(
      dataset,
      df_hac_quality,
      df_dyn_insertion_quality,
      df_dyn_deletion_quality,
      df_grinch_inserton_quality,
      df_grinch_deletion_quality,
      df_grove_insertion_quality,
  )
  plot_time(
      dataset,
      df_hac_time,
      df_dyn_insertion_time,
      df_dyn_deletion_time,
      df_grinch_inserton_time,
      df_grinch_deletion_time,
      df_grove_insertion_time,
  )


def plot_tail_time_quality(dataset, n):
  epsilon = "0.1"
  df_hac_time = plot_io.get_hac_time(dataset)
  df_hac_quality = plot_io.get_hac_quality(dataset)

  df_dyn_insertion_time = plot_io.get_tail_time(dataset, epsilon, n, True)
  df_dyn_insertion_time = df_dyn_insertion_time.iloc[1:, :]
  plot_io.ComputeSlidingWindow(df_dyn_insertion_time, "Round")
  df_dyn_insertion_quality = plot_io.get_tail_quality(dataset, epsilon, n, True)
  df_dyn_deletion_time = plot_io.get_tail_time(dataset, epsilon, n, False)
  df_dyn_deletion_time["Index"] = n - df_dyn_deletion_time["Index"]
  plot_io.ComputeSlidingWindow(df_dyn_deletion_time, "Clustering")
  df_dyn_deletion_quality = plot_io.get_tail_quality(dataset, epsilon, n, False)

  (
      df_grinch_inserton_time,
      df_grinch_inserton_quality,
      df_grinch_deletion_time,
      df_grinch_deletion_quality,
  ) = plot_io.get_grinch_dfs(dataset)
  # remove for new tail GRINCH time csvs directly written from tail
  # df_grinch_deletion_time["Index"] = n - df_grinch_deletion_time["Index"]

  (
      df_grove_insertion_time,
      df_grove_insertion_quality,
  ) = plot_io.get_grove_dfs(dataset)

  plot_time(
      dataset,
      df_hac_time,
      df_dyn_insertion_time,
      df_dyn_deletion_time,
      df_grinch_inserton_time,
      df_grinch_deletion_time,
      df_grove_insertion_time,
  )

  plot_quality(
      dataset,
      df_hac_quality,
      df_dyn_insertion_quality,
      df_dyn_deletion_quality,
      df_grinch_inserton_quality,
      df_grinch_deletion_quality,
      df_grove_insertion_quality,
  )


def plot_num_dirty_edges():

  times, num_dirty = plot_io.get_df_times_dirty_edges()

  fontsize = 20
  plt.clf()
  fig = plt.figure(figsize=(8, 6))
  plt.scatter(times, num_dirty)
  plt.xlabel("Time (s)", fontsize=fontsize)
  plt.ylabel("Num. Dirty Edges", fontsize=fontsize)
  plt.tick_params(axis="both", labelsize=fontsize)
  plt.tight_layout()
  plt.savefig(base_dir + "plots/mnist_time_num_edges.png")
  plt.clf()


def plot_num_rounds():
  indices, num_rounds = plot_io.get_df_num_rounds()

  fontsize = 20
  plt.clf()
  fig = plt.figure(figsize=(8, 6))
  plt.scatter(indices, num_rounds)
  plt.xscale("log")
  plt.xticks([1000, 10000, 70000], ["1000", "10000", "70000"])
  plt.xlabel("# Points ", fontsize=fontsize)
  plt.ylabel("Num. Rounds", fontsize=fontsize)
  plt.tick_params(axis="both", labelsize=fontsize)
  plt.tight_layout()
  plt.savefig(base_dir + "plots/mnist_num_rounds.png")
  plt.clf()


def plot_epsilons_single(dataset, n, insertion=True):
  plt.clf()
  font_size = 20  # Adjust the font size as needed

  markerstyles = {
      "0.0": "o",
      "0.1": "s",
      "1.0": "D",
  }

  mode = ""
  if not insertion:
    mode = "_deletion"

  fig = plt.figure(figsize=(8, 6))
  for epsilon in ["0.0", "0.1", "1.0"]:
    df_quality = plot_io.get_tail_quality(dataset, epsilon, n, insertion)

    sns.lineplot(
        x="Index", y="NMI", data=df_quality, label=epsilon, linewidth=10
    )

  # Set labels and legend
  plt.xlabel("Num. Points", fontsize=font_size)
  plt.ylabel("NMI", fontsize=font_size)
  plt.tick_params(axis="both", labelsize=font_size)

  plt.legend(
      prop={"size": font_size},
      markerscale=5,
  )

  plt.tight_layout()
  plt.savefig(base_dir + f"plots/{dataset}{mode}_epsilons_quality.png")
  plt.clf()

  plt.clf()
  fig = plt.figure(figsize=(8, 6))
  for epsilon in ["0.0", "0.1", "1.0"]:
    df_time = plot_io.get_tail_time(dataset, epsilon, n, insertion)
    df_time = df_time.iloc[1:, :]
    plot_io.ComputeSlidingWindow(df_time, "Clustering")

    sns.scatterplot(
        x="Index",
        y="Clustering",
        data=df_time,
        label=epsilon,
        s=10,
        marker=markerstyles[epsilon],
    )
    sns.lineplot(x="Index", y="Sliding_Window_Avg", data=df_time, linewidth=5)
  plt.xlabel("Num. Points", fontsize=font_size)
  plt.ylabel("Clustering Time (s)", fontsize=font_size)
  plt.tick_params(axis="both", labelsize=font_size)

  plt.legend().set_visible(False)
  plt.yscale("log")

  if dataset == "mnist" and not insertion:
    plt.yticks(
        [0.02, 0.05, 0.09, 0.2, 0.5], ["0.02", "0.05", "0.09", "0.2", "0.5"]
    )
  elif dataset == "mnist":
    plt.yticks(
        [0.003, 0.006, 0.008, 0.015, 0.03], ["0.003", "0.006", "0.008", "0.015", "0.03"]
    )
  elif dataset == "aloi":
    plt.xticks(rotation=20)
    plt.yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])

  plt.tight_layout()
  plt.savefig(base_dir + f"plots/{dataset}{mode}_epsilons.png")
  plt.clf()


def plot_bars():
  plt.clf()
  font_size = 16

  data_sizes = {"mnist": 70000, "aloi": 108000, "ilsvrc_small": 50000}
  ins_lst = []
  del_lst = []

  colors = {
      "Static HAC": "blue",
      "DynHAC": "orange",
      "GRINCH": "red",
      "Grove": "pink",
  }

  for dataset, n in data_sizes.items():
    tmp_ins, tmp_del = plot_io.get_bar_dataframe(dataset, n)
    ins_lst.append(tmp_ins)
    del_lst.append(tmp_del)

  df_ins = pd.concat(ins_lst)
  df_del = pd.concat(del_lst)

  print(df_ins.set_index(["Dataset"]).to_latex())
  print(df_del.set_index(["Dataset"]).to_latex())

  fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
  sns.barplot(
      x="Dataset",
      y="Speedup",
      hue="Algorithm",
      ax=axs[0],
      data=df_ins[df_ins["Algorithm"] != "Static HAC"],
      palette=colors,
  )
  sns.barplot(
      x="Dataset",
      y="Speedup",
      hue="Algorithm",
      ax=axs[1],
      data=df_del[df_del["Algorithm"] != "Static HAC"],
      palette=colors,
  )

  axs[0].set_xlabel("")  # Remove x-label from top subplot
  axs[0].set_title("Insertions", fontsize=font_size)
  axs[1].set_title("Deletions", fontsize=font_size)
  axs[1].set_xlabel("Dataset", fontsize=font_size)

  for ax in axs:
    ax.axhline(y=1, linestyle=":", linewidth=5, label="Static HAC")
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=font_size)
    ax.set_ylabel("Speedup over Static HAC", fontsize=font_size)

  axs[0].legend(
      loc="upper center",
      bbox_to_anchor=(0.5, 1.3),
      prop={"size": font_size},
      markerscale=5,
      ncol=4,
  )
  axs[1].get_legend().remove()

  plt.tight_layout()
  plt.savefig(base_dir + "plots/time_bar.png")
  plt.clf()

  fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)

  sns.barplot(
      x="Dataset", y="NMI", hue="Algorithm", data=df_ins, palette=colors, ax=ax
  )
  ax.set_ylabel("NMI", fontsize=font_size)

  ax.set_xlabel("Dataset", fontsize=font_size)
  ax.tick_params(axis="both", labelsize=font_size)
  plt.legend(
      loc="upper center",
      bbox_to_anchor=(0.5, 1.3),
      prop={"size": font_size},
      markerscale=5,
      ncol=4,
  )

  plt.tight_layout()
  plt.savefig(base_dir + "plots/nmi_bar.png")
  plt.clf()


def plot_bars_epsilon():
  plt.clf()
  font_size = 16

  data_sizes = {"mnist": 70000, "aloi": 108000, "ilsvrc_small": 50000}
  ins_lst = []
  del_lst = []

  colors = {"0.0": "green", "0.1": "orange", "1.0": "purple"}

  for dataset, n in data_sizes.items():
    tmp_ins, tmp_del = plot_io.get_bar_dataframe_epsilon(dataset, n)
    ins_lst.append(tmp_ins)
    del_lst.append(tmp_del)

  df_ins = pd.concat(ins_lst)
  df_del = pd.concat(del_lst)

  print(df_ins.set_index(["Dataset"]).to_latex())
  print(df_del.set_index(["Dataset"]).to_latex())

  fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
  sns.barplot(
      x="Dataset",
      y="Speedup",
      hue="epsilon",
      data=df_ins,
      ax=axs[0],
      palette=colors,
  )
  sns.barplot(
      x="Dataset",
      y="Speedup",
      hue="epsilon",
      data=df_del,
      ax=axs[1],
      palette=colors,
  )

  axs[0].set_xlabel("")  # Remove x-label from top subplot
  axs[0].set_title("Insertions", fontsize=font_size)
  axs[1].set_title("Deletions", fontsize=font_size)
  axs[1].set_xlabel("Dataset", fontsize=font_size)

  axs[0].legend(
      loc="upper center",
      bbox_to_anchor=(0.5, 1.3),
      prop={"size": font_size},
      markerscale=5,
      ncol=4,
  )
  axs[1].get_legend().remove()

  for ax in axs:
    ax.tick_params(axis="both", labelsize=font_size)
    ax.set_ylabel("Speedup over epsilon=0", fontsize=font_size)

  plt.tight_layout()
  plt.savefig(base_dir + "plots/time_bar_epsilon.png")
  plt.clf()

  fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

  sns.barplot(
      x="Dataset",
      y="NMI",
      hue="epsilon",
      data=df_ins,
      palette=colors,
      ax=axs[0],
  )
  sns.barplot(
      x="Dataset",
      y="NMI",
      hue="epsilon",
      data=df_del,
      palette=colors,
      ax=axs[1],
  )
  axs[0].set_title("Insertions", fontsize=font_size)
  axs[1].set_title("Deletions", fontsize=font_size)
  for ax in axs:
    ax.set_ylabel("NMI", fontsize=font_size)
    ax.set_ylim((0.85, 0.9))

    ax.set_xlabel("Dataset", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.get_legend().remove()

  plt.tight_layout()
  plt.savefig(base_dir + "plots/nmi_bar_epsilon.png")
  plt.clf()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  # plot_mnist()
  # plot_tail_time_quality("aloi", 108000)
  # plot_tail_time_quality("ilsvrc_small", 50000)
  # plot_num_dirty_edges()
  # plot_num_rounds()
  for mode in [True, False]: #
    plot_epsilons_single("mnist", 70000, mode)
    # plot_epsilons_single("aloi", 108000, mode)
    # plot_epsilons_single("ilsvrc_small", 50000, mode)

  # plot_bars()
  # plot_bars_epsilon()


if __name__ == "__main__":
  app.run(main)
