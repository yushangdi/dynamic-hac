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

"""Compute numbers from experimental results."""

from collections.abc import Sequence
import re

from absl import app
import jinja2  # dependency for pringing latex table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_io
import seaborn as sns


def get_max_ari_diff(df_hac_quality, df):
  max_diff = 0
  for _, row in df_hac_quality.iterrows():
    index = int(row["Index"])
    static_ari = row["NMI"]
    try:
      dyn_ari = plot_io.get_df_value(df, "NMI", index)
      max_diff = max(max_diff, static_ari - dyn_ari)
    except Exception as e:
      print(index)
  return max_diff


def compute_ari_diff():
  epsilon = "0.1"
  n = 70000
  df_hac_quality = plot_io.get_hac_quality("mnist")
  df_dyn_insertion_quality = plot_io.get_full_quality("mnist", epsilon, n, True)
  df_dyn_deletion_quality = plot_io.get_full_quality("mnist", epsilon, n, False)

  max_ins = get_max_ari_diff(df_hac_quality, df_dyn_insertion_quality)
  max_del = get_max_ari_diff(df_hac_quality, df_dyn_deletion_quality)
  print(max_ins, max_del)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  compute_ari_diff()


if __name__ == "__main__":
  app.run(main)
