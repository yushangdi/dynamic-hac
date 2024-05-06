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

"""Embeds MNIST data into two dimensions by UMAP."""

from collections.abc import Sequence
import os 

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
import umap

_PLOT = flags.DEFINE_bool(
    "plot", default=False, help="Plot the embedded data."
)

_OUTPUT = flags.DEFINE_string("output", default=None, help="Output directory.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  plot = _PLOT.value
  output_dir = _OUTPUT.value
  mnist = fetch_openml("mnist_784", version=1)

  reducer = umap.UMAP(random_state=42)
  embedding = reducer.fit_transform(mnist.data)
  labels = mnist.target.astype(int)
  if not os.path.exists(output_dir):
    print("creating directory, ", output_dir)
    os.makedirs(output_dir)
  np.save(output_dir + "/mnist_embed.gt", labels)
  np.save(output_dir + "/mnist_embed.npy", embedding)

  if plot:
    sns.set(context="paper", style="white")
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=0.1
    )
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
    plt.savefig(
        output_dir + "/mnist_embed.png",
    )

if __name__ == "__main__":
  flags.mark_flags_as_required(["output"])
  app.run(main)
