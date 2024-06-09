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

"""Convert csv files to fvecs and read fvecs file."""

from collections.abc import Sequence
from io import StringIO
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.datasets import fetch_covtype

_DATA_FILE = flags.DEFINE_string(
    "data_file",
    default=None,
    help="Input data file.",
)

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    default=None,
    help="Output prefix for fves points and labels.",
)

_DATA = flags.DEFINE_string("data", default=None, help="Name of data set.")

_PERMUTE = flags.DEFINE_bool("permute", default=True, help="Permute the data randomly if true.")

_SORT = flags.DEFINE_bool("sort", default=False, help="Sort the points by labels if true.")

class Error(Exception):
  """Base class for exceptions in this module."""


def load_data_to_numpy_array(input_file, dim=128):
  """Loads data from the specified input file into a NumPy array.

  Args:
      input_file (str): Path to the input file.
      dim (int, optional): Dimensionality of the vectors. Defaults to 128.

  Returns:
      numpy.ndarray: The NumPy array containing the loaded data. The first
      column is label.
  """

  data = []
  with open(input_file, "r") as fin:
    for line in fin:
      splt = line.strip().split(" ")
      vec = np.zeros(dim + 1)
      vec[0] = int(splt[0])  # the label, must be an integer
      for i in range(1, len(splt)):
        if len(splt[i]) > 0:
          dim_val = splt[i].split(":")
          if len(dim_val) != 2:
            print(dim_val)
          vec[int(dim_val[0])] = float(dim_val[1])
      data.append(vec)

  return np.array(data)


def write_file(colossus_pathname, contents):
  """Writes contents to a new Colossus pathname.

  Args:
    colossus_pathname: The Colossus pathname of the output file to write to.
    contents: The contents to write to the output file.
  """
  directory = os.path.dirname(colossus_pathname)
  if not os.path.exists(directory):
    print("creating directory, ", directory)
    os.makedirs(directory)
  with open(colossus_pathname, "wb") as output_file:
    output_file.write(contents)


def read_file(colossus_pathname, mode = "r"):
  """Returns the contents of a Colossus file.

  Args:
    colossus_pathname: The Colossus pathname of the file to read.

  Returns:
    The file contents of `colossus_pathname`.

  Raises:
    ColossusFileError: `colossus_pathname` could not be opened.
  """
  try:
    with open(colossus_pathname, mode) as input_file:
      return input_file.read()
  except Exception as e:
    log_pattern, *log_args = (
        "Cannot open Colossus pathname %s: %r",
        colossus_pathname,
        e,
    )
    # Logging functions should receive a pattern-string plus args,
    # since some loggers may enter parameters into a database where
    # log messages are searchable also by pattern.
    logging.error(log_pattern, *log_args)
    raise Error(log_pattern % tuple(log_args)) from e


# store float32 points into fvecs format
# points are arrays
def store_points(points, output_file):
  d = len(points[0])
  points = points.astype(np.float32)
  points = points.view(np.int32)
  # insert d to the beginning and flatten points
  points_flatten = np.insert(points, 0, d, axis=1).flatten()
  # points_flatten.tofile(output_file + ".fvecs")
  points_bytes = points_flatten.tobytes()
  write_file(output_file + ".fvecs", points_bytes)
  print("stored points to ", output_file + ".fvecs")
  return points_flatten


def store(input_file, output_file, dataset, permute=True, sort=False):
  """store points from input_file to fvecs format.

  Args:
      input_file (str): Path to the input file containing the points.
      output_file (str): Path to the output file where the fvecs data will be
        stored.
      dataset (str): Name of the dataset to be processed. Currently supports
        "aloi".
      permute (bool, optional): If True, randomly permutes the order of the
        points before storing. Defaults to True. Only one of `permute` and `sort` 
        should be True.
      sort (bool, optional): If True, sort by cluster ids. Defaults to False. 
        Only one of `permute` and `sort` should be True.

  Raises:
      KeyError: If an unknown dataset is specified.

  Returns:
      None
  """
  # points = np.loadtxt(input_file, delimiter="\t").astype(np.float32)
  if dataset == "covtype":
    cov_type = fetch_covtype(random_state=42, shuffle=permute)
    if permute:
      print("permuting")
      permute = False  # has been permuted
    # remove soil categorical features
    points = cov_type.data.astype(np.float32)  # [:, 0:10]
    # normalize
    points = preprocessing.MinMaxScaler().fit_transform(points)
    labels = cov_type.target
    points = np.hstack((labels[:, None], points)).astype(np.float32)
  elif dataset == "ilsvrc_small":
    # with open(input_file, "r") as f:
    #   points = np.loadtxt(f, delimiter="\t").astype(np.float32)
    points = np.load(input_file).astype(np.float32)
    points = points[:, 1:]
  elif dataset == "aloi":
    points = load_data_to_numpy_array(input_file).astype(np.float32)
  elif dataset == "mnist":
    points = np.load(input_file + "/mnist_embed.npy").astype(np.float32)
    labels = np.load(input_file + "/mnist_embed.gt.npy").astype(np.float32)
    points = np.hstack((labels[:, None], points))
  elif dataset == "iris":
    iris = datasets.load_iris()
    points = iris.data
    labels = iris.target
    points = np.hstack((labels[:, None], points)).astype(np.float32)
  else:
    raise KeyError("Unknown dataset")
  if permute:
    print("permuting")
    # Generate random indices
    permuted_indices = np.random.default_rng(seed=42).permutation(
        points.shape[0]
    )
    points = points[permuted_indices]  # Permute the rows using the indices
  elif sort:
    print("sorting")
    points = points[points[:,0].argsort()]
  points, labels = preprocessing_data(points)
  print("points shape: ", points.shape)
  print("labels shape: ", labels.shape)
  print("num class: ", len(np.unique(labels)))
  stored_points = store_points(points, output_file)

  print("labels", labels)
  labels_bytes = labels.astype(int).tobytes()
  write_file(output_file + "_label.bin", labels_bytes)
  print("stored labels to ", output_file + "_label.bin")
  return stored_points


def ivecs_read(fname):
  a = np.fromfile(fname, dtype="int32")
  d = a[0]
  return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
  return ivecs_read(fname).view("float32")


def preprocessing_data(points):
  # remove second column, which is label
  labels = points[:, 0]
  points = points[:, 1:]
  return points, labels


def main(argv):
  # input file, output file, data set
  store(_DATA_FILE.value, _OUTPUT_FILE.value, _DATA.value, _PERMUTE.value, _SORT.value)


if __name__ == "__main__":
  flags.mark_flags_as_required(["data_file", "output_file", "data"])
  app.run(main)
