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

"""Run dynamic hac experiments."""

from collections.abc import Sequence
import subprocess
import os

from absl import app
from absl import flags
import evaluate_utils

_GROUND_TRUTH = flags.DEFINE_string(
    "ground_truth",
    default=None,
    help="binary file to read for ground truth.",
)

_INPUT_DATA = flags.DEFINE_string(
    "input_data",
    default=None,
    help="Input data in fvecs format.",
)


_CLUSTERING = flags.DEFINE_string(
    "clustering", default=None, help="file to read for clustering."
)


_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", default=None, help="output file."
)

_WEIGHT = flags.DEFINE_float("weight", default=0.1, help="weight.")

_EPSILON = flags.DEFINE_float("epsilon", default=0.1, help="epsilon.")

_K = flags.DEFINE_integer("k", default=50, help="k.")

_MAX_DEGREE = flags.DEFINE_integer(
    "max_degree", default=100, help="max_degree."
)

_L = flags.DEFINE_integer("l", default=100, help="l.")

_NUM_BATCH = flags.DEFINE_integer("num_batch", default=100, help="num_batch.")

_RUN_HAC = flags.DEFINE_bool(
    "run_hac",
    default=True,
    help="if false, use the existing output file without running hac.",
)

_METHOD = flags.DEFINE_string(
    "method", default="parhac", help="which hac to use. parhac or dynamic_hac or dynamic_hac_deletion"
)

_OUTPUT_KNN = flags.DEFINE_string(
    "output_knn", default="", help="output file direction for knn."
)

_USE_OUTPUT_KNN = flags.DEFINE_bool(
    "use_output_knn", default=False, help="if true, use output_knn."
)

_FIRST_BATCH_RATIO = flags.DEFINE_float(
    "first_batch_ratio", default=-1, help="first batch ratio."
)

_STORE_BATCH_SIZE = flags.DEFINE_integer(
    "store_batch_size", default=-1, help="num batch to store."
)


class Config:
  """configs for ParHac experiments on metric data."""

  def __init__(
      self,
      epsilon=0.1,
      weight_threshold=0.1,
      k=50,
      max_degree=100,
      l=100,
      num_batch=100,
  ):
    self.epsilon = epsilon
    self.weight_threshold = weight_threshold
    self.k = k
    self.max_degree = max_degree
    self.l = l
    self.num_batch = num_batch


def run_experiment(
    log_file_name,
    input_data,
    output_clustering_dir,
    config,
):
  """run experiments."""
  exp_root = os.environ.get('EXP_ROOT')
  base_dir = exp_root + "/bazel-bin/parclusterer_exp/benchmark/"
  os.environ["PARLAY_NUM_THREADS"] = "1"
  if _METHOD.value == "dynamic_hac_deletion":
    command = [
        base_dir + "deletion_main",
        "--input_data=" + input_data,
        "--output_clustering=" + output_clustering_dir,
        "--k=" + str(config.k),  # Convert numbers to strings
        "--max_degree=" + str(config.max_degree),
        "--l=" + str(config.l),
        "--epsilon=" + str(config.epsilon),
        "--weight_threshold=" + str(config.weight_threshold),
        "--store_batch_size=" + str(_STORE_BATCH_SIZE.value),
        "--output_knn=" + _OUTPUT_KNN.value,
        "--use_output_knn=" + str(_USE_OUTPUT_KNN.value),
        "--early_stop_ratio=" + str(_FIRST_BATCH_RATIO.value),
    ]
  else:
    command = [
        base_dir + "parhac_main",
        "--input_data=" + input_data,
        "--output_clustering=" + output_clustering_dir,
        "--k=" + str(config.k),  # Convert numbers to strings
        "--max_degree=" + str(config.max_degree),
        "--l=" + str(config.l),
        "--epsilon=" + str(config.epsilon),
        "--weight_threshold=" + str(config.weight_threshold),
        "--num_batch=" + str(config.num_batch),
        "--method=" + _METHOD.value,
        "--output_knn=" + _OUTPUT_KNN.value,
        "--use_output_knn=" + str(_USE_OUTPUT_KNN.value),
        "--first_batch_ratio=" + str(_FIRST_BATCH_RATIO.value),
        "--store_batch_size=" + str(_STORE_BATCH_SIZE.value),
    ]

  print("Command running:")
  print(" ".join(command), " > ", log_file_name)
  print()

  create_dirs = [os.path.dirname(f) for f in [log_file_name, output_clustering_dir, _OUTPUT_KNN.value]]

  for directory in create_dirs:
    if directory and not os.path.exists(directory):
      print("creating directory, ", directory)
      os.makedirs(directory)

  with open(log_file_name, "w") as log_file:
    try:
      process = subprocess.Popen(
          command,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          bufsize=1,  # Line-buffered, flush after each line
      )
      if not process or not process.stdout:
        raise (subprocess.CalledProcessError)

      output_text = ""
      for line in iter(process.stdout.readline, ""):
        log_file.write(line)  # Write output to log file
        log_file.flush()  # Flush the buffer to the file
        output_text += line

      process.wait()  # Wait for the subprocess to finish
      print("Command completed successfully.")
      # return result.stdout
      return output_text
    except subprocess.CalledProcessError as e:
      print("Command failed with error:", e.returncode)
      print("Error output:", e.stdout)
      print(f"Command failed with error: {e}")
      return f"Command failed with error: {e}"
  return "Command completed successfully."


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = Config(
      epsilon=_EPSILON.value,
      weight_threshold=_WEIGHT.value,
      k=_K.value,
      max_degree=_MAX_DEGREE.value,
      l=_L.value,
      num_batch=_NUM_BATCH.value,
  )
  input_data = _INPUT_DATA.value
  ground_truth = _GROUND_TRUTH.value
  output_clustering_dir = f"{_CLUSTERING.value}_{config.epsilon}_{config.weight_threshold}_{config.num_batch}"

  log_file_name = f"{_OUTPUT_FILE.value}/log_eps_{config.epsilon}_weight_{config.weight_threshold}_{config.num_batch}.txt"
  ari_plot = f"{_OUTPUT_FILE.value}/fig_eps_{config.epsilon}_weight_{config.weight_threshold}_{config.num_batch}"
  runtime_plot = ari_plot + ".png"

  if _RUN_HAC.value:
    log_text = run_experiment(
        log_file_name, input_data, output_clustering_dir, config
    )
    if "Command failed with error" in log_text:
      print(log_text)
      return
  else:
    with open(log_file_name, "r") as log_file:
      log_text = log_file.read()

  if _METHOD.value == "dynamic_hac_deletion":
    evaluate_utils.plot_running_times_deletion(log_text, runtime_plot)
  else:
    evaluate_utils.plot_running_times(log_text, runtime_plot)
  evaluate_utils.evaluate(
      ground_truth,
      log_text,
      output_clustering_dir,
      ari_plot,
      config.weight_threshold,
  )


if __name__ == "__main__":
  flags.mark_flags_as_required(
      ["clustering", "ground_truth", "input_data", "output_file"]
  )
  app.run(main)
