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

"""Process output log and plot the running times."""

from collections.abc import Sequence

from absl import app
from absl import flags
import evaluate_utils


_INPUT_FILE = flags.DEFINE_string("input_file", default=None, help="log file.")

_OUTPUT_FILE = flags.DEFINE_string(
    "output_file", default=None, help="output file."
)


def main(unused_argv):
  # input file, output file, data set
  with open(_INPUT_FILE.value, "r") as file:
    text = file.read()

  if "deletion" in _INPUT_FILE.value:
    evaluate_utils.plot_running_times_deletion(text, _OUTPUT_FILE.value)
  else:
    evaluate_utils.plot_running_times(text, _OUTPUT_FILE.value)


if __name__ == "__main__":
  flags.mark_flags_as_required(["input_file", "output_file"])
  app.run(main)
