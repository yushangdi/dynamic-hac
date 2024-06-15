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

"""Evaluate the clustering result."""

from collections.abc import Sequence

from absl import app
from absl import flags
import evaluate_utils
import pandas as pd

_GROUND_TRUTH = flags.DEFINE_string(
    'ground_truth',
    default=None,
    help='Colossus binary file to read for ground truth.',
)

_LOG_FILE = flags.DEFINE_string('log_file', default=None, help='log file.')

_CLUSTERING = flags.DEFINE_string(
    'clustering', default=None, help='Colossus SSTable to read for clustering.'
)

_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', default=None, help='output file.'
)

_PLOT_ONLY = flags.DEFINE_bool('plot_only', default=False, help='plot only.')

_THRESHOLD = flags.DEFINE_float('threshold', default=0.1, help='threshold.')


def main(argv):

  if _PLOT_ONLY.value:
    df = pd.read_csv(_OUTPUT_FILE.value + '_ari.csv')
    evaluate_utils.plot(
        df['Index'], df['ARI'], df['Num_Clusters'], _OUTPUT_FILE.value
    )
    return

  with open(_LOG_FILE.value, 'r') as file:
    text = file.read()

  evaluate_utils.evaluate(
      _GROUND_TRUTH.value,
      text,
      _CLUSTERING.value,
      _OUTPUT_FILE.value,
      _THRESHOLD.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_file'])
  app.run(main)
