// Copyright 2024 The Approximate Hac Experiments Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "parclusterer_exp/benchmark/io.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace py = ::pybind11;

using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
using graph_mining::in_memory::Dendrogram;

namespace graph_mining::in_memory {

Dendrogram ReadDendrogramWrapper(const std::string& dendrogram_file) {
  const auto& result = ReadDendrogram(dendrogram_file);
  if (!result.ok()) {
    LOG(ERROR) << result.status();
    return Dendrogram(0);
  }
  return result.value();
}

// Returns the flat clustering.
Clustering CutDendrogramAt(const Dendrogram& dendrogram, double threshold) {
  const auto result = dendrogram.FlattenSubtreeClustering(threshold);
  if (!result.ok()) {
    LOG(ERROR) << result.status();
    return Clustering();
  }
  return result.value();
}
}  // namespace graph_mining::in_memory

namespace {
PYBIND11_MODULE(cut_dendrogram, m) {
  m.doc() = "Pybind11 bindings for cut dendrogram";

  m.def("CutDendrogramAt", &graph_mining::in_memory::CutDendrogramAt,
        "Returns the flat clustering for a given dendrogram and a threshold to "
        "cut.",
        py::arg("dendrogram"), py::arg("threshold"));

  m.def("ReadDendrogram", &graph_mining::in_memory::ReadDendrogramWrapper,
        "Reads a dendrogram from a file.", py::arg("dendrogram_file"));
}
}  // namespace
