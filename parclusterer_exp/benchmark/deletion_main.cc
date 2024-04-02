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

#include <stdbool.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "parclusterer_exp/benchmark/edge_feeder.h"
#include "parclusterer_exp/benchmark/io.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac.pb.h"
#include "in_memory/clustering/dynamic/hac/hac.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "utils/timer.h"
#include "parlay/parallel.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/point_range.h"

ABSL_FLAG(std::string, input_data, "", "Input points in fvecs format");
ABSL_FLAG(double, epsilon, 0.1, "epsilon");
ABSL_FLAG(double, weight_threshold, 0.1, "weight_threshold");
ABSL_FLAG(int, max_degree, 100, "max degree");
ABSL_FLAG(int, l, 100, "l");
ABSL_FLAG(int, k, 3, "k used in k nearest neighbors");

ABSL_FLAG(int, num_batch, 100, "number of batches");

ABSL_FLAG(
    std::string, output_clustering, "",
    "Output filename of a clustering (SSTable of node ID --> cluster "
    "ID). May be sharded (e.g., \"MyOutput@100\" leads to a 100-shard output "
    "file). If the filename contains '<level>' as a substring, "
    "HierarchicalFlatCluster() is used to compute the clustering. "
    "In this case, if the flag value is /path/clustering-L<level>, and "
    "the computed hierarchy has two elements, they will be written to "
    "/path/clustering-L0 and /path/clustering-L1.");

ABSL_FLAG(std::string, output_knn, "", "knn location in SSTable");

ABSL_FLAG(bool, use_output_knn, false,
          "if true, use the knn edges stored in output_knn.");

ABSL_FLAG(double, early_stop_ratio, 0, "stop at ratio * n.");

ABSL_FLAG(int, store_batch_size, -1,
          "Let this value be x. We store at index i mod x == 0. Default value "
          "is n/100.");

namespace graph_mining {
namespace in_memory {
namespace {
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
using T = float;
using Point = Euclidian_Point<T>;
using PointRange = PointRange<T, Point>;
using indexType = unsigned int;
using graph_mining::in_memory::Dendrogram;

DynamicHacConfig DynamicHacConfig(double epsilon, double weight_threshold) {
  class DynamicHacConfig dynamic_hac_config;
  dynamic_hac_config.set_epsilon(epsilon);
  dynamic_hac_config.set_weight_threshold(weight_threshold);
  return dynamic_hac_config;
}

std::vector<std::size_t> InitialDegrees(std::vector<AdjacencyList>& edges) {
  std::vector<std::size_t> degrees;
  degrees.reserve(edges.size());
  for (int i = 0; i < edges.size(); ++i) {
    degrees.push_back(edges[i].outgoing_edges.size());
  }
  return degrees;
}

absl::Status Main() {
  QCHECK(!absl::GetFlag(FLAGS_input_data).empty());
  // graph_mining::in_memory::ParallelSchedulerReference scheduler;
  std::string input_data = absl::GetFlag(FLAGS_input_data);
  std::string output_clustering_path = absl::GetFlag(FLAGS_output_clustering);
  double epsilon = absl::GetFlag(FLAGS_epsilon);
  double weight_threshold = absl::GetFlag(FLAGS_weight_threshold);
  int maxDeg = absl::GetFlag(FLAGS_max_degree);
  int L = absl::GetFlag(FLAGS_l);
  int k = absl::GetFlag(FLAGS_k);
  // int num_batch = absl::GetFlag(FLAGS_num_batch);
  int num_knn_batch = 100;
  double alpha = 1.2;
  bool use_output_knn = absl::GetFlag(FLAGS_use_output_knn);
  std::string knn_output_file = absl::GetFlag(FLAGS_output_knn);
  double early_stop_ratio = absl::GetFlag(FLAGS_early_stop_ratio);
  CHECK_LT(early_stop_ratio, 1.0);
  CHECK_GE(early_stop_ratio, 0.0);

  std::cout << "method: dynhac deletion \n";
  std::cout << "thread: " << parlay::num_workers() << "\n";
  std::cout << "epsilon: " << epsilon << "\n";
  std::cout << "weight threshold: " << weight_threshold << "\n";
  std::cout << "max degree: " << maxDeg << "\n";
  std::cout << "L: " << L << "\n";
  std::cout << "k: " << k << "\n";
  // std::cout << "num batch: " << num_batch << "\n";
  std::cout << "early stop ratio: " << early_stop_ratio << "\n";
  std::cout << "num knn batch: " << num_knn_batch << "\n";
  std::cout << "File: " << input_data << std::endl;

  // Read points and initialize Vamana graph for ANNS.
  auto points = BuildPointRange<T>(input_data);
  std::size_t early_stop_index = points.size() * early_stop_ratio;
  std::cout << "early stop index: " << early_stop_index << "\n";

  std::size_t store_result_batch = points.size() / 100;
  if (absl::GetFlag(FLAGS_store_batch_size) != -1) {
    store_result_batch = absl::GetFlag(FLAGS_store_batch_size);
  }
  std::cout << "store batch size: " << store_result_batch << "\n";

  std::vector<std::string> node_id_map(points.size());
  parlay::parallel_for(0, points.size(), [&](std::size_t i) {
    node_id_map[i] = std::to_string(i);
  });
  std::cout << "read data\n";

  WallTimer subtimer;
  subtimer.Restart();
  std::vector<AdjacencyList> edges;
  if (use_output_knn) {
    auto edge_reader =
        EdgeReader<T>(points.size(), num_knn_batch, knn_output_file, 0);
    while (true) {
      ASSIGN_OR_RETURN(const auto& result, edge_reader.NextBatch());
      if (result.empty()) break;
      edges.insert(edges.end(), result.begin(), result.end());
    }
  } else {
    auto edge_feeder =
        EdgeFeeder<T>(k, points, num_knn_batch, maxDeg, L, alpha, 0);
    while (true) {
      ASSIGN_OR_RETURN(const auto& result, edge_feeder.NextBatch());
      if (result.empty()) break;
      edges.insert(edges.end(), result.begin(), result.end());
    }
  }
  std::cout << "KNN time: " << subtimer.GetSeconds() << " seconds\n";
  subtimer.Restart();

  auto dynamic_hac_config = DynamicHacConfig(epsilon, weight_threshold);
  auto dynamic_hac_clusterer = DynamicHacClusterer(dynamic_hac_config);
  RETURN_IF_ERROR(FixAdjList(edges));
  std::cout << "FixAdjList time: " << subtimer.GetSeconds() << " seconds\n";
  subtimer.Restart();

  ASSIGN_OR_RETURN(const auto stats, dynamic_hac_clusterer.Insert(edges));
  std::cout << "Num. Rounds: " << dynamic_hac_clusterer.NumRounds() << "\n";
  ASSIGN_OR_RETURN(const auto& result,
                   dynamic_hac_clusterer.Dendrogram().ConvertToDendrogram());
  Dendrogram dendrogram = std::move(result.first);

  std::cout << "Batch cluster time: " << subtimer.GetSeconds() << " seconds\n";
  RETURN_IF_ERROR(WriteDendrogram(
      dendrogram, absl::StrCat(output_clustering_path, "-all-dendro.bin")));

  WallTimer timer;
  timer.Restart();
  for (NodeId current_index = points.size() - 1;
       current_index >= early_stop_index; --current_index) {
    const auto num_nodes_left = current_index;
    const auto num_nodes_removed = points.size() - num_nodes_left;
    std::cout << "====== index: " << num_nodes_removed << "\n";
    std::cout << "removing: " << current_index << "\n";

    subtimer.Restart();
    ASSIGN_OR_RETURN(const auto stats, dynamic_hac_clusterer.Remove({current_index}));
    std::cout << "Clustering time: " << subtimer.GetSeconds() << " seconds\n";
    std::cout << "Num. Rounds: " << dynamic_hac_clusterer.NumRounds() << "\n";
    stats.LogStats();

    if (num_nodes_left % store_result_batch == 0 ||
        num_nodes_left == early_stop_index) {
      if (!output_clustering_path.empty()) {
        std::cout << "Store index: " << num_nodes_left << "\n";
        ASSIGN_OR_RETURN(
            const auto& result,
            dynamic_hac_clusterer.Dendrogram().ConvertToDendrogram());
        dendrogram = std::move(result.first);

        RETURN_IF_ERROR(WriteDendrogram(
            dendrogram, absl::StrCat(output_clustering_path, "-",
                                     num_nodes_left, "-dendro.bin")));

        std::cout << "Store time: " << timer.GetSeconds() << " seconds\n";
        timer.Restart();
      }
    }

    timer.Restart();
  }

  return absl::OkStatus();
}
}  // namespace
}  // namespace in_memory
}  // namespace graph_mining

int main(int argc, char* argv[]) { QCHECK_OK(graph_mining::in_memory::Main()); }
