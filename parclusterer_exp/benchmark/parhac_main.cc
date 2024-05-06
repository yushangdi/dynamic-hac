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
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "parclusterer_exp/benchmark/edge_feeder.h"
#include "parclusterer_exp/benchmark/io.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac.pb.h"
#include "in_memory/clustering/hac/parhac.h"
#include "in_memory/clustering/dynamic/hac/hac.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
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

ABSL_FLAG(int, store_batch_size, -1,
          "Let this value be x. We store at index i mod x == 0. Default value "
          "is n/100.");

ABSL_FLAG(std::string, method, "parhac", "method to use");

// ABSL_FLAG(int, first_batch_size, 1000, "first batch size");

ABSL_FLAG(double, first_batch_ratio, 0,
          "first batch size be ratio * n. Ignored by parhac method, which "
          "takes value 0.");

ABSL_FLAG(
    std::string, output_clustering, "",
    "Output filename of a clustering (SSTable of node ID --> cluster "
    "ID). May be sharded (e.g., \"MyOutput@100\" leads to a 100-shard output "
    "file). If the filename contains '<level>' as a substring, "
    "HierarchicalFlatCluster() is used to compute the clustering. "
    "In this case, if the flag value is /path/clustering-L<level>, and "
    "the computed hierarchy has two elements, they will be written to "
    "/path/clustering-L0 and /path/clustering-L1.");

ABSL_FLAG(std::string, output_knn, "", "store knn into SSTable");

ABSL_FLAG(bool, use_output_knn, false,
          "if true, use the knn edges stored in output_knn.");

namespace graph_mining {
namespace in_memory {
namespace {
// using Cluster =
//     std::initializer_list<graph_mining::in_memory::InMemoryClusterer::NodeId>;
// using Clustering = graph_mining::in_memory::Clustering;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
// using ::proto2::contrib::parse_proto::ParseTextProtoOrDie;
using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
using T = float;
using Point = Euclidian_Point<T>;
using PointRange = PointRange<T, Point>;
using indexType = unsigned int;
// using research_graph::in_memory::ClustererConfig;
// using research_graph::in_memory::ParHacConfig;

constexpr size_t kRowsPerShard = 50'000'000;

// Joins a clustering (without cluster IDs) and a vector of cluster IDs.
struct ClusteringAndIds {
  // Vector of clusters as integer (node index) vectors.
  InMemoryClusterer::Clustering clustering;
  // Index i gives string cluster ID for cluster with integer index i.
  std::vector<std::string> cluster_ids;
};

// Computes a (potentially hierarchical) clustering and clustering ids using the
// provided clusterer and config.
// If initial_clustering is provided, refines the provided clusters (using
// RefineClusters). If additionally initial_clustering has nonempty cluster_ids,
// the resulting clusters inherit the ids (cluster i gets the id of the ith
// cluster in initial_clustering.cluster_ids).
// If hierarchical is set, uses HierarchicalFlatCluster to compute the
// clustering and returns a hierarchy of 0 or more levels. Otherwise, the
// returned vector has exactly one element.
// Unless the initial clusters and ids are provided, the resulting clusters have
// ids computed by using node_id_map and research_graph::ClusterIdFromNodeIds.
// TODO: Add unit tests.
// /*use_simple_cluster_id=*/true
absl::StatusOr<std::vector<ClusteringAndIds>> ComputeClustersAndIds(
    const InMemoryClusterer& clusterer, const ClustererConfig& config,
    bool hierarchical) {
  std::vector<ClusteringAndIds> result;
  if (!hierarchical) {
    result.emplace_back();
    ASSIGN_OR_RETURN(result.back().clustering, clusterer.Cluster(config));
  } else {
    ASSIGN_OR_RETURN(
        std::vector<InMemoryClusterer::Clustering> clustering_hierarchy,
        clusterer.HierarchicalFlatCluster(config));
    for (auto& clustering_level : clustering_hierarchy) {
      result.emplace_back();
      result.back().clustering = std::move(clustering_level);
    }
  }

  for (auto& clustering_level : result) {
    if (!clustering_level.cluster_ids.empty()) continue;
    for (graph_mining::in_memory::NodeId i = 0;
         i < clustering_level.clustering.size(); i++) {
      clustering_level.cluster_ids.push_back(absl::StrFormat("%08x", i));
    }
  }
  return result;
}

ClustererConfig Config(double epsilon, double weight_threshold) {
  ParHacConfig parhac_clusterer_config;
  parhac_clusterer_config.set_epsilon(epsilon);
  parhac_clusterer_config.set_weight_threshold(weight_threshold);
  ClustererConfig clusterer_config;
  clusterer_config.mutable_parhac_clusterer_config()->Swap(
      &parhac_clusterer_config);
  return clusterer_config;
}

DynamicHacConfig DynamicHacConfig(double epsilon, double weight_threshold) {
  class DynamicHacConfig dynamic_hac_config;
  dynamic_hac_config.set_epsilon(epsilon);
  dynamic_hac_config.set_weight_threshold(weight_threshold);
  return dynamic_hac_config;
}

absl::Status Main() {
  QCHECK(!absl::GetFlag(FLAGS_input_data).empty());
  // graph_mining::in_memory::ParallelSchedulerReference scheduler;
  std::string input_data = absl::GetFlag(FLAGS_input_data);
  std::string output_clustering_path = absl::GetFlag(FLAGS_output_clustering);
  std::string method = absl::GetFlag(FLAGS_method);
  double epsilon = absl::GetFlag(FLAGS_epsilon);
  double weight_threshold = absl::GetFlag(FLAGS_weight_threshold);
  int maxDeg = absl::GetFlag(FLAGS_max_degree);
  int L = absl::GetFlag(FLAGS_l);
  int k = absl::GetFlag(FLAGS_k);
  int num_batch = absl::GetFlag(FLAGS_num_batch);
  double alpha = 1.2;
  // bool hierarchical = absl::GetFlag(FLAGS_hierarchical);
  bool use_output_knn = absl::GetFlag(FLAGS_use_output_knn);
  std::string knn_output_file = absl::GetFlag(FLAGS_output_knn);
  double first_batch_ratio = absl::GetFlag(FLAGS_first_batch_ratio);
  CHECK_LE(first_batch_ratio, 1.0);
  CHECK_GE(first_batch_ratio, 0.0);
  if (method == "parhac") {
    first_batch_ratio = 0;
  }
  if (use_output_knn && knn_output_file.empty()) {
    return absl::InvalidArgumentError(
        "output_knn must be specified if use_output_knn.");
  }

  std::cout << "method: " << method << "\n";
  std::cout << "thread: " << parlay::num_workers() << "\n";
  std::cout << "epsilon: " << epsilon << "\n";
  std::cout << "weight threshold: " << weight_threshold << "\n";
  std::cout << "max degree: " << maxDeg << "\n";
  std::cout << "L: " << L << "\n";
  std::cout << "k: " << k << "\n";
  std::cout << "num batch: " << num_batch << "\n";
  std::cout << "File: " << input_data << std::endl;
  std::cout << "first batch ratio: " << first_batch_ratio << "\n";

  // Read points and initialize Vamana graph for ANNS.
  auto points = BuildPointRange<T>(input_data);
  std::size_t first_batch_size = first_batch_ratio * points.size();
  std::cout << "first batch size: " << first_batch_size << "\n";
  std::size_t store_batch_size = points.size() / 100;
  if (absl::GetFlag(FLAGS_store_batch_size) != -1) {
    store_batch_size = absl::GetFlag(FLAGS_store_batch_size);
  }
  std::cout << "store batch size: " << store_batch_size << "\n";

  std::vector<std::string> node_id_map(points.size());
  parlay::parallel_for(0, points.size(), [&](std::size_t i) {
    node_id_map[i] = std::to_string(i);
  });
  std::cout << "read data\n";
  auto edge_feeder =
      EdgeFeeder<T>(k, points, num_batch, maxDeg, L, alpha, first_batch_size);
  auto edge_reader = EdgeReader<T>(points.size(), num_batch, knn_output_file,
                                   first_batch_size);

  auto graph = GraphUpdater();
  auto config = Config(epsilon, weight_threshold);

  auto dynamic_hac_config = DynamicHacConfig(epsilon, weight_threshold);
  auto dynamic_hac_clusterer = DynamicHacClusterer(dynamic_hac_config);

  auto all_edges = std::vector<std::vector<AdjacencyList>>();
  auto indices = std::vector<int>();

  WallTimer timer;
  timer.Restart();
  int insertion_round = 0;
  while (true) {
    insertion_round++;
    WallTimer subtimer;
    subtimer.Restart();
    std::vector<AdjacencyList> edges;
    std::size_t current_index;
    if (use_output_knn) {
      ASSIGN_OR_RETURN(edges, edge_reader.NextBatch());
      current_index = edge_reader.CurrentIndex();
    } else {
      ASSIGN_OR_RETURN(edges, edge_feeder.NextBatch());
      current_index = edge_feeder.CurrentIndex();
    }
    if (edges.empty()) break;
    std::cout << "====== index: " << current_index << "\n";
    std::cout << "KNN time: " << subtimer.GetSeconds() << " seconds\n";
    if (!use_output_knn) {
      all_edges.push_back(edges);
      indices.push_back(current_index);
    }
    subtimer.Restart();

    std::vector<ClusteringAndIds> clustering_hierarchy;
    Dendrogram dendrogram(0);
    // std::vector<NodeId> mapping;
    if (method == "parhac") {
      RETURN_IF_ERROR(graph.Update(edges));
      // auto clusterer = InMemoryClusterer::CreateOrDie("ParHacClusterer");
      // auto clusterer = graph_mining::in_memory::ParHacClusterer();
      std::unique_ptr<graph_mining::in_memory::InMemoryClusterer> clusterer;
      clusterer.reset(new graph_mining::in_memory::ParHacClusterer);
      RETURN_IF_ERROR(CopyGraph(*graph.graph_, clusterer->MutableGraph()));
      std::cout << "Graph update time: " << subtimer.GetSeconds()
                << " seconds\n";
      subtimer.Restart();

      ASSIGN_OR_RETURN(dendrogram, clusterer->HierarchicalCluster(config));

      std::cout << "Clustering time: " << subtimer.GetSeconds() << " seconds\n";
      std::cout << "Round time: " << timer.GetSeconds() << " seconds\n";
    } else if (method == "dynamic_hac") {
      // Need to make edges contain both directions.
      RETURN_IF_ERROR(FixAdjList(edges));
      std::cout << "FixAdjList time: " << subtimer.GetSeconds() << " seconds\n";
      subtimer.Restart();

      ASSIGN_OR_RETURN(const auto stats, dynamic_hac_clusterer.Insert(edges));
      std::cout << "Num. Rounds: " << dynamic_hac_clusterer.NumRounds() << "\n";
      stats.LogStats();
      std::cout << "Clustering time: " << subtimer.GetSeconds() << " seconds\n";
      std::cout << "Round time: " << timer.GetSeconds() << " seconds\n";
      timer.Restart();
    } else {
      return absl::UnimplementedError("Unknown method: " + method);
    }
    // Note: we are assuming that nodes inserted are consecutive, so
    // we can ignore mapping.
    // for (int i = 0; i < mapping.size(); ++i) {
    //   CHECK(mapping[i] == i)
    //       << "mapping[" << mapping[i] << "] != " << i << "\n";
    // }

    // Store ~100 batches and the last batch. We only store the ones we want to
    // evaluate, because storing is expensive in time (~1s).
    if (current_index % store_batch_size == 0 ||
        current_index == points.size() || insertion_round == 1) {
      if (!output_clustering_path.empty()) {
        std::cout << "Store index: " << current_index << "\n";
        if (method == "dynamic_hac") {
          ASSIGN_OR_RETURN(
              const auto& result,
              dynamic_hac_clusterer.Dendrogram().ConvertToDendrogram());
          dendrogram = std::move(result.first);
        }
        RETURN_IF_ERROR(WriteDendrogram(
            dendrogram, absl::StrCat(output_clustering_path, "-", current_index,
                                     "-dendro.bin")));
        std::cout << "Store time: " << timer.GetSeconds() << " seconds\n";
        timer.Restart();
      }
    }

    timer.Restart();
  }

  if (!use_output_knn)
    RETURN_IF_ERROR(StoreEdges(knn_output_file, all_edges, indices));

  return absl::OkStatus();
}
}  // namespace
}  // namespace in_memory
}  // namespace graph_mining

int main(int argc, char* argv[]) { 
  absl::ParseCommandLine(argc, argv);
  QCHECK_OK(graph_mining::in_memory::Main()); 
}
