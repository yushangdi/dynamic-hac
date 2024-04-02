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

/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_BENCHMARK_EDGE_FEEDER_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_BENCHMARK_EDGE_FEEDER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "parclusterer_exp/benchmark/io.h"
#include "graph_mining/in_memory/clustering/config.proto.h"
#include "graph_mining/in_memory/clustering/graph.h"
#include "graph_mining/in_memory/clustering/in_memory_clusterer.h"
#include "graph_mining/in_memory/status_macros.h"
#include "graph_mining/utils/status/thread_safe_status.h"
#include "parlay/include/parlay/parallel.h"
#include "parlayann/utils/beamSearch.h"
#include "parlayann/utils/euclidian_point.h"
#include "parlayann/utils/graph.h"
#include "parlayann/utils/point_range.h"
#include "parlayann/utils/stats.h"
#include "parlayann/utils/types.h"
#include "parlayann/vamana/index.h"

namespace graph_mining::in_memory {

// return another PointRange that contains the first n points in `pr`.s
template <typename T>
PointRange<T, Euclidian_Point<T>> PointRangeCopy(
    const PointRange<T, Euclidian_Point<T>>& pr, const size_t start,
    const size_t end) {
  assert(start < end);
  assert(end <= pr.size());
  size_t n = end - start;
  unsigned int dims = pr.dimension();
  unsigned int aligned_dims = dim_round_up(dims, sizeof(T));
  T* values =
      (T*)aligned_alloc(ABSL_CACHELINE_SIZE, n * aligned_dims * sizeof(T));
  parlay::parallel_for(start, end, [&](size_t i) {
    std::memmove(values + i * aligned_dims, pr.get_values() + i * aligned_dims,
                 dims * sizeof(T));
  });
  return PointRange<T, Euclidian_Point<T>>(n, dims, aligned_dims, values);
}

// Return a clique with nodes 0 ...  n-1, and edge weights are distances between
// points.
template <class PointRange>
std::vector<graph_mining::in_memory::InMemoryClusterer::AdjacencyList> Clique(
    std::size_t n, PointRange& points) {
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::AdjacencyList;
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  std::vector<AdjacencyList> edges(n);
  parlay::parallel_for(0, n, [&](std::size_t i) {
    std::vector<std::pair<NodeId, double>> outgoing_edges;
    outgoing_edges.reserve(n - 1);
    for (std::size_t j = 0; j < n; ++j) {
      if (i != j) {
        outgoing_edges.push_back(
            std::make_pair(j, points[j].distance(points[i])));
      }
    }
    // for (std::size_t j = 0; j < outgoing_edges.size(); ++j) {
    //   std::cout << i << " " << outgoing_edges[j].first << " "
    //             << outgoing_edges[j].second << "\n";
    // }
    edges[i] = AdjacencyList(i, 1, std::move(outgoing_edges), std::nullopt);
  });
  return edges;
}

// Keep the k nearest neighbors of a metric data set. The points are inserted in
// batches, and the neighbors are only found from points inserted in previous
// and current batches.
template <typename T>
class EdgeFeeder {
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  using Point = Euclidian_Point<T>;
  using PointRange = PointRange<T, Point>;
  using indexType = unsigned int;
  using findex = knn_index<Point, PointRange, indexType>;

 public:
  EdgeFeeder(int k, PointRange& points, std::size_t num_batch = 100,
             int max_degree = 50, int L = 50, double alpha = 1.2,
             std::size_t first_batch_size = 0)
      : k_(k),  // increment 1 because the result contains the point itself.
        batch_size_(std::max(std::size_t(1),
                             points.size() / num_batch +
                                 (points.size() % num_batch == 0 ? 0 : 1))),
        max_degree_(max_degree),
        beam_size_(L),
        alpha_(alpha),
        points_(points),
        first_batch_size_(first_batch_size) {
    if (first_batch_size != 0) {
      CHECK_GE(first_batch_size, k_ + 1);
    } else {
      CHECK_GE(batch_size_, k_ + 1);
    }
    CHECK_LE(first_batch_size_, points.size());
    knn_ = std::vector<std::vector<std::pair<indexType, T>>>(points.size());
    G_ = Graph<unsigned int>(max_degree_, points_.size());
    auto BP = BuildParams(max_degree_, beam_size_, alpha_, 0, 0, 0, 0);
    I_ = std::make_unique<findex>(BP);
    I_->start_point = 0;
  }

  std::size_t CurrentIndex() const { return current_index_; }

  // Return empty vector if reaches the end of all points.
  absl::StatusOr<std::vector<AdjacencyList>> NextBatch() {
    if (current_index_ == points_.size()) return std::vector<AdjacencyList>();
    std::size_t start = current_index_;
    std::size_t end = std::min(start + batch_size_, points_.size());
    if (start == 0 && first_batch_size_ != 0) {
      end = first_batch_size_;
    }
    current_index_ = end;
    std::cout << "start ind: " << start << " end ind: " << end << "\n";

    // Update index
    stats<unsigned int> BuildStats(G_.size());
    parlay::sequence<indexType> inserts = parlay::tabulate(
        end - start,
        [&](size_t i) { return static_cast<indexType>(i + start); });
    I_->batch_insert(inserts, G_, points_, BuildStats, true, 2, .02);
    std::cout << "index updated\n";

    // Build index
    // auto points_insert = PointRangeCopy(points_, 0, end);
    // auto BP = BuildParams(max_degree_, beam_size_, alpha_, 0, 0, 0, 0);
    // I_ = std::make_unique<findex>(BP);
    // I_->build_index(G_, points_insert, BuildStats);

    // Find k-nearest neighbors of points_[start, end).
    auto QP = QueryParams(k_ + 1, beam_size_, 1.35, G_.size(), G_.max_degree());
    parlay::parallel_for(start, end, [&](size_t i) {
      auto start_point = i;
      auto neighbors = (beam_search<Point, PointRange, indexType>(
                            points_[i], G_, points_, start_point, QP))
                           .first.first;
      CHECK_GE(neighbors.size(), QP.k) << "id: " << i;
      knn_[i].resize(QP.k);
      for (indexType j = 0; j < QP.k; j++) {
        const auto neighbor = neighbors[j].first;
        const auto distance = points_[i].distance(points_[neighbor]);
        knn_[i][j] = {neighbor, sqrt(distance)};
      }
    });
    return Edges(start, end);
  }

  // Convert the knn of points[start, end) to adjacency lists, remove itself as
  // the nearest neighbor.
  std::vector<AdjacencyList> Edges(std::size_t start, std::size_t end) {
    std::size_t n = end - start;
    std::vector<AdjacencyList> edges(n);
    parlay::parallel_for(start, end, [&](std::size_t i) {
      std::vector<std::pair<NodeId, double>> outgoing_edges;
      outgoing_edges.reserve(knn_[i].size() - 1);
      for (std::size_t j = 0; j < knn_[i].size(); ++j) {
        if (knn_[i][j].first == i) continue;
        outgoing_edges.push_back(
            {knn_[i][j].first, 1.0 / (1 + knn_[i][j].second)});
      }
      edges[i - start] =
          AdjacencyList(i, 1, std::move(outgoing_edges), std::nullopt);
    });
    return edges;
  }

 private:
  const int k_;
  const std::size_t batch_size_;
  const int max_degree_;
  std::size_t current_index_ = 0;
  const int beam_size_;
  const double alpha_;
  PointRange& points_;
  std::vector<std::vector<std::pair<indexType, T>>> knn_;
  Graph<unsigned int> G_;
  std::unique_ptr<findex> I_;
  std::size_t first_batch_size_;
};

template <typename T>
class EdgeReader {
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;

 public:
  EdgeReader(std::size_t n, std::size_t num_batch, std::string filename,
             std::size_t first_batch_size = 0)
      : filename_(filename),
        n_(n),
        batch_size_(n / num_batch + (n % num_batch == 0 ? 0 : 1)),
        first_batch_size_(first_batch_size) {
    batch_size_ = std::max(batch_size_, std::size_t(1));
    CHECK_LE(first_batch_size_, n);
  }

  std::size_t CurrentIndex() const { return current_index_; }

  // Return empty vector if reaches the end of all points.
  absl::StatusOr<std::vector<AdjacencyList>> NextBatch() {
    if (current_index_ == n_) return std::vector<AdjacencyList>();
    std::size_t start = current_index_;
    std::size_t end = std::min(start + batch_size_, n_);
    if (start == 0 && first_batch_size_ != 0) {
      end = first_batch_size_;
    }
    current_index_ = end;
    std::cout << "start ind: " << start << " end ind: " << end << "\n";
    ASSIGN_OR_RETURN(auto edges,
                     ReadEdges(absl::StrCat(filename_, "_", end, ".bin")));
    return std::move(edges);
  }

 private:
  std::size_t current_index_ = 0;
  std::string filename_;
  std::size_t n_;
  std::size_t batch_size_;
  std::size_t first_batch_size_;
};

class GraphUpdater {
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;

 public:
  explicit GraphUpdater() {
    graph_ = std::make_unique<graph_mining::in_memory::SimpleUndirectedGraph>();
  }

  absl::Status Update(const std::vector<AdjacencyList>& edges) {
    RETURN_IF_ERROR(graph_->PrepareImport(graph_->NumNodes() + edges.size()));
    graph_mining::ThreadSafeStatus loop_status;
    parlay::parallel_for(0, edges.size(), [&](std::size_t i) {
      auto status = graph_->Import(std::move(edges[i]));
      loop_status.Update(status);
    });
    RETURN_IF_ERROR(loop_status.status());
    RETURN_IF_ERROR(graph_->FinishImport());
    return absl::OkStatus();
  }
  std::unique_ptr<graph_mining::in_memory::SimpleUndirectedGraph> graph_;
};

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_BENCHMARK_EDGE_FEEDER_H_
