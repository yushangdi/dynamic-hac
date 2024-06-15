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
 * Copyright 2024 Approximate Hac Experiments Authors
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

#ifndef EXPERIMENTAL_USERS_SHANGDI_PARCLUSTERER_EXP_BENCHMARK_IO_H_
#define EXPERIMENTAL_USERS_SHANGDI_PARCLUSTERER_EXP_BENCHMARK_IO_H_

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"
#include "parlay/slice.h"
#include "algorithms/utils/euclidian_point.h"
#include "algorithms/utils/point_range.h"

namespace graph_mining::in_memory {

std::vector<char> ReadFileToBuffer(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);  // Open in binary mode

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
  }

  // Get file size (assuming pre-processing or another way to get size)
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);  // Seek back to beginning

  // Allocate buffer based on file size
  std::vector<char> buffer(file_size);

  // Read entire file into buffer
  file.read(buffer.data(), file_size);

  if (!file.good()) {
    std::cerr << "Error reading file: " << filename << std::endl;
  }

  file.close();  // Close the file

  return buffer;
}

bool writeToFile(const std::string& filename, const std::vector<char>& buffer) {
  std::ofstream outputFile(filename, std::ios::binary);  // Open in binary mode

  if (!outputFile.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return false;
  }

  outputFile.write(buffer.data(), buffer.size());  // Write the entire buffer

  if (!outputFile.good()) {
    std::cerr << "Error writing to file: " << filename << std::endl;
    return false;
  }

  outputFile.close();  // Close the file

  return true;  // Indicate successful writing
}

// Read in fvecs format file and return PointRange.
// Differ from the original point range constructor in two places
// 1. the file does not start with the size, only the dimension
// 2. each point has size (dim + 1), the first dim is the dimension
template <typename T>
PointRange<T, Euclidian_Point<T>> BuildPointRange(
    const std::string input_data) {
  std::ifstream inputFile(input_data, std::ios::binary);  // Open in binary mode

  if (!inputFile.is_open()) {
    std::cerr << "Error opening input file: " << input_data << std::endl;
  }
  inputFile.seekg(0, std::ios::end);
  size_t file_size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);  // Seek back to beginning

  unsigned int dims;
  inputFile.read(reinterpret_cast<char*>(&dims), sizeof(unsigned int));
  unsigned int num_points = file_size / 4 / (dims + 1);
  // unsigned int num_points = stat_proto.length() / 4 / (dims + 1);
  std::cout << "Detected " << num_points << " points with dimension " << dims
            << "\n";
  unsigned int aligned_dims = dim_round_up(dims, sizeof(T));
  if (aligned_dims != dims)
    std::cout << "Aligning dimension to " << aligned_dims << "\n";

  T* values = (T*)aligned_alloc(ABSL_CACHELINE_SIZE,
                                num_points * aligned_dims * sizeof(T));
  CHECK(values != nullptr);
  size_t BLOCK_SIZE = 1000000;
  size_t index = 0;
  while (index < num_points) {
    size_t floor = index;
    size_t ceiling =
        index + BLOCK_SIZE <= num_points ? index + BLOCK_SIZE : num_points;
    T* data_start = new T[(ceiling - floor) * (dims + 1)];
    // CHECK_OK(file::ReadToBuffer(fp, (char*)(data_start),
    //                             sizeof(T) * (ceiling - floor) * (dims + 1),
    //                             &read_bytes, file::PartialRead()));
    inputFile.read(reinterpret_cast<char*>(data_start),
                   sizeof(T) * (ceiling - floor) * (dims + 1));
    std::cout << "\n";
    std::cout << "data start\n";
    T* data_end = data_start + (ceiling - floor) * (dims + 1);
    parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
    int data_bytes = dims * sizeof(T);
    parlay::parallel_for(floor, ceiling, [&](size_t i) {
      std::memmove(values + i * aligned_dims,
                   data.begin() + (i - floor) * (dims + 1), data_bytes);
    });
    delete[] data_start;
    index = ceiling;
  }

  std::cout << "sainty check, first 5 values\n";
  for (size_t i = 0; i < 5; ++i) {
    std::cout << values[i] << " ";
  }
  std::cout << "\n";
  inputFile.close();
  return PointRange<T, Euclidian_Point<T>>(num_points, dims, aligned_dims,
                                           values);
}

absl::StatusOr<std::vector<
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>
ReadEdges(std::string input_file) {
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  std::vector<char> buffer = ReadFileToBuffer(input_file);
  char* buffer_ptr = buffer.data();

  // Read the number of tuples from the file
  std::size_t num_nodes;
  std::size_t num_edges;
  std::memcpy(&num_nodes, buffer_ptr, sizeof(std::size_t));
  buffer_ptr += sizeof(std::size_t);
  std::memcpy(&num_edges, buffer_ptr, sizeof(std::size_t));
  buffer_ptr += sizeof(std::size_t);

  std::cout << "num nodes: " << num_nodes << "\n";
  std::cout << "num edges: " << num_edges << "\n";

  std::vector<AdjacencyList> data(num_nodes);

  // Read each tuple from the file
  int cur_nodes = -1;
  NodeId previous_node_id = -1;
  for (int i = 0; i < num_edges; ++i) {
    NodeId u;
    NodeId v;
    double w;
    std::memcpy(&u, buffer_ptr, sizeof(NodeId));
    buffer_ptr += sizeof(NodeId);
    std::memcpy(&v, buffer_ptr, sizeof(NodeId));
    buffer_ptr += sizeof(NodeId);
    std::memcpy(&w, buffer_ptr, sizeof(double));
    buffer_ptr += sizeof(double);
    if (u != previous_node_id) {
      cur_nodes++;
      previous_node_id = u;
      data[cur_nodes].id = u;
      data[cur_nodes].weight = 1;
    }
    data[cur_nodes].outgoing_edges.push_back({v, w});
  }

  return data;
}

// first line: num nodes
// second line: num edges
// u, v, w
// u, v, w
absl::Status StoreEdges(
    std::string knn_output_file,
    const std::vector<std::vector<
        graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>&
        edges,
    const std::vector<int>& indices) {
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  if (knn_output_file.empty()) {
    return absl::OkStatus();
  }
  const auto num_batch = edges.size();
  for (int batch_index = 0; batch_index < num_batch; ++batch_index) {
    const std::size_t num_nodes = edges[batch_index].size();
    std::size_t num_edges = 0;
    for (const auto& edge : edges[batch_index]) {
      num_edges += edge.outgoing_edges.size();
    }
    int buffer_size =
        2 * sizeof(std::size_t)  // For number of tuples
        + num_edges * (sizeof(NodeId) + sizeof(NodeId) + sizeof(double));
    std::vector<char> buffer(buffer_size);

    // Copy data into the buffer
    char* buffer_ptr = buffer.data();
    std::memcpy(buffer_ptr, &num_nodes, sizeof(std::size_t));
    buffer_ptr += sizeof(std::size_t);
    std::memcpy(buffer_ptr, &num_edges, sizeof(std::size_t));
    buffer_ptr += sizeof(std::size_t);
    for (const auto& edge : edges[batch_index]) {
      for (const auto& [v, w] : edge.outgoing_edges) {
        std::memcpy(buffer_ptr, &edge.id, sizeof(NodeId));
        buffer_ptr += sizeof(NodeId);
        std::memcpy(buffer_ptr, &v, sizeof(NodeId));
        buffer_ptr += sizeof(NodeId);
        std::memcpy(buffer_ptr, &w, sizeof(double));
        buffer_ptr += sizeof(double);
      }
    }

    // Write the buffer to file in one go
    const std::string filename =
        absl::StrCat(knn_output_file, "_", indices[batch_index], ".bin");
    if (!writeToFile(filename, buffer)) {
      return absl::InternalError(
          absl::StrCat("Failed to write to file, ", filename));
    }
  }

  return absl::OkStatus();
}

//  num nodes
//  parent, merge_similarity
//  parent, merge_similarity
// ....
absl::Status WriteDendrogram(
    const graph_mining::in_memory::Dendrogram& dendrogram,
    const std::string& output_file) {
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;

  if (output_file.empty()) {
    return absl::OkStatus();
  }

  const auto& nodes = dendrogram.Nodes();
  const std::size_t num_clustered_nodes = dendrogram.NumClusteredNodes();

  const std::size_t n = nodes.size();
  int buffer_size =
      2 * sizeof(std::size_t) + n * (sizeof(NodeId) + sizeof(double));
  std::vector<char> buffer(buffer_size);

  const auto& sizes = dendrogram.GetClusterSizes();

  // Copy data into the buffer
  char* buffer_ptr = buffer.data();
  std::memcpy(buffer_ptr, &n, sizeof(std::size_t));
  buffer_ptr += sizeof(std::size_t);
  std::memcpy(buffer_ptr, &num_clustered_nodes, sizeof(std::size_t));
  buffer_ptr += sizeof(std::size_t);
  for (const auto& node : nodes) {
    const auto& [p, w] = node;
    std::memcpy(buffer_ptr, &p, sizeof(NodeId));
    buffer_ptr += sizeof(NodeId);
    std::memcpy(buffer_ptr, &w, sizeof(double));
    buffer_ptr += sizeof(double);
  }

  // Write the buffer to file in one go
  if (writeToFile(output_file, buffer)) {
    return absl::OkStatus();
  } else {
    return absl::InternalError(
        absl::StrCat("Failed to write to file, ", output_file));
  }
}

absl::StatusOr<graph_mining::in_memory::Dendrogram> ReadDendrogram(
    const std::string& input_file) {
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  using graph_mining::in_memory::Dendrogram;
  using graph_mining::in_memory::DendrogramNode;

  std::vector<char> buffer = ReadFileToBuffer(input_file);
  char* buffer_ptr = buffer.data();

  std::size_t num_nodes;
  std::size_t num_clustered_nodes;
  std::memcpy(&num_nodes, buffer_ptr, sizeof(std::size_t));
  buffer_ptr += sizeof(std::size_t);
  std::memcpy(&num_clustered_nodes, buffer_ptr, sizeof(std::size_t));
  buffer_ptr += sizeof(std::size_t);

  auto nodes = std::vector<DendrogramNode>();
  for (int i = 0; i < num_nodes; ++i) {
    NodeId p;
    double w;
    std::memcpy(&p, buffer_ptr, sizeof(NodeId));
    buffer_ptr += sizeof(NodeId);
    std::memcpy(&w, buffer_ptr, sizeof(double));
    buffer_ptr += sizeof(double);
    nodes.push_back({p, w});
  }

  Dendrogram dendrogram(0);
  RETURN_IF_ERROR(dendrogram.Init(std::move(nodes), num_clustered_nodes));

  return dendrogram;
}

absl::StatusOr<std::vector<graph_mining::in_memory::InMemoryClusterer::NodeId>>
ReadClusterIds(std::string& input_file) {
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  using graph_mining::in_memory::Dendrogram;
  using graph_mining::in_memory::DendrogramNode;

  std::vector<char> buffer = ReadFileToBuffer(input_file);
  char* buffer_ptr = buffer.data();

  size_t num_elements = buffer.size() / sizeof(int);

  std::vector<int> labels_vector(num_elements);
  memcpy(labels_vector.data(), buffer_ptr, buffer.size());

  return labels_vector;
}

absl::Status FixAdjList(
    std::vector<
        graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>&
        edges) {
  using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
  // Map node id to its index in `edges`.
  absl::flat_hash_map<NodeId, NodeId> index_map;
  for (NodeId i = 0; i < edges.size(); ++i) {
    index_map[edges[i].id] = i;
  }

  // Return true if we need add `neighbor` to `u`'s neighbor list. We need to
  // add if `u` is in `index_map` (it's a new node) and `neighbor` is not
  // already in `u`'s neighbor list.
  auto add_neighbor = [&](const NodeId u, const NodeId neighbor) {
    auto itr = index_map.find(u);
    // `u` is not a new node, does not need to add.
    if (itr == index_map.end()) return false;
    const auto u_index = itr->second;
    bool has_node = false;
    for (auto& [v, w] : edges[u_index].outgoing_edges) {
      if (v == neighbor) {
        has_node = true;
        break;
      }
    }
    return !has_node;
  };

  // Takes O(NK^2).
  for (auto& node : edges) {
    for (auto& [v, w] : node.outgoing_edges) {
      if (add_neighbor(v, node.id)) {
        const auto v_index = index_map.find(v)->second;
        edges[v_index].outgoing_edges.push_back({node.id, w});
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace graph_mining::in_memory
#endif  // EXPERIMENTAL_USERS_SHANGDI_PARCLUSTERER_EXP_BENCHMARK_IO_H_
