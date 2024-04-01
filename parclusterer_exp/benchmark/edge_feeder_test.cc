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

#include "approximate_hac_experiments/parclusterer_exp/benchmark/edge_feeder.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/base/optimization.h"
#include "third_party/graph_mining/in_memory/status_macros.h"
#include "third_party/parlayann/utils/euclidian_point.h"
#include "third_party/parlayann/utils/point_range.h"

namespace research_graph::in_memory {
namespace {

using PointRange = PointRange<float, Euclidian_Point<float>>;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(EdgeFeederTest, SimpleTest) {
  std::size_t n = 4;
  int64_t dim = 2;
  int k = 1;
  auto aligned_dim = dim_round_up(dim, sizeof(float));
  float* values = (float*)aligned_alloc(ABSL_CACHELINE_SIZE,
                                        n * aligned_dim * sizeof(float));

  std::vector<std::vector<float>> raw_points{
      {5.0, 5.0}, {2.0, 2.0}, {1.0, 1.0}, {6.0, 6.0}};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < dim; ++j) {
      values[i * aligned_dim + j] = raw_points[i][j];
    }
  }
  auto points = PointRange(n, dim, aligned_dim, values);
  EdgeFeeder<float> feeder(k, points, /* num_batch =*/2);

  ASSERT_OK_AND_ASSIGN(auto edges, feeder.NextBatch());
  EXPECT_THAT(
      edges,
      ElementsAre(FieldsAre(0, 1,
                            UnorderedElementsAre(
                                Pair(1, DoubleNear(1 / (1 + sqrt(18)), 1e-6))),
                            std::nullopt),
                  FieldsAre(1, 1,
                            UnorderedElementsAre(
                                Pair(0, DoubleNear(1 / (1 + sqrt(18)), 1e-6))),
                            std::nullopt)));

  ASSERT_OK_AND_ASSIGN(edges, feeder.NextBatch());
  EXPECT_THAT(
      edges, ElementsAre(FieldsAre(2, 1,
                                   UnorderedElementsAre(Pair(
                                       1, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt),
                         FieldsAre(3, 1,
                                   UnorderedElementsAre(Pair(
                                       0, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt)));

  ASSERT_OK_AND_ASSIGN(edges, feeder.NextBatch());
  EXPECT_THAT(edges, IsEmpty());
  free(values);
}

TEST(EdgeFeederTest, NeighborWithinBatchTest) {
  std::size_t n = 5;
  int64_t dim = 2;
  int k = 1;
  auto aligned_dim = dim_round_up(dim, sizeof(float));
  float* values = (float*)aligned_alloc(ABSL_CACHELINE_SIZE,
                                        n * aligned_dim * sizeof(float));
  std::vector<std::vector<float>> raw_points{
      {1.0, 1.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < dim; ++j) {
      values[i * aligned_dim + j] = raw_points[i][j];
    }
  }
  auto points = PointRange(n, dim, aligned_dim, values);
  EdgeFeeder<float> feeder(k, points, /* num_batch =*/3);

  ASSERT_OK_AND_ASSIGN(auto edges, feeder.NextBatch());
  EXPECT_THAT(
      edges, ElementsAre(FieldsAre(0, 1,
                                   UnorderedElementsAre(Pair(
                                       1, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt),
                         FieldsAre(1, 1,
                                   UnorderedElementsAre(Pair(
                                       0, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt)));

  ASSERT_OK_AND_ASSIGN(edges, feeder.NextBatch());
  EXPECT_THAT(
      edges, ElementsAre(FieldsAre(2, 1,
                                   UnorderedElementsAre(Pair(
                                       3, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt),
                         FieldsAre(3, 1,
                                   UnorderedElementsAre(Pair(
                                       2, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt)));

  ASSERT_OK_AND_ASSIGN(edges, feeder.NextBatch());

  EXPECT_THAT(
      edges, ElementsAre(FieldsAre(4, 1,
                                   UnorderedElementsAre(Pair(
                                       3, DoubleNear(1 / (1 + sqrt(2)), 1e-6))),
                                   std::nullopt)));

  ASSERT_OK_AND_ASSIGN(edges, feeder.NextBatch());
  EXPECT_THAT(edges, IsEmpty());
  free(values);
}

}  // namespace
}  // namespace research_graph::in_memory