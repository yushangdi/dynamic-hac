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

package(default_visibility = [
    "//visibility:public",
])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_library(
    name = "edge_feeder",
    hdrs = ["edge_feeder.h"],
    deps = [
        ":io",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_github_graph_mining//in_memory:status_macros",
        "@com_github_graph_mining//in_memory/clustering:config_cc_proto",
        "@com_github_graph_mining//in_memory/clustering:graph",
        "@com_github_graph_mining//in_memory/clustering:in_memory_clusterer",
        "@com_github_graph_mining//in_memory/clustering/hac:parhac",
        "@com_github_graph_mining//utils/status:thread_safe_status",
        "@parlaylib//parlay:parallel",
        "@parlayann//algorithms/utils:beamSearch",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:graph",
        "@parlayann//algorithms/utils:point_range",
        "@parlayann//algorithms/utils:stats",
        "@parlayann//algorithms/utils:types",
        "@parlayann//algorithms/vamana:index",
    ],
)

cc_test(
    name = "edge_feeder_test",
    srcs = ["edge_feeder_test.cc"],
    deps = [
        ":edge_feeder",
        "@com_google_googletest//:gtest_main",     
        "@com_google_googletest//:gtest",  
        "@com_google_absl//absl/base:core_headers",
        "@com_github_graph_mining//in_memory:status_macros",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:point_range",
    ],
)

cc_library(
    name = "parhac_main_lib",
    srcs = ["parhac_main.cc"],
    deps = [
        ":edge_feeder",
        ":io",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_github_graph_mining//in_memory:status_macros",
        "@com_github_graph_mining//in_memory/clustering:config_cc_proto",
        "@com_github_graph_mining//in_memory/clustering:dendrogram",
        "@com_github_graph_mining//in_memory/clustering:graph",
        "@com_github_graph_mining//in_memory/clustering:in_memory_clusterer",
        "@com_github_graph_mining//in_memory/clustering:types",
        "@com_github_graph_mining//in_memory/clustering/dynamic/hac",
        "@com_github_graph_mining//in_memory/clustering/dynamic/hac:dynamic_hac_cc_proto",
        "@com_github_graph_mining//in_memory/clustering/hac:parhac",
        "@com_github_graph_mining//in_memory/parallel:scheduler",
        "@com_github_graph_mining//utils:timer",
        "@parlaylib//parlay:parallel",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:point_range",
    ],
)

cc_binary(
    name = "parhac_main",
    deps = [
        "parhac_main_lib",
    ],
)

cc_library(
    name = "deletion_main_lib",
    srcs = ["deletion_main.cc"],
    deps = [
        ":edge_feeder",
        ":io",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_github_graph_mining//in_memory:status_macros",
        "@com_github_graph_mining//in_memory/clustering:config_cc_proto",
        "@com_github_graph_mining//in_memory/clustering:dendrogram",
        "@com_github_graph_mining//in_memory/clustering:in_memory_clusterer",
        "@com_github_graph_mining//in_memory/clustering/dynamic/hac",
        "@com_github_graph_mining//in_memory/clustering/dynamic/hac:dynamic_hac_cc_proto",
        "@com_github_graph_mining//in_memory/clustering/hac:parhac",
        "@com_github_graph_mining//in_memory/parallel:scheduler",
        "@com_github_graph_mining//utils:timer",
        "@parlaylib//parlay/include/parlay:parallel",
        "@parlayann//utils:euclidian_point",
        "@parlayann//utils:point_range",
    ],
)

cc_binary(
    name = "deletion_main",
    deps = [
        "deletion_main_lib",
    ],
)

# pytype_strict_library(
#     name = "evaluate_utils",
#     srcs = ["evaluate_utils.py"],
#     deps = [
#         ":cut_dendrogram",
#         "//third_party/py/matplotlib",
#         "//third_party/py/numpy",
#         "//third_party/py/pandas:pandas_internal",
#         "//third_party/py/sklearn",
#         "//third_party/py/tqdm",
#     ],
# )

# pytype_strict_binary(
#     name = "evaluate_clustering",
#     srcs = ["evaluate_clustering.py"],
#     test_lib = True,
#     deps = [
#         ":evaluate_utils",
#         "//third_party/py/absl:app",
#         "//third_party/py/absl/flags",
#         "//third_party/py/pandas:pandas_internal",
#         "//third_party/py/sklearn",
#     ],
# )

# pytype_strict_binary(
#     name = "process_result",
#     srcs = ["process_result.py"],
#     test_lib = True,
#     deps = [
#         ":evaluate_utils",
#         "//third_party/py/absl:app",
#         "//third_party/py/absl/flags",
#     ],
# )

# pytype_strict_binary(
#     name = "run_experiment",
#     srcs = ["run_experiment.py"],
#     deps = [
#         ":evaluate_utils",
#         "//third_party/py/absl:app",
#         "//third_party/py/absl/flags",
#     ],
# )

# pytype_strict_binary(
#     name = "grove_main",
#     srcs = ["grove_main.py"],
#     python_version = "PY3",
#     srcs_version = "PY3ONLY",
#     deps = [
#         ":evaluate_utils",
#         "//third_party/py/absl:app",
#         "//third_party/py/absl/flags",
#         "//third_party/py/graphgrove",
#         "//third_party/py/numpy",
#         "//third_party/py/pandas:pandas_internal",
#         "//third_party/py/scipy",
#         "//third_party/py/sklearn",
#         "//third_party/py/tqdm",
#     ],
# )

cc_library(
    name = "io",
    hdrs = ["io.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_github_graph_mining//in_memory:status_macros",
        "@com_github_graph_mining//in_memory/clustering:dendrogram",
        "@com_github_graph_mining//in_memory/clustering:in_memory_clusterer",
        "@com_github_graph_mining//in_memory/clustering/hac:parhac",
        "@parlaylib//parlay:slice",
        "@parlayann//algorithms/utils:euclidian_point",
        "@parlayann//algorithms/utils:point_range",
    ],
)

pybind_extension(
    name = "cut_dendrogram",
    srcs = ["cut_dendrogram.cc"],
    deps = [
        ":io",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_github_graph_mining//in_memory/clustering:dendrogram",
        "@com_github_graph_mining//in_memory/clustering:in_memory_clusterer",
        "@com_github_graph_mining//in_memory/clustering:types",
    ],
)


# pytype_strict_library(
#     name = "grinch",
#     srcs = ["grinch.py"],
#     deps = [
#         "//third_party/py/numpy",
#         "//third_party/py/scipy",
#     ],
# )

# pytype_strict_binary(
#     name = "grinch_main",
#     srcs = ["grinch_main.py"],
#     # python_version = "PY3",
#     deps = [
#         ":evaluate_utils",
#         ":grinch",
#         "//third_party/py/absl:app",
#         "//third_party/py/absl/flags",
#         "//third_party/py/absl/logging",
#         "//third_party/py/numpy",
#         "//third_party/py/pandas:pandas_internal",
#         "//third_party/py/tqdm",
#     ],
# )
