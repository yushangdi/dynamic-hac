# Bazel workspace file is used to reference external dependencies required to
# build the project. You can use multiple WORKSPACE.bazel files in the same
# project to create new workspaces in subdirectories.

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Google Abseil Libs
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20240116.1",
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "930c2c3b5ecc6c9c12615cf5ad93f1cd6e12d0aba862b572e076259970ac3a53",
    strip_prefix = "protobuf-3.21.12",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.21.12.tar.gz"],
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

http_archive(
    name = "com_google_googletest",
    sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
    strip_prefix = "googletest-release-1.11.0",
    urls = ["https://github.com/google/googletest/archive/release-1.11.0.tar.gz"],
)

git_repository(
    name = "com_github_gbbs",
    remote = "https://github.com/ParAlg/gbbs.git",
    commit = "8d512cb62677fa3c878e8a0a0105bc2e38a8ce43",
)

git_repository(
    name = "parlaylib",
    remote = "https://github.com/ParAlg/parlaylib.git",
    commit = "6b4a4cdbfeb3c481608a42db0230eb6ebb87bf8d",
    strip_prefix = "include/",
)

git_repository(
    name = "com_github_graph_mining",
    remote = "https://github.com/google/graph-mining.git",
    commit = "aaf5a5a1b84d35341776a8b1694640ea07c7c596"
)

git_repository(
    name = "parlayann",
    remote = "git@github.com:cmuparlay/ParlayANN.git",
    commit = "67a0ad46c4728fcda1e7825a3786e4e5f07f43ed", # branch sync
)

# local_repository(
#     name = "parlayann",
#     path = "ParlayANN/",
# )

FARMHASH_COMMIT = "0d859a811870d10f53a594927d0d0b97573ad06d"
FARMHASH_SHA256 = "18392cf0736e1d62ecbb8d695c31496b6507859e8c75541d7ad0ba092dc52115"

http_archive(
    name = "farmhash_archive",
    build_file = "@com_github_graph_mining//utils:farmhash.BUILD",
    sha256 = FARMHASH_SHA256,
    strip_prefix = "farmhash-{commit}".format(commit = FARMHASH_COMMIT),
    urls = ["https://github.com/google/farmhash/archive/{commit}.tar.gz".format(commit = FARMHASH_COMMIT)],
)


# https://github.com/Mizux/bazel-pybind11
# Bazel Extensions
## Bazel Skylib rules.
git_repository(
    name = "bazel_skylib",
    tag = "1.5.0",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

## Bazel rules.
git_repository(
    name = "platforms",
    tag = "0.0.8",
    remote = "https://github.com/bazelbuild/platforms.git",
)

git_repository(
    name = "rules_cc",
    tag = "0.0.9",
    remote = "https://github.com/bazelbuild/rules_cc.git",
)

git_repository(
    name = "rules_python",
    tag = "0.27.1",
    remote = "https://github.com/bazelbuild/rules_python.git",
)

## Python
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()


## `pybind11_bazel`
git_repository(
    name = "pybind11_bazel",
    commit = "23926b00e2b2eb2fc46b17e587cf0c0cfd2f2c4b", # 2023/11/29
    patches = ["//patches:pybind11_bazel.patch"],
    patch_args = ["-p1"],
    remote = "https://github.com/pybind/pybind11_bazel.git",
)

new_git_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    tag = "v2.11.1",
    remote = "https://github.com/pybind/pybind11.git",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python", python_version = "3")
bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)