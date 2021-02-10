workspace(name = "autoproduction")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# GoogleTest/GoogleMock framework. Used by most unit-tests.
# Last updated 2020-06-30.
http_archive(
    name = "com_google_googletest",
    sha256 = "04a1751f94244307cebe695a69cc945f9387a80b0ef1af21394a490697c5c895",
    strip_prefix = "googletest-aee0f9d9b5b87796ee8a0ab26b7587ec30e8858e",
    urls = ["https://github.com/google/googletest/archive/aee0f9d9b5b87796ee8a0ab26b7587ec30e8858e.zip"],
)

load("//third_party:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "autoproduction_config_cuda")

load("//third_party:tensorrt_configure.bzl", "tensorrt_configure")

tensorrt_configure(name = "autoproduction_config_tensorrt")


new_local_repository(
    name = "opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr/local",
)

_BAZEL_TOOLCHAINS_VERSION = "4.0.0"

http_archive(
    name = "bazel_toolchains",
    strip_prefix = "bazel-toolchains-{}".format(_BAZEL_TOOLCHAINS_VERSION),
    urls = [
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/{}/bazel-toolchains-{}.tar.gz".format(_BAZEL_TOOLCHAINS_VERSION, _BAZEL_TOOLCHAINS_VERSION),
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/releases/download/{}/bazel-toolchains-{}.tar.gz".format(_BAZEL_TOOLCHAINS_VERSION, _BAZEL_TOOLCHAINS_VERSION),
    ],
)

# Build against Kythe master.  Run `bazel sync` to update to the latest commit.
http_archive(
    name = "io_kythe",
    strip_prefix = "kythe-master",
    urls = ["https://github.com/google/kythe/archive/master.zip"],
)

load("@io_kythe//:setup.bzl", "kythe_rule_repositories", "maybe")

kythe_rule_repositories()

# TODO(d61h6k4): remove this, when kythe will resolve it.
# This needs to be loaded before loading the
# go_* rules.  Normally, this is done by go_rules_dependencies in external.bzl,
# but because we want to overload some of those dependencies, we need the go_*
# rules before go_rules_dependencies.  Likewise, we can't precisely control
# when loads occur within a Starlark file so we now need to load this
# manually...
load("@io_bazel_rules_go//go/private:repositories.bzl", "go_name_hack")

maybe(
    go_name_hack,
    name = "io_bazel_rules_go_name_hack",
    is_rules_go = False,
)

load("@io_kythe//:external.bzl", "kythe_dependencies")

kythe_dependencies()
