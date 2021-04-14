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
