cc_library(
    name = "chopper",
    srcs = ["chopper.cc"],
    hdrs = ["chopper.h"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:nppi_geometry",
    ],
)

cc_test(
    name = "chopper_test",
    srcs = ["chopper_test.cc"],
    data = [
        "//test_data:test_images",
    ],
    deps = [
        ":chopper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@opencv//:opencv",
        "@autoproduction_config_cuda//:cuda_runtime"
    ],
)
