cc_library(
    name = "chopper",
    srcs = [],
    hdrs = ["chopper.h"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:nppi_geometry",
    ],
)

cc_test(
    name = "chopper_test",
    srcs = ["chopper_test.cc"],
    deps = [
        ":chopper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
