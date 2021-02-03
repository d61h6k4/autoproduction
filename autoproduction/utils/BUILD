cc_library(
    name = "cuda_helper",
    srcs = ["cuda_helper.cc"],
    hdrs = ["cuda_helper.h"],
    linkstatic = 1,
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:cuda_runtime",
        "@autoproduction_config_cuda//:npp_core",
    ],
)

cc_test(
    name = "cuda_helper_test",
    srcs = ["cuda_helper_test.cc"],
    deps = [
        ":cuda_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
