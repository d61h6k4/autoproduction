cc_library(
    name = "cuda_helper",
    srcs = ["cuda_helper.cc"],
    hdrs = ["cuda_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:npp_core",
    ],
)

cc_library(
    name = "detection",
    hdrs = ["detection.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "trt_helper",
    srcs = ["trt_helper.cc"],
    hdrs = ["trt_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":detection",
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_tensorrt//:tensorrt_headers",
        "@autoproduction_config_tensorrt//:tensorrt_onnxparser",
        "@autoproduction_config_tensorrt//:tensorrt_infer",
    ],
)


cc_test(
    name = "cuda_helper_test",
    srcs = ["cuda_helper_test.cc"],
    linkstatic=1,
    deps = [
        ":cuda_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@autoproduction_config_cuda//:cuda_runtime",
    ],
)


cc_test(
    name = "trt_helper_test",
    srcs = ["trt_helper_test.cc"],
    linkstatic=1,
    data = [
        "//models:object_detection_football",
        "//test_data:test_images",
    ],
    deps = [
        ":trt_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@autoproduction_config_cuda//:cuda_runtime",
        "//autoproduction/preprocessing:image_chopper",
        "//autoproduction/preprocessing:image_dtype_converter",
        "//autoproduction/utils:cuda_helper",
        "@opencv//:opencv",
    ],
)
