
cc_library(
    name = "object_detection",
    hdrs = ["object_detection.h"],
    deps = [
        "//autoproduction/preprocessing:image_chopper",
        "//autoproduction/preprocessing:image_dtype_converter",
        "//autoproduction/postprocessing:chopped_image_detections_joiner",
        "//autoproduction/utils:cuda_helper",
        "//autoproduction/utils:trt_helper",
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_tensorrt//:tensorrt_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "object_detection_test",
    srcs = ["object_detection_test.cc"],
    linkstatic=1,
    data = [
        "//models:object_detection_football",
        "//test_data:test_images",
    ],
    deps = [
        ":object_detection",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@autoproduction_config_cuda//:cuda_runtime",
        "@opencv//:opencv",
    ],
)
