cc_library(
    name = "image_chopper",
    srcs = ["image_chopper.cc"],
    hdrs = ["image_chopper.h"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:nppi_geometry",
    ],
)

cc_library(
    name = "image_dtype_converter",
    srcs = ["image_dtype_converter.cc"],
    hdrs = ["image_dtype_converter.h"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:nppi_data_exchange_and_initialization",
    ],
)

cc_test(
    name = "image_chopper_test",
    srcs = ["image_chopper_test.cc"],
    data = [
        "//test_data:test_images",
    ],
    deps = [
        ":image_chopper",
        "//autoproduction/utils:cuda_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@opencv//:opencv",
        "@autoproduction_config_cuda//:cuda_runtime"
    ],
)

cc_test(
    name = "image_preprocessing_test",
    srcs = ["image_preprocessing_test.cc"],
    data = [
        "//test_data:test_images",
    ],
    deps = [
        ":image_chopper",
        ":image_dtype_converter",
        "//autoproduction/utils:cuda_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@opencv//:opencv",
        "@autoproduction_config_cuda//:cuda_runtime"
    ],
)
