cc_library(
    name = "image_grid",
    hdrs = ["image_grid.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
    ],
)

cc_library(
    name = "image_chopper",
    srcs = ["image_chopper.cc"],
    hdrs = ["image_chopper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_grid",
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:nppi_geometry",
    ],
)

cc_library(
    name = "image_dtype_converter",
    srcs = ["image_dtype_converter.cc"],
    hdrs = ["image_dtype_converter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@autoproduction_config_cuda//:cuda_headers",
        "@autoproduction_config_cuda//:nppi_data_exchange_and_initialization",
    ],
)

cc_test(
    name = "image_grid_test",
    srcs = ["image_grid_test.cc"],
    deps = [
        ":image_grid",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "image_chopper_test",
    srcs = ["image_chopper_test.cc"],
    data = [
        "//test_data:test_images",
    ],
    linkstatic = 1,
    deps = [
        ":image_chopper",
        "//autoproduction/utils:cuda_helper",
        "@autoproduction_config_cuda//:cuda_runtime",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@opencv",
    ],
)

cc_test(
    name = "image_preprocessing_test",
    srcs = ["image_preprocessing_test.cc"],
    data = [
        "//test_data:test_images",
    ],
    linkstatic = 1,
    deps = [
        ":image_chopper",
        ":image_dtype_converter",
        "//autoproduction/utils:cuda_helper",
        "@autoproduction_config_cuda//:cuda_runtime",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@opencv",
    ],
)
