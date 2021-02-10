cc_library(
    name = "chopped_image_detections_joiner",
    srcs = ["chopped_image_detections_joiner.cc"],
    hdrs = ["chopped_image_detections_joiner.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//autoproduction/preprocessing:image_grid",
        "//autoproduction/utils:detection",
    ],
)

cc_test(
    name = "chopped_image_detections_joiner_test",
    srcs = ["chopped_image_detections_joiner_test.cc"],
    deps = [
        ":chopped_image_detections_joiner",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
