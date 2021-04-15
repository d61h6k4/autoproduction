
#include "autoproduction/inference/object_detection.h"

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include "gtest/gtest.h"

namespace Autoproduction {
namespace Inference {
namespace {
struct Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    if (severity != nvinfer1::ILogger::Severity::kVERBOSE) {
      std::clog << msg << "\n";
    }
  }
};
}  // namespace
class ObjectDetectionModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_id = 0;
    cudaError_t st = cudaSetDevice(device_id);
    EXPECT_TRUE(st == cudaSuccess);

    st = cudaStreamCreate(&cuda_stream_);
    EXPECT_TRUE(st == cudaSuccess);

    logger_ = std::make_shared<Logger>();

    img_ = cv::imread("test_data/frames_0001.png", cv::IMREAD_COLOR);
    ctx_ = Autoproduction::Util::fromCudaStreamAndDeviceId(cuda_stream_,
                                                           device_id);
  }
  cudaStream_t cuda_stream_;
  std::shared_ptr<Logger> logger_;
  std::string path_to_the_model_ =
      "models/object_detection_football_int8.engine";

  cv::Mat img_;
  NppStreamContext ctx_;
};

TEST_F(ObjectDetectionModelTest, SanityCheck) {
  constexpr int rows_num = 1;
  constexpr int columns_num = 5;
  auto odmodel =
      Autoproduction::Inference::ObjectDetectionModel<rows_num, columns_num>(
          path_to_the_model_, true, img_.rows, img_.cols, 640, 640,
          cuda_stream_, 0, logger_);

  Npp8u* img_ptr;
  size_t input_size = img_.rows * img_.cols * img_.channels() * sizeof(char);
  cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&img_ptr), input_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  cv::cuda::GpuMat cuda_img(img_.rows, img_.cols, CV_8UC3, img_ptr);
  cuda_img.upload(img_);

  auto dets = odmodel(img_ptr);
  EXPECT_EQ(rows_num * columns_num * 100, dets.size());
}

}  // namespace Inference
}  // namespace Autoproduction
