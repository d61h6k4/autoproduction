
#include "autoproduction/preprocessing/image_chopper.h"
#include "autoproduction/preprocessing/image_dtype_converter.h"
#include "autoproduction/utils/cuda_helper.h"

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include "autoproduction/utils/trt_helper.h"

#include "gtest/gtest.h"

namespace Autoproduction {
namespace Util {
namespace {
struct Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    if (severity != nvinfer1::ILogger::Severity::kVERBOSE) {
      std::clog << msg << "\n";
    }
  }
};
}  // namespace
class TrtEngineTest : public ::testing::Test {
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
  std::string path_to_the_model_ = "models/object_detection_football.onnx";

  cv::Mat img_;
  NppStreamContext ctx_;
};

TEST_F(TrtEngineTest, SanityCheck) {
  int target_height = 800;
  int target_width = 1280;

  constexpr int rows_num = 1;
  constexpr int columns_num = 3;
  auto chopper =
      Autoproduction::Preprocessing::ImageChopper<rows_num, columns_num>(
          img_.rows, img_.cols, target_height, target_width);

  Npp8u* img_ptr;
  size_t input_size = img_.rows * img_.cols * img_.channels() * sizeof(char);
  cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&img_ptr), input_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  cv::cuda::GpuMat cuda_img(img_.rows, img_.cols, CV_8UC3, img_ptr);
  cuda_img.upload(img_);

  Npp8u* dest_ptr;
  size_t dest_ptr_size = columns_num * target_height * target_width *
                         img_.channels() * sizeof(char);
  st = cudaMalloc(reinterpret_cast<void**>(&dest_ptr), dest_ptr_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  auto npp_st = chopper(img_ptr, dest_ptr, ctx_);
  EXPECT_EQ(npp_st, NPP_SUCCESS);

  Npp32f* converted_ptr;
  size_t converted_ptr_size = columns_num * target_height * target_width *
                              img_.channels() * sizeof(Npp32f);
  st = cudaMalloc(reinterpret_cast<void**>(&converted_ptr), converted_ptr_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  npp_st = Autoproduction::Preprocessing::ImageDTypeConverter(
      target_height, target_width, columns_num)(dest_ptr, converted_ptr, ctx_);
  EXPECT_EQ(npp_st, NPP_SUCCESS);

  TrtEngine trt_engine(path_to_the_model_, columns_num, target_height,
                       target_width, 3, logger_, cuda_stream_);
  std::cerr << "Ready to apply" << std::endl;

  auto dets = trt_engine(converted_ptr);
  EXPECT_EQ(3, dets.size());
}

}  // namespace Util
}  // namespace Autoproduction
