
#include "autoproduction/preprocessing/image_chopper.h"
#include "autoproduction/utils/cuda_helper.h"

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include "gtest/gtest.h"

namespace Autoproduction {
namespace Preprocessing {

class ImageChopperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    img_ = cv::imread("test_data/frames_0001.png", cv::IMREAD_COLOR);
    int device_id = 0;
    cudaError_t st = cudaSetDevice(device_id);
    EXPECT_TRUE(st == cudaSuccess);

    cudaStream_t cuda_stream;
    st = cudaStreamCreate(&cuda_stream);
    EXPECT_TRUE(st == cudaSuccess);

    ctx_ =
        Autoproduction::Util::fromCudaStreamAndDeviceId(cuda_stream, device_id);
  }
  cv::Mat img_;
  NppStreamContext ctx_;
};

TEST_F(ImageChopperTest, SimpleResize) {
  int target_height = 800;
  int target_width = 3840;
  auto chopper = ImageChopper<1, 1>(img_.rows, img_.cols);

  Npp8u* img_ptr;
  size_t input_size = img_.rows * img_.cols * img_.channels() * sizeof(char);
  cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&img_ptr), input_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  cv::cuda::GpuMat cuda_img(img_.rows, img_.cols, CV_8UC3, img_ptr);
  cuda_img.upload(img_);

  Npp8u* dest_ptr;
  size_t dest_ptr_size =
      target_height * target_width * img_.channels() * sizeof(char);
  st = cudaMalloc(reinterpret_cast<void**>(&dest_ptr), dest_ptr_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  cv::cuda::GpuMat cuda_dest_img(target_height, target_width, CV_8UC3,
                                 dest_ptr);
  auto npp_st = chopper(img_ptr, dest_ptr, ctx_);
  EXPECT_EQ(npp_st, NPP_SUCCESS);

  cv::Mat res;
  cuda_dest_img.download(res);

  cv::Mat exp = cv::imread("test_data/frames_0001.png");

  cv::Mat dst;
  cv::bitwise_xor(exp, res, dst);

  cv::Mat channels[3];
  cv::split(dst, channels);
  EXPECT_EQ(0, cv::countNonZero(channels[0]));
  EXPECT_EQ(0, cv::countNonZero(channels[1]));
  EXPECT_EQ(0, cv::countNonZero(channels[2]));
}

TEST_F(ImageChopperTest, ChopAndResize1x3) {
  int target_height = 800;
  int target_width = 1408;
  auto chopper = ImageChopper<1, 3>(img_.rows, img_.cols);

  Npp8u* img_ptr;
  size_t input_size = img_.rows * img_.cols * img_.channels() * sizeof(char);
  cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&img_ptr), input_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  cv::cuda::GpuMat cuda_img(img_.rows, img_.cols, CV_8UC3, img_ptr);
  cuda_img.upload(img_);

  Npp8u* dest_ptr;
  size_t dest_ptr_size =
      3 * target_height * target_width * img_.channels() * sizeof(char);
  st = cudaMalloc(reinterpret_cast<void**>(&dest_ptr), dest_ptr_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate input layer");
  }
  cv::cuda::GpuMat cuda_dest_img(3 * target_height, target_width, CV_8UC3,
                                 dest_ptr);
  auto npp_st = chopper(img_ptr, dest_ptr, ctx_);
  EXPECT_EQ(npp_st, NPP_SUCCESS);

  cv::Mat res;
  cuda_dest_img.download(res);

  cv::Mat exp = cv::imread("test_data/frames_0001_rescaled_1x3_720x1408.png");

  cv::Mat dst;
  cv::bitwise_xor(exp, res, dst);

  cv::Mat channels[3];
  cv::split(dst, channels);
  EXPECT_EQ(0, cv::countNonZero(channels[0]));
  EXPECT_EQ(0, cv::countNonZero(channels[1]));
  EXPECT_EQ(0, cv::countNonZero(channels[2]));
}

}  // namespace Preprocessing
}  // namespace Autoproduction
