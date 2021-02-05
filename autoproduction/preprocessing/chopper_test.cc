
#include "autoproduction/preprocessing/chopper.h"
#include "autoproduction/utils/cuda_helper.h"

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include "gtest/gtest.h"

namespace Autoproduction {
namespace Preprocessing {
namespace {
constexpr std::size_t ORIG_IMAGE_HEIGHT = 800;
constexpr std::size_t ORIG_IMAGE_WIDTH = 3840;

}  // namespace

class GridTest : public ::testing::Test {};

TEST_F(GridTest, IdentityTargetHeight) {
  Grid<1, 1> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(ORIG_IMAGE_HEIGHT, g.cells_[0].height);
}

TEST_F(GridTest, IdentityTargetWidth) {
  Grid<1, 1> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(static_cast<float>(ORIG_IMAGE_WIDTH), g.cells_[0].width);
}

TEST_F(GridTest, Grid1x3) {
  Grid<1, 3> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));

  EXPECT_EQ(1408.f, g.cells_[0].width);

  EXPECT_EQ(3, g.cells_.size());

  EXPECT_EQ(0.f, g.cells_[0].y);
  EXPECT_EQ(0.f, g.cells_[0].x);

  EXPECT_EQ(0.f, g.cells_[1].y);
  EXPECT_EQ(1280.f, g.cells_[1].x);

  EXPECT_EQ(0.f, g.cells_[2].y);
  EXPECT_EQ(2432.f, g.cells_[2].x);
}

TEST_F(GridTest, Grid2x3) {
  Grid<2, 3> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));

  EXPECT_EQ(1408.f, g.cells_[0].width);

  EXPECT_EQ(6, g.cells_.size());

  EXPECT_EQ(0.f, g.cells_[0].y);
  EXPECT_EQ(0.f, g.cells_[0].x);

  EXPECT_EQ(0.f, g.cells_[1].y);
  EXPECT_EQ(1280.f, g.cells_[1].x);

  EXPECT_EQ(0.f, g.cells_[2].y);
  EXPECT_EQ(2432.f, g.cells_[2].x);

  EXPECT_EQ(360.f, g.cells_[3].y);
  EXPECT_EQ(0.f, g.cells_[3].x);

  EXPECT_EQ(360.f, g.cells_[4].y);
  EXPECT_EQ(1280.f, g.cells_[4].x);

  EXPECT_EQ(360.f, g.cells_[5].y);
  EXPECT_EQ(2432.f, g.cells_[5].x);
}

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
  int target_height = 720;
  int target_width = 1280;
  auto chopper =
      ImageChopper<1, 1>(img_.rows, img_.cols, target_height, target_width);

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

  cv::Mat exp = cv::imread("test_data/frames_0001_rescaled_1x1_720x1280.png");

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
  int target_width = 1280;
  auto chopper =
      ImageChopper<1, 3>(img_.rows, img_.cols, target_height, target_width);

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

  cv::Mat exp = cv::imread("test_data/frames_0001_rescaled_1x3_720x1280.png");

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
