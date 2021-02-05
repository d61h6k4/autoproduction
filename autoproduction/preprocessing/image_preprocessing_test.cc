
#include "autoproduction/preprocessing/image_chopper.h"
#include "autoproduction/preprocessing/image_dtype_converter.h"
#include "autoproduction/utils/cuda_helper.h"

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include "gtest/gtest.h"

namespace Autoproduction {
namespace Preprocessing {
class ImagePreprocessingEndToEndTest : public ::testing::Test {
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

TEST_F(ImagePreprocessingEndToEndTest, ChopAndResize1x3AndConvert) {
  int target_height = 800;
  int target_width = 1280;

  constexpr int rows_num = 1;
  constexpr int columns_num = 3;
  auto chopper = ImageChopper<rows_num, columns_num>(
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
  npp_st = ImageDTypeConverter(target_height, target_width, columns_num)(
      dest_ptr, converted_ptr, ctx_);
  EXPECT_EQ(npp_st, NPP_SUCCESS);

  cv::cuda::GpuMat cuda_dest_img(columns_num * target_height, target_width,
                                 CV_32FC3, converted_ptr);

  cv::Mat res;
  cuda_dest_img.download(res);

  cv::Mat exp_u8 =
      cv::imread("test_data/frames_0001_rescaled_1x3_720x1280.png");
  cv::Mat exp;
  exp_u8.convertTo(exp, CV_32FC3);

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
