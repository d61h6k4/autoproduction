#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include "NvInfer.h"
#include "autoproduction/postprocessing/chopped_image_detections_joiner.h"
#include "autoproduction/preprocessing/image_chopper.h"
#include "autoproduction/preprocessing/image_dtype_converter.h"
#include "autoproduction/utils/cuda_helper.h"
#include "autoproduction/utils/trt_helper.h"

namespace Autoproduction {
namespace Inference {

template <int CropNumHeight, int CropNumWidth>
class ObjectDetectionModel {
 public:
  ObjectDetectionModel(const std::string& path_to_the_onnx_model,
                       int image_height, int image_width,
                       int model_image_height, int model_image_width,
                       cudaStream_t cuda_stream, int device_id,
                       std::shared_ptr<nvinfer1::ILogger> logger)
      : image_chopper_(image_height, image_width),
        image_dtype_converter_(image_chopper_.TargetHeight(),
                               image_chopper_.TargetWidth(),
                               CropNumHeight * CropNumWidth),
        chopped_image_detections_joiner_(image_height, image_width),
        trt_engine_(path_to_the_onnx_model, CropNumHeight * CropNumWidth,
                    image_chopper_.TargetHeight(), image_chopper_.TargetWidth(),
                    model_image_height, model_image_width, 3, logger,
                    cuda_stream) {
    size_t chopped_images_size =
        CropNumHeight * CropNumWidth * image_chopper_.TargetHeight() *
        image_chopper_.TargetWidth() * 3 * sizeof(Npp8u);
    auto st = cudaMalloc(reinterpret_cast<void**>(&chopped_images_),
                         chopped_images_size);
    if (st != cudaSuccess) {
      throw std::runtime_error("Could not allocate memory for chopped images");
    }
    size_t float_images_size =
        CropNumHeight * CropNumWidth * image_chopper_.TargetHeight() *
        image_chopper_.TargetWidth() * 3 * sizeof(Npp32f);
    st =
        cudaMalloc(reinterpret_cast<void**>(&float_images_), float_images_size);
    if (st != cudaSuccess) {
      throw std::runtime_error(
          "Could not allocate memory for float version of images");
    }

    npp_stream_ =
        Autoproduction::Util::fromCudaStreamAndDeviceId(cuda_stream, device_id);
    logger_ = logger;
  }

  std::vector<Autoproduction::Util::Detection> operator()(const Npp8u* img) {
    auto npp_st = image_chopper_(img, chopped_images_, npp_stream_);
    if (npp_st != NPP_SUCCESS) {
      logger_->log(nvinfer1::ILogger::Severity::kERROR,
                   "Image chopper returned error code. Very unexpected. If "
                   "error repeats then better rerun the program.");
      return {};
    }

    npp_st =
        image_dtype_converter_(chopped_images_, float_images_, npp_stream_);
    if (npp_st != NPP_SUCCESS) {
      logger_->log(
          nvinfer1::ILogger::Severity::kERROR,
          "Image convert to float returned error code. Very unexpected. If "
          "error repeats then better rerun the program.");
      return {};
    }
    return chopped_image_detections_joiner_(trt_engine_(float_images_));
  }

 private:
  Npp8u* chopped_images_;
  Npp32f* float_images_;

  NppStreamContext npp_stream_;

  Autoproduction::Preprocessing::ImageChopper<CropNumHeight, CropNumWidth>
      image_chopper_;
  Autoproduction::Preprocessing::ImageDTypeConverter image_dtype_converter_;
  Autoproduction::Postprocessing::ChoppedImageDetectionsJoiner<CropNumHeight,
                                                               CropNumWidth>
      chopped_image_detections_joiner_;

  Autoproduction::Util::TrtEngine trt_engine_;

  std::shared_ptr<nvinfer1::ILogger> logger_;
};
}  // namespace Inference
}  // namespace Autoproduction
