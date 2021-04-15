#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "autoproduction/utils/detection.h"

namespace Autoproduction {
namespace Util {
// Helper class to clean
// the objects with smart pointers
struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

using Detections = std::vector<Detection>;

class TrtEngine {
 public:
  TrtEngine(const std::string& path_to_the_model, bool is_engine,
            int batch_size, int image_height, int image_width,
            int model_image_height, int model_image_width, int image_channel,
            std::shared_ptr<nvinfer1::ILogger> logger, cudaStream_t stream);
  ~TrtEngine();

  std::vector<Detections> operator()(float* img);

 private:
  void BuildEngine(const std::string& path_to_the_model, bool is_engine);
  void SetModel();
  std::vector<Detections> Postprocess(const std::vector<float>& detection_boxes,
                                      const std::vector<float>& scores,
                                      const std::vector<float>& classes,
                                      const std::vector<float>& num_detections);

 private:
  cudaStream_t cuda_stream_;

  std::shared_ptr<nvinfer1::ILogger> logger_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  int batch_size_;
  // These are input image size
  int image_height_;
  int image_width_;

  // These are output (model) image size
  // they may be differ then input image size
  // because model may have resize step
  int model_image_height_;
  int model_image_width_;
  int image_channel_;

  int detection_boxes_num_;
  int detection_boxes_size_;
  int scores_num_;
  int classes_num_;
  int num_detections_size_;

  float* detection_boxes_layer_;
  float* scores_layer_;
  float* classes_layer_;
  float* num_detections_layer_;
};

}  // namespace Util
}  // namespace Autoproduction
