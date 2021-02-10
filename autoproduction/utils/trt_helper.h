
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "NvInfer.h"
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
  TrtEngine(const std::string& path_to_the_model,
            std::shared_ptr<nvinfer1::ILogger> logger, cudaStream_t stream);
  ~TrtEngine();

  std::vector<Detections> operator()(float* img);

 private:
  void BuildEngine(const std::string& path_to_the_model);
  void SetModel();
  std::vector<Detections> Postprocess(const std::vector<float>& raw_detections);

 private:
  cudaStream_t cuda_stream_;

  std::shared_ptr<nvinfer1::ILogger> logger_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  int batch_size_ = 3;
  int image_height_ = 800;
  int image_width_ = 1280;
  int image_channel_ = 3;
  int output_detections_num_;
  int output_detections_size_;

  float* output_layer_;
};

}  // namespace Util
}  // namespace Autoproduction
