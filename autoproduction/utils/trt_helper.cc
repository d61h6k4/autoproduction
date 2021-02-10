
#include "autoproduction/utils/trt_helper.h"

#include "NvOnnxParser.h"

namespace Autoproduction {
namespace Util {
TrtEngine::TrtEngine(const std::string& path_to_the_model, int batch_size,
                     int image_height, int image_width, int image_channel,
                     std::shared_ptr<nvinfer1::ILogger> logger,
                     cudaStream_t stream) {
  if (!logger) {
    throw std::runtime_error("logger can not be nullptr.");
  }
  logger_ = std::move(logger);

  batch_size_ = batch_size;
  image_height_ = image_height;
  image_width_ = image_width;
  image_channel_ = image_channel;

  cuda_stream_ = stream;

  BuildEngine(path_to_the_model);
  SetModel();
}

void TrtEngine::BuildEngine(const std::string& path_to_the_model) {
  auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(
      nvinfer1::createInferBuilder(*logger_));
  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(
      builder->createNetworkV2(explicit_batch));
  auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(
      nvonnxparser::createParser(*network, *logger_));
  parser->parseFromFile(
      path_to_the_model.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

  if (parser->getNbErrors() > 0) {
    throw std::runtime_error(parser->getError(0)->desc());
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(
      builder->createBuilderConfig());
  // 1024 MB
  config->setMaxWorkspaceSize(1 << 30);
  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  builder->setMaxBatchSize(batch_size_);

  auto profile = builder->createOptimizationProfile();
  auto input_name = network->getInput(0)->getName();
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims4{batch_size_, image_height_,
                                         image_width_, image_channel_});
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims4{batch_size_, image_height_,
                                         image_width_, image_channel_});
  profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims4{batch_size_, image_height_,
                                         image_width_, image_channel_});
  config->addOptimizationProfile(profile);
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config), InferDeleter());
}

void TrtEngine::SetModel() {
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext(), InferDeleter());
  if (!context_) {
    throw std::runtime_error("Failed to create execution context");
  }

  int input_index = engine_->getBindingIndex("image:0");
  auto input_dim = engine_->getBindingDimensions(input_index);
  if (batch_size_ != input_dim.d[0] || image_height_ != input_dim.d[1] ||
      image_width_ != input_dim.d[2] || image_channel_ != input_dim.d[3]) {
    throw std::runtime_error("The model input and TrtEngine input don't match");
  }

  int output_index = engine_->getBindingIndex("Identity:0");
  auto output_dim = engine_->getBindingDimensions(output_index);
  output_detections_num_ = output_dim.d[1];
  output_detections_size_ = output_dim.d[2];
  if (batch_size_ != output_dim.d[0] || output_detections_size_ != 7) {
    throw std::runtime_error("The model output doesn't match with expectation");
  }

  size_t output_size = batch_size_ * output_detections_num_ *
                       output_detections_size_ * sizeof(float);

  auto st = cudaMalloc(reinterpret_cast<void**>(&output_layer_), output_size);
  if (st != cudaSuccess) {
    throw std::runtime_error("Could not allocate output layer");
  }
}

std::vector<Detections> TrtEngine::Postprocess(
    const std::vector<float>& output) {
  std::vector<Detections> detections(batch_size_);
  for (auto batch_ix = 0; batch_ix < batch_size_; ++batch_ix) {
    int batch_output_start_ix =
        batch_ix * output_detections_num_ * output_detections_size_;

    // detections_num is stored as a 7-th value
    int detections_num = static_cast<int>(output[batch_output_start_ix + 6]);
    detections[batch_ix].reserve(detections_num);
    for (auto detection_ix = 0; detection_ix < detections_num; ++detection_ix) {
      auto in_batch_detection_ix =
          batch_output_start_ix + detection_ix * output_detections_size_;
      detections[batch_ix].push_back(Detection(
          output[in_batch_detection_ix + 0], output[in_batch_detection_ix + 1],
          output[in_batch_detection_ix + 2], output[in_batch_detection_ix + 3],
          output[in_batch_detection_ix + 4],
          output[in_batch_detection_ix + 5]));
    }
  }
  return detections;
}

std::vector<Detections> TrtEngine::operator()(float* img) {
  void* buffers[2] = {img, output_layer_};
  bool status = context_->enqueueV2(buffers, cuda_stream_, nullptr);

  if (!status) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to call TRT engine");
    return {};
  }

  size_t output_size =
      batch_size_ * output_detections_num_ * output_detections_size_;
  std::vector<float> output(output_size);
  cudaError_t st =
      cudaMemcpyAsync(output.data(), output_layer_, output_size * sizeof(float),
                      cudaMemcpyDeviceToHost, cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to copy data from output layer");
    return {};
  }
  st = cudaStreamSynchronize(cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to synchornize CUDA stream");
    return {};
  }

  return Postprocess(output);
}

TrtEngine::~TrtEngine() { cudaFree(output_layer_); }
}  // namespace Util
}  // namespace Autoproduction
