
#include "autoproduction/utils/trt_helper.h"

#include "NvOnnxParser.h"

namespace Autoproduction {
namespace Util {
TrtEngine::TrtEngine(const std::string& path_to_the_model, int batch_size,
                     int image_height, int image_width, int model_image_height,
                     int model_image_width, int image_channel,
                     std::shared_ptr<nvinfer1::ILogger> logger,
                     cudaStream_t stream) {
  if (!logger) {
    throw std::runtime_error("logger can not be nullptr.");
  }
  logger_ = std::move(logger);

  batch_size_ = batch_size;
  image_height_ = image_height;
  image_width_ = image_width;

  model_image_height_ = model_image_height;
  model_image_width_ = model_image_width;
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

  int input_index = engine_->getBindingIndex("inputs:0");
  auto input_dim = engine_->getBindingDimensions(input_index);
  if (batch_size_ != input_dim.d[0] || image_height_ != input_dim.d[1] ||
      image_width_ != input_dim.d[2] || image_channel_ != input_dim.d[3]) {
    throw std::runtime_error("The model input and TrtEngine input don't match");
  }

  {
    int detection_boxes_index = engine_->getBindingIndex("detection_boxes");
    auto detection_boxes_dim =
        engine_->getBindingDimensions(detection_boxes_index);
    detection_boxes_num_ = detection_boxes_dim.d[1];
    detection_boxes_size_ = detection_boxes_dim.d[2];

    if (batch_size_ != detection_boxes_dim.d[0] || detection_boxes_size_ != 4) {
      throw std::runtime_error(
          "The model output (detection boxes) doesn't match with expectation");
    }

    size_t output_size = batch_size_ * detection_boxes_num_ *
                         detection_boxes_size_ * sizeof(float);

    auto st = cudaMalloc(reinterpret_cast<void**>(&detection_boxes_layer_),
                         output_size);
    if (st != cudaSuccess) {
      throw std::runtime_error("Could not allocate output layer");
    }
  }
  {
    int scores_index = engine_->getBindingIndex("detection_scores");
    auto scores_dim = engine_->getBindingDimensions(scores_index);
    scores_num_ = scores_dim.d[1];

    if (batch_size_ != scores_dim.d[0]) {
      throw std::runtime_error(
          "The model output (detection scores) doesn't match with expectation");
    }

    size_t output_size = batch_size_ * scores_num_ * sizeof(float);

    auto st = cudaMalloc(reinterpret_cast<void**>(&scores_layer_), output_size);
    if (st != cudaSuccess) {
      throw std::runtime_error("Could not allocate output layer");
    }
  }

  {
    int classes_index = engine_->getBindingIndex("detection_classes");
    auto classes_dim = engine_->getBindingDimensions(classes_index);
    classes_num_ = classes_dim.d[1];
    if (batch_size_ != classes_dim.d[0]) {
      throw std::runtime_error(
          "The model output (detection classes) doesn't match with "
          "expectation");
    }

    size_t output_size = batch_size_ * classes_num_ * sizeof(float);

    auto st =
        cudaMalloc(reinterpret_cast<void**>(&classes_layer_), output_size);
    if (st != cudaSuccess) {
      throw std::runtime_error("Could not allocate output layer");
    }
  }

  {
    int num_detections_index = engine_->getBindingIndex("num_detections");
    auto num_detections_dim =
        engine_->getBindingDimensions(num_detections_index);
    num_detections_size_ = num_detections_dim.d[1];

    if (batch_size_ != num_detections_dim.d[0]) {
      throw std::runtime_error(
          "The model output (detection num) doesn't match with expectation");
    }

    size_t output_size = batch_size_ * num_detections_size_ * sizeof(float);
    auto st = cudaMalloc(reinterpret_cast<void**>(&num_detections_layer_),
                         output_size);
    if (st != cudaSuccess) {
      throw std::runtime_error("Could not allocate output layer");
    }
  }
}

std::vector<Detections> TrtEngine::Postprocess(
    const std::vector<float>& detection_boxes, const std::vector<float>& scores,
    const std::vector<float>& classes,
    const std::vector<float>& num_detections) {
  std::vector<Detections> detections(batch_size_);
  for (auto batch_ix = 0; batch_ix < batch_size_; ++batch_ix) {
    int batch_output_start_ix = batch_ix * detection_boxes_num_;

    // detections_num is stored as a 7-th value
    int detections_num = 100;  // static_cast<int>(num_detections[batch_ix]);
    detections[batch_ix].reserve(detections_num);
    for (auto detection_ix = 0; detection_ix < detections_num; ++detection_ix) {
      auto in_batch_detection_ix =
          batch_output_start_ix * detection_boxes_size_ +
          detection_ix * detection_boxes_size_;
      detections[batch_ix].push_back(
          Detection(detection_boxes[in_batch_detection_ix + 0] /
                        static_cast<float>(model_image_height_),
                    detection_boxes[in_batch_detection_ix + 1] /
                        static_cast<float>(model_image_width_),
                    detection_boxes[in_batch_detection_ix + 2] /
                        static_cast<float>(model_image_height_),
                    detection_boxes[in_batch_detection_ix + 3] /
                        static_cast<float>(model_image_width_),
                    scores[batch_output_start_ix + detection_ix],
                    classes[batch_output_start_ix + detection_ix]));
    }
  }
  return detections;
}

std::vector<Detections> TrtEngine::operator()(float* img) {
  void* buffers[5] = {img, num_detections_layer_, detection_boxes_layer_,
                      scores_layer_, classes_layer_};
  bool status = context_->enqueueV2(buffers, cuda_stream_, nullptr);

  if (!status) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to call TRT engine");
    return {};
  }
  size_t detection_boxes_size =
      batch_size_ * detection_boxes_num_ * detection_boxes_size_;

  std::vector<float> detection_boxes(detection_boxes_size);
  cudaError_t st =
      cudaMemcpyAsync(detection_boxes.data(), detection_boxes_layer_,
                      detection_boxes_size * sizeof(float),
                      cudaMemcpyDeviceToHost, cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to copy data from detection_boxes layer");
    return {};
  }

  size_t scores_size = batch_size_ * scores_num_;

  std::vector<float> scores(scores_size);
  st =
      cudaMemcpyAsync(scores.data(), scores_layer_, scores_size * sizeof(float),
                      cudaMemcpyDeviceToHost, cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to copy data from scores layer");
    return {};
  }
  size_t classes_size = batch_size_ * classes_num_;

  std::vector<float> classes(classes_size);
  st = cudaMemcpyAsync(classes.data(), classes_layer_,
                       classes_size * sizeof(float), cudaMemcpyDeviceToHost,
                       cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to copy data from classes layer");
    return {};
  }

  size_t num_detections_size = batch_size_ * num_detections_size_;

  std::vector<float> num_detections(num_detections_size);
  st = cudaMemcpyAsync(num_detections.data(), num_detections_layer_,
                       num_detections_size * sizeof(float),
                       cudaMemcpyDeviceToHost, cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to copy data from num_detection layer");
    return {};
  }

  st = cudaStreamSynchronize(cuda_stream_);
  if (st != cudaSuccess) {
    logger_->log(nvinfer1::ILogger::Severity::kERROR,
                 "Failed to synchornize CUDA stream");
    return {};
  }

  return Postprocess(detection_boxes, scores, classes, num_detections);
}

TrtEngine::~TrtEngine() { cudaFree(detection_boxes_layer_); }
}  // namespace Util
}  // namespace Autoproduction
