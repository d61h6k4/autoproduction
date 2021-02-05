
#include "autoproduction/utils/cuda_helper.h"

#include <cuda_runtime_api.h>

#include <stdexcept>

namespace Autoproduction {
namespace Util {
NppStreamContext fromCudaStreamAndDeviceId(cudaStream_t cuda_stream,
                                           int device_id) {
  cudaDeviceProp prop;
  cudaError_t st = cudaGetDeviceProperties(&prop, device_id);
  if (st != cudaSuccess) {
    throw std::runtime_error(
        "cudaGetDeviceProperties returns cudaErrorInvalidDevice");
  }

  NppStreamContext ctx;
  ctx.hStream = cuda_stream;
  ctx.nCudaDeviceId = device_id;
  ctx.nMultiProcessorCount = prop.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;

  st = cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                              cudaDevAttrComputeCapabilityMajor, device_id);
  if (st != cudaSuccess) {
    if (st == cudaErrorInvalidDevice) {
      throw std::runtime_error(
          "cudaGetDeviceGetAttrubute returns cudaErrorInvalidDevice");
    } else if (st == cudaErrorInvalidValue) {
      throw std::runtime_error(
          "cudaGetDeviceGetAttrubute for value "
          "cudaDevAttrComputeCapabilityMajor returns cudaErrorInvalidValue");
    } else {
      throw std::runtime_error(
          "cudaGetDeviceGetAttrubute returns unexpected error code");
    }
  }

  st = cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                              cudaDevAttrComputeCapabilityMinor, device_id);
  if (st != cudaSuccess) {
    if (st == cudaErrorInvalidDevice) {
      throw std::runtime_error(
          "cudaGetDeviceGetAttrubute returns cudaErrorInvalidDevice");
    } else if (st == cudaErrorInvalidValue) {
      throw std::runtime_error(
          "cudaGetDeviceGetAttrubute for value "
          "cudaDevAttrComputeCapabilityMinor returns cudaErrorInvalidValue");
    } else {
      throw std::runtime_error(
          "cudaGetDeviceGetAttrubute returns unexpected error code");
    }
  }

  st = cudaStreamGetFlags(cuda_stream, &ctx.nStreamFlags);
  if (st != cudaSuccess) {
    if (st == cudaErrorInvalidValue) {
      throw std::runtime_error(
          "cudaStreamGetFlags returns cudaErrorInvalidValue");
    } else if (st == cudaErrorInvalidResourceHandle) {
      throw std::runtime_error(
          "cudaStreamGetFlags returns cudaErrorInvalidResourceHandle");
    } else {
      throw std::runtime_error(
          "cudaStreamGetFlags returns unexpected error code");
    }
  }

  return ctx;
}
}  // namespace Util
}  // namespace Autoproduction
