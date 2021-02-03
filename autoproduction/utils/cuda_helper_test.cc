
#include "autoproduction/utils/cuda_helper.h"

#include <cuda_runtime_api.h>

#include "gtest/gtest.h"

namespace Autproduction {
namespace Util {
TEST(FromCudaStreamTest, SanityCheck) {
  int device_id = 0;
  cudaError_t st = cudaSetDevice(device_id);
  EXPECT_TRUE(st == cudaSuccess);

  cudaStream_t cuda_stream;
  st = cudaStreamCreate(&cuda_stream);
  EXPECT_TRUE(st == cudaSuccess);

  auto ctx = fromCudaStreamAndDeviceId(cuda_stream, device_id);
  EXPECT_EQ(cuda_stream, ctx.hStream);
  EXPECT_EQ(device_id, ctx.nCudaDeviceId);
}
}  // namespace Util
}  // namespace Autproduction
