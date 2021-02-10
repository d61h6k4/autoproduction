#pragma once

#include <cuda_runtime.h>
#include <nppdefs.h>

namespace Autoproduction {
namespace Util {
NppStreamContext fromCudaStreamAndDeviceId(cudaStream_t cuda_stream,
                                           int device_id);
}
}  // namespace Autoproduction
