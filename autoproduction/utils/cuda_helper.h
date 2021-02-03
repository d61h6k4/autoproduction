
#include <cuda_runtime.h>
#include <nppdefs.h>

namespace Autproduction {
namespace Util {
NppStreamContext fromCudaStreamAndDeviceId(cudaStream_t cuda_stream,
                                           int device_id);
}
}  // namespace Autproduction
