
#include "autoproduction/preprocessing/image_dtype_converter.h"
#include <nppi_data_exchange_and_initialization.h>

namespace Autoproduction {
namespace Preprocessing {
NppStatus ImageDTypeConverter::operator()(const Npp8u* p_src, Npp32f* p_dst,
                                          NppStreamContext ctx) {
  constexpr int channel_num = 3;
  const int src_image_line_step =
      static_cast<int>(image_width_ * channel_num * sizeof(Npp8u));
  const int dst_image_line_step =
      static_cast<int>(image_width_ * channel_num * sizeof(Npp32f));
  return nppiConvert_8u32f_C3R_Ctx(
      p_src, src_image_line_step, p_dst, dst_image_line_step,
      NppiSize{.width = image_width_, .height = image_height_ * batch_size_},
      ctx);
}
}  // namespace Preprocessing
}  // namespace Autoproduction
