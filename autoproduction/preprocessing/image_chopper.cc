
#include "autoproduction/preprocessing/image_chopper.h"

namespace Autoproduction {
namespace Preprocessing {
NppStatus ImageChopper<1, 1>::operator()(const Npp8u* p_src, Npp8u* p_dst,
                                         NppStreamContext ctx) {
  constexpr int channel_num = 3;
  const int src_image_line_step =
      static_cast<int>(image_width_ * channel_num * sizeof(Npp8u));
  const int dst_image_line_step =
      static_cast<int>(target_image_width_ * channel_num * sizeof(Npp8u));
  return nppiResize_8u_C3R_Ctx(
      p_src, src_image_line_step,
      NppiSize{.width = image_width_, .height = image_height_},
      NppiRect{.x = 0, .y = 0, .width = image_width_, .height = image_height_},
      p_dst, dst_image_line_step,
      NppiSize{.width = target_image_width_, .height = target_image_height_},
      NppiRect{.x = 0,
               .y = 0,
               .width = target_image_width_,
               .height = target_image_height_},
      NPPI_INTER_NN, ctx);
}
}  // namespace Preprocessing
}  // namespace Autoproduction
