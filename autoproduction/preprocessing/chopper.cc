
#include "autoproduction/preprocessing/chopper.h"

namespace Autoproduction {
namespace Preprocessing {
NppStatus ImageChopper<1, 1>::operator()(const Npp8u* p_src, Npp8u* p_dst) {
  constexpr int channel_num = 3;
  return nppiResize_8u_C3R(
      p_src, image_width_ * channel_num,
      NppiSize{.width = image_width_, .height = image_height_},
      NppiRect{.x = 0, .y = 0, .width = image_width_, .height = image_height_},
      p_dst, target_image_width_ * channel_num,
      NppiSize{.width = target_image_width_, .height = target_image_height_},
      NppiRect{.x = 0,
               .y = 0,
               .width = target_image_width_,
               .height = target_image_height_},
      NPPI_INTER_NN);
}
}  // namespace Preprocessing
}  // namespace Autoproduction
