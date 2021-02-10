#pragma once

#include <nppcore.h>
#include <nppi_geometry_transforms.h>
#include <array>

#include "autoproduction/preprocessing/image_grid.h"

namespace Autoproduction {
namespace Preprocessing {

// ImageChopper - chops the given image by grid according the
// given row and column num. Return the crops as a batch
// of images.
//
// Grid's cells overlap (you can find more in tests)
//
// !NOTE: works only for unsigned int 3 channel images.
template <int CropNumHeight, int CropNumWidth>
class ImageChopper {
 public:
  ImageChopper(int image_height, int image_width, int target_image_height,
               int target_image_width)
      : image_grid_(image_height, image_width),
        image_height_(image_height),
        image_width_(image_width),
        target_image_height_(target_image_height),
        target_image_width_(target_image_width) {}

  NppStatus operator()(const Npp8u* p_src, Npp8u* p_dst, NppStreamContext ctx) {
    constexpr int channel_num = 3;

    NppStatus st;
    std::size_t dst_ix = 0;
    const std::size_t target_image_size =
        target_image_height_ * target_image_width_ * channel_num * sizeof(char);
    for (std::size_t ix = 0; ix < CropNumHeight * CropNumWidth; ++ix) {
      dst_ix = ix * target_image_size;

      st = nppiResize_8u_C3R_Ctx(
          p_src, image_width_ * channel_num,
          NppiSize{.width = image_width_, .height = image_height_},
          image_grid_.cells_[ix], p_dst + dst_ix,
          target_image_width_ * channel_num,
          NppiSize{.width = target_image_width_,
                   .height = target_image_height_},
          NppiRect{.x = 0,
                   .y = 0,
                   .width = target_image_width_,
                   .height = target_image_height_},
          NPPI_INTER_NN, ctx);
      if (st != NPP_SUCCESS) {
        return st;
      }
    }
    return st;
  }

 private:
  Grid<CropNumHeight, CropNumWidth> image_grid_;
  int image_height_;
  int image_width_;
  int target_image_height_;
  int target_image_width_;
};

// Special version of image chopper because
// grid 1x1 is nogrid.
// This class equal to Resize
template <>
class ImageChopper<1, 1> {
 public:
  ImageChopper(int image_height, int image_width, int target_image_height,
               int target_image_width)
      : image_height_(image_height),
        image_width_(image_width),
        target_image_height_(target_image_height),
        target_image_width_(target_image_width) {}

  NppStatus operator()(const Npp8u* p_src, Npp8u* p_dst, NppStreamContext ctx);

 private:
  int image_height_;
  int image_width_;
  int target_image_height_;
  int target_image_width_;
};
}  // namespace Preprocessing
}  // namespace Autoproduction
