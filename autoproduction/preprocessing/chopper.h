#pragma once

#include <array>
#include <cmath>
#include <nppcore.h>

namespace Autoproduction {
namespace Preprocessing {

template <std::size_t CropNumHeight, std::size_t CropNumWidth>
struct Grid {
  Grid(float image_height, float image_width) {
    float overlap_size = 0.1;

    float step_height = image_height;
    target_height_ = image_height;
    if (CropNumHeight > 1) {
      step_height = ceilf(image_height / static_cast<float>(CropNumHeight));
      target_height_ = ceilf((1.f + overlap_size) * step_height);
    }

    float step_width = image_width;
    target_width_ = image_width;
    if (CropNumWidth > 1) {
      step_width = ceilf(image_width / static_cast<float>(CropNumWidth));
      target_width_ = ceilf((1.f + overlap_size) * step_width);
    }

    float offset_height = 0.f;
    float offset_width = 0.f;
    std::size_t offset_ix = 0;
    for (std::size_t row_ix = 0; row_ix < CropNumHeight; ++row_ix) {
      offset_height = static_cast<float>(row_ix) * step_height;

      for (std::size_t column_ix = 0; column_ix < CropNumWidth; ++column_ix) {
        offset_width = static_cast<float>(column_ix) * step_width;

        if (image_width < offset_width + target_width_) {
          offset_width = image_width - target_width_;
        }

        offset_ix = row_ix * CropNumWidth + column_ix;
        offsets_width_[offset_ix] = offset_width;

        if (image_height < offset_height + target_height_) {
          offset_height = image_height - target_height_;
        }
        offsets_height_[offset_ix] = offset_height;
      }
    }
  }

  std::array<float, CropNumHeight * CropNumWidth> offsets_height_;
  std::array<float, CropNumHeight * CropNumWidth> offsets_width_;
  float target_height_;
  float target_width_;
};

template <std::size_t CropNumHeight, std::size_t CropNumWidth>
class ImageChopper {
 public:
  ImageChopper(float image_height, float image_width, float target_image_height,
               float target_image_width)
      : image_grid_(image_height, image_width) {}

 private:
  Grid<CropNumHeight, CropNumWidth> image_grid_;
};

template <>
class ImageChopper<1, 1> {
 public:
  ImageChopper(float image_height, float image_width, float target_image_height,
               float target_image_width)
      : image_height_(image_height),
        image_width_(image_width),
        target_image_height_(target_image_height),
        target_image_width_(target_image_width) {}

  
  NppStatus operator()(const Npp8u* p_src, Npp8u* p_dst);

 private:
  float image_height_;
  float image_width_;
  float target_image_height_;
  float target_image_width_;
};
}  // namespace Preprocessing
}  // namespace Autoproduction
