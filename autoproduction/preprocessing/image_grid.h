#pragma once

#include <nppcore.h>
#include <array>
#include <cmath>

namespace Autoproduction {
namespace Preprocessing {

template <int CropNumHeight, int CropNumWidth>
struct Grid {
  Grid(int image_height, int image_width) {
    float overlap_size = 0.1;

    int step_height = image_height;
    target_height_ = image_height;
    if (CropNumHeight > 1) {
      step_height = ceilf(static_cast<float>(image_height) /
                          static_cast<float>(CropNumHeight));
      target_height_ = static_cast<int>(
          ceilf((1.f + overlap_size) * static_cast<float>(step_height)));
    }

    int step_width = image_width;
    target_width_ = image_width;
    if (CropNumWidth > 1) {
      step_width = ceilf(static_cast<float>(image_width) /
                         static_cast<float>(CropNumWidth));
      target_width_ = static_cast<int>(
          ceilf((1.f + overlap_size) * static_cast<float>(step_width)));
    }

    int offset_height = 0;
    int offset_width = 0;
    std::size_t offset_ix = 0;
    for (int row_ix = 0; row_ix < CropNumHeight; ++row_ix) {
      offset_height = row_ix * step_height;

      for (int column_ix = 0; column_ix < CropNumWidth; ++column_ix) {
        offset_width = column_ix * step_width;

        if (image_width < offset_width + target_width_) {
          offset_width = image_width - target_width_;
        }

        if (image_height < offset_height + target_height_) {
          offset_height = image_height - target_height_;
        }

        offset_ix = row_ix * CropNumWidth + column_ix;
        cells_[offset_ix] = NppiRect{.x = offset_width,
                                     .y = offset_height,
                                     .width = target_width_,
                                     .height = target_height_};
      }
    }
  }

  std::array<NppiRect, CropNumHeight * CropNumWidth> cells_;
  int target_height_;
  int target_width_;
};
}  // namespace Preprocessing
}  // namespace Autoproduction
