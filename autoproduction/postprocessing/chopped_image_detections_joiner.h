#pragma once

#include <vector>
#include "autoproduction/preprocessing/image_grid.h"
#include "autoproduction/utils/detection.h"

namespace Autoproduction {
namespace Postprocessing {

template <int CropNumHeight, int CropNumWidth>
class ChoppedImageDetectionsJoiner {
 public:
  ChoppedImageDetectionsJoiner(int image_height, int image_width)
      : image_grid_(image_height, image_width),
        image_height_(image_height),
        image_width_(image_width) {}

  std::vector<Autoproduction::Util::Detection> operator()(
      const std::vector<std::vector<Autoproduction::Util::Detection>>&
          raw_detections) {
    std::vector<Autoproduction::Util::Detection> image_detections;
    for (std::size_t subimage_ix = 0; subimage_ix < raw_detections.size();
         ++subimage_ix) {
      for (const auto& detection : raw_detections[subimage_ix]) {
        image_detections.emplace_back(Autoproduction::Util::Detection(
            SubImageYToImageY(detection.bbox_.ymin_, subimage_ix),
            SubImageXToImageX(detection.bbox_.xmin_, subimage_ix),
            SubImageYToImageY(detection.bbox_.ymax_, subimage_ix),
            SubImageXToImageX(detection.bbox_.xmax_, subimage_ix),
            detection.score_, detection.class_name_));
      }
    }

    return image_detections;
  }

 private:
  float SubImageYToImageY(float value, std::size_t subimage_ix) noexcept {
    return (value * image_grid_.cells_[subimage_ix].height +
            image_grid_.cells_[subimage_ix].y) /
           static_cast<float>(image_height_);
  }
  float SubImageXToImageX(float value, std::size_t subimage_ix) noexcept {
    return (value * image_grid_.cells_[subimage_ix].width +
            image_grid_.cells_[subimage_ix].x) /
           static_cast<float>(image_width_);
  }

 private:
  Autoproduction::Preprocessing::Grid<CropNumHeight, CropNumWidth> image_grid_;
  int image_height_;
  int image_width_;
};
}  // namespace Postprocessing
}  // namespace Autoproduction
