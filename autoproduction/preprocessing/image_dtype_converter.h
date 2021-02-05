#pragma once

#include <nppcore.h>

namespace Autoproduction {
namespace Preprocessing {

// Convert the given images from
// unsigned char to float 32bit.
class ImageDTypeConverter {
 public:
  ImageDTypeConverter(int image_height, int image_width, int batch_size)
      : image_height_(image_height),
        image_width_(image_width),
        batch_size_(batch_size){};

  NppStatus operator()(const Npp8u* p_src, Npp32f* p_dst, NppStreamContext ctx);

 private:
  int image_height_;
  int image_width_;
  int batch_size_;
};
}  // namespace Preprocessing
}  // namespace Autoproduction
