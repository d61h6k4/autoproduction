
#include "autoproduction/preprocessing/chopper.h"

#include <stddef.h>

#include "gtest/gtest.h"

namespace Autoproduction {
namespace Preprocessing {
namespace {
constexpr std::size_t ORIG_IMAGE_HEIGHT = 800;
constexpr std::size_t ORIG_IMAGE_WIDTH = 3840;

}  // namespace

class GridTest : public ::testing::Test {};

TEST_F(GridTest, IdentityTargetHeight) {
  Grid<1, 1> g(static_cast<float>(ORIG_IMAGE_HEIGHT), static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(static_cast<float>(ORIG_IMAGE_HEIGHT), g.target_height_);
}

TEST_F(GridTest, IdentityTargetWidth) {
  Grid<1, 1> g(static_cast<float>(ORIG_IMAGE_HEIGHT), static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(static_cast<float>(ORIG_IMAGE_WIDTH), g.target_width_);
}

TEST_F(GridTest, Grid1x3) {
  Grid<1, 3> g(static_cast<float>(ORIG_IMAGE_HEIGHT), static_cast<float>(ORIG_IMAGE_WIDTH));

  EXPECT_EQ(1408.f, g.target_width_);

  EXPECT_EQ(3, g.offsets_width_.size());

  EXPECT_EQ(0.f, g.offsets_height_[0]);
  EXPECT_EQ(0.f, g.offsets_width_[0]);

  EXPECT_EQ(0.f, g.offsets_height_[1]);
  EXPECT_EQ(1280.f, g.offsets_width_[1]);

  EXPECT_EQ(0.f, g.offsets_height_[2]);
  EXPECT_EQ(2432.f, g.offsets_width_[2]);
}

TEST_F(GridTest, Grid2x3) {
  Grid<2, 3> g(static_cast<float>(ORIG_IMAGE_HEIGHT), static_cast<float>(ORIG_IMAGE_WIDTH));

  EXPECT_EQ(1408.f, g.target_width_);

  EXPECT_EQ(6, g.offsets_width_.size());

  EXPECT_EQ(0.f, g.offsets_height_[0]);
  EXPECT_EQ(0.f, g.offsets_width_[0]);

  EXPECT_EQ(0.f, g.offsets_height_[1]);
  EXPECT_EQ(1280.f, g.offsets_width_[1]);

  EXPECT_EQ(0.f, g.offsets_height_[2]);
  EXPECT_EQ(2432.f, g.offsets_width_[2]);

  EXPECT_EQ(360.f, g.offsets_height_[3]);
  EXPECT_EQ(0.f, g.offsets_width_[3]);

  EXPECT_EQ(360.f, g.offsets_height_[4]);
  EXPECT_EQ(1280.f, g.offsets_width_[4]);

  EXPECT_EQ(360.f, g.offsets_height_[5]);
  EXPECT_EQ(2432.f, g.offsets_width_[5]);

}
}  // namespace Inference
}  // namespace Autoproduction
