

#include "autoproduction/preprocessing/image_grid.h"

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
  Grid<1, 1> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(ORIG_IMAGE_HEIGHT, g.cells_[0].height);
}

TEST_F(GridTest, IdentityTargetWidth) {
  Grid<1, 1> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(static_cast<float>(ORIG_IMAGE_WIDTH), g.cells_[0].width);
}

TEST_F(GridTest, Grid1x3) {
  Grid<1, 3> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));

  EXPECT_EQ(1408.f, g.cells_[0].width);

  EXPECT_EQ(3, g.cells_.size());

  EXPECT_EQ(0.f, g.cells_[0].y);
  EXPECT_EQ(0.f, g.cells_[0].x);

  EXPECT_EQ(0.f, g.cells_[1].y);
  EXPECT_EQ(1280.f, g.cells_[1].x);

  EXPECT_EQ(0.f, g.cells_[2].y);
  EXPECT_EQ(2432.f, g.cells_[2].x);
}

TEST_F(GridTest, Grid2x3) {
  Grid<2, 3> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));

  EXPECT_EQ(1408.f, g.cells_[0].width);

  EXPECT_EQ(6, g.cells_.size());

  EXPECT_EQ(0.f, g.cells_[0].y);
  EXPECT_EQ(0.f, g.cells_[0].x);

  EXPECT_EQ(0.f, g.cells_[1].y);
  EXPECT_EQ(1280.f, g.cells_[1].x);

  EXPECT_EQ(0.f, g.cells_[2].y);
  EXPECT_EQ(2432.f, g.cells_[2].x);

  EXPECT_EQ(360.f, g.cells_[3].y);
  EXPECT_EQ(0.f, g.cells_[3].x);

  EXPECT_EQ(360.f, g.cells_[4].y);
  EXPECT_EQ(1280.f, g.cells_[4].x);

  EXPECT_EQ(360.f, g.cells_[5].y);
  EXPECT_EQ(2432.f, g.cells_[5].x);
}

TEST_F(GridTest, Grid1x5) {
  Grid<1, 5> g(static_cast<float>(ORIG_IMAGE_HEIGHT),
               static_cast<float>(ORIG_IMAGE_WIDTH));
  EXPECT_EQ(845, g.target_width_);
}

}  // namespace Preprocessing
}  // namespace Autoproduction
