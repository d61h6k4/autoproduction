#include "autoproduction/postprocessing/chopped_image_detections_joiner.h"
#include "gtest/gtest.h"

namespace Autoproduction {
namespace Postprocessing {

class ChoppedImageDetectionsJoinerTest : public ::testing::Test {};

TEST_F(ChoppedImageDetectionsJoinerTest, OneDetectectionIsIdentity) {
  ChoppedImageDetectionsJoiner<3, 3> cidjoiner(720, 1280);
  std::vector<std::vector<Autoproduction::Util::Detection>> raw_detections;
  std::vector<Autoproduction::Util::Detection> raw_detection;
  raw_detection.emplace_back(
      Autoproduction::Util::Detection(0.5, 0.5, 0.6, 0.6, 1., 1.));
  raw_detections.push_back(raw_detection);

  auto res = cidjoiner(raw_detections);
  EXPECT_EQ(1, res.size());
}

TEST_F(ChoppedImageDetectionsJoinerTest, InCenterOnSubimageNotInCenterOnImage) {
  ChoppedImageDetectionsJoiner<1, 2> cidjoiner(720, 1280);
  std::vector<std::vector<Autoproduction::Util::Detection>> raw_detections;
  std::vector<Autoproduction::Util::Detection> left_subimage_detections;
  left_subimage_detections.emplace_back(
      Autoproduction::Util::Detection(0.45, 0.45, 0.55, 0.55, 1., 1.));
  std::vector<Autoproduction::Util::Detection> right_subimage_detections;
  right_subimage_detections.emplace_back(
      Autoproduction::Util::Detection(0.45, 0.45, 0.55, 0.55, 1., 1.));

  raw_detections.push_back(left_subimage_detections);
  raw_detections.push_back(right_subimage_detections);

  auto res = cidjoiner(raw_detections);
  EXPECT_EQ(2, res.size());

  EXPECT_FLOAT_EQ(left_subimage_detections[0].bbox_.ymin_, res[0].bbox_.ymin_);
  EXPECT_GT(left_subimage_detections[0].bbox_.xmin_, res[0].bbox_.xmin_);
  EXPECT_FLOAT_EQ(left_subimage_detections[0].bbox_.ymax_, res[0].bbox_.ymax_);
  EXPECT_GT(left_subimage_detections[0].bbox_.xmax_, res[0].bbox_.xmax_);

  EXPECT_FLOAT_EQ(right_subimage_detections[0].bbox_.ymin_, res[1].bbox_.ymin_);
  EXPECT_LT(right_subimage_detections[0].bbox_.xmin_, res[1].bbox_.xmin_);
  EXPECT_FLOAT_EQ(right_subimage_detections[0].bbox_.ymax_, res[1].bbox_.ymax_);
  EXPECT_LT(right_subimage_detections[0].bbox_.xmax_, res[1].bbox_.xmax_);
}
}  // namespace Postprocessing
}  // namespace Autoproduction
