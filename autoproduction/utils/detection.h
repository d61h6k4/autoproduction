namespace Autoproduction {
namespace Util {
struct NormalizedBBox {
  float ymin_;
  float xmin_;
  float ymax_;
  float xmax_;
};

using Score = float;
enum class ClassName { unknown_, person_, sport_ball_ };

struct Detection {
  NormalizedBBox bbox_;
  Score score_;
  ClassName class_name_;

  Detection(float ymin, float xmin, float ymax, float xmax, float score,
            float class_id) {
    bbox_ = NormalizedBBox{
        .ymin_ = ymin, .xmin_ = xmin, .ymax_ = ymax, .xmax_ = xmax};
    score_ = score;
    switch (static_cast<int>(class_id)) {
      case 1:
        class_name_ = ClassName::person_;
        break;
      case 2:
        class_name_ = ClassName::sport_ball_;
        break;
      default:
        class_name_ = ClassName::unknown_;
    }
  }

  Detection(float ymin, float xmin, float ymax, float xmax, Score score,
            ClassName class_name) {
    bbox_ = NormalizedBBox{
        .ymin_ = ymin, .xmin_ = xmin, .ymax_ = ymax, .xmax_ = xmax};
    score_ = score;
    class_name_ = class_name;
  }
};
}  // namespace Util
}  // namespace Autoproduction
