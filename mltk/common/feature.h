// Copyright (c) 2013 MLTK project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Feature class.

#ifndef MLTK_COMMON_FEATURE_H_
#define MLTK_COMMON_FEATURE_H_

#include <assert.h>
#include <stdint.h>

namespace mltk {
namespace common {

// feature: f(x, y)
class Feature {
 public:
  enum { MAX_LABEL_TYPES = 255 };

  Feature(const int32_t label, const int32_t feature)
      : body_((feature << 8) + label) {
    assert(label >= 0 && label <= MAX_LABEL_TYPES);
    assert(feature >= 0 && feature <= 0xffffff);
  }
  ~Feature() {}

  int32_t LabelId() const { return static_cast<int32_t>(body_ & 0xff); }

  int32_t FeatureId() const { return static_cast<int32_t>(body_ >> 8); }

  uint32_t Body() const { return body_; }

 private:
  uint32_t body_;  // feature = x, lable = y
};

}  // namespace common
}  // namespace mltk

#endif // MLTK_COMMON_FEATURE_H_

