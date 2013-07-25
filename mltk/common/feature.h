// Copyright (c) 2013 MLTK project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Copy and Modify from 'http://www.nactem.ac.uk/tsuruoka/maxent/'.
// maxent.h
//
// The Feature class.

#ifndef MLTK_COMMON_FEATURE_H_
#define MLTK_COMMON_FEATURE_H_

#include <assert.h>
#include <stdint.h>

namespace mltk {
namespace common {

// feature: f(x, y), x: feature_name_id, y: label_id
class Feature {
 public:
  enum { MAX_LABEL_TYPES = 255 };

  Feature(const int32_t label_id, const int32_t feature_name_id)
      : body_((feature_name_id << 8) + label_id) {
    assert(label_id >= 0 && label_id <= MAX_LABEL_TYPES);
    assert(feature_name_id >= 0 && feature_name_id <= 0xffffff);
  }
  ~Feature() {}

  int32_t LabelId() const { return static_cast<int32_t>(body_ & 0xff); }

  int32_t FeatureNameId() const { return static_cast<int32_t>(body_ >> 8); }

  uint32_t Body() const { return body_; }

 private:
  uint32_t body_;  // feature = x, lable = y
};

}  // namespace common
}  // namespace mltk

#endif // MLTK_COMMON_FEATURE_H_

