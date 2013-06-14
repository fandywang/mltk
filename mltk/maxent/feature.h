// Copyright (c) 2013 MLTK project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Feature class.

#ifndef MLTK_MAXENT_FEATURE_H_
#define MLTK_MAXENT_FEATURE_H_

#include <assert.h>
#include <map>
#include <vector>

namespace mltk {
namespace maxent {

typedef std::map<uint32_t/* Feature.body */, int32_t/* id */> FeatureMapType;

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

class FeatureVocabulary {
 public:
  FeatureVocabulary() {}
  ~FeatureVocabulary() {}

  int32_t Put(const Feature& f) {
    FeatureMapType::const_iterator citer = feature2id_.find(f.Body());
    if (citer == feature2id_.end()) {
      int32_t id = static_cast<int32_t>(id2feature_.size());
      id2feature_.push_back(f);
      feature2id_[f.Body()] = id;

      return id;
    }
    return citer->second;
  }

  int32_t FeatureId(const Feature& f) const {
    FeatureMapType::const_iterator citer = feature2id_.find(f.Body());
    if (citer == feature2id_.end()) {
      return -1;
    }
    return citer->second;
  }

  const Feature& GetFeature(int32_t id) const {
    assert(id >= 0 && id < static_cast<int32_t>(id2feature_.size()));
    return id2feature_[id];
  }

  int32_t Size() const { return id2feature_.size(); }

  void Clear() {
    feature2id_.clear();
    id2feature_.clear();
  }

 private:
  FeatureMapType feature2id_;  // key = Feature.body(), value = Feature's id
  std::vector<Feature> id2feature_;
};

}  // namespace maxent
}  // namespace mltk

#endif // MLTK_MAXENT_FEATURE_H_

