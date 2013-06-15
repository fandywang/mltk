// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Feature Vocabulary.

#ifndef MLTK_COMMON_FEATURE_VOCABULARY_H_
#define MLTK_COMMON_FEATURE_VOCABULARY_H_

#include <assert.h>
#include <stdint.h>
#include <map>
#include <vector>

#include "mltk/common/feature.h"

namespace mltk {
namespace common {

typedef std::map<uint32_t/* Feature.body */, int32_t/* id */> FeatureMapType;

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

}  // namespace common
}  // namespace mltk

#endif  // MLTK_COMMON_FEATURE_VOCABULARY_H_

