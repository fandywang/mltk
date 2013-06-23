// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Memory Insance class.

#ifndef MLTK_COMMON_MEM_INSTANCE_H_
#define MLTK_COMMON_MEM_INSTANCE_H_

#include <assert.h>
#include <utility>
#include <vector>

namespace mltk {
namespace common {

// MemInstance is the inside data format of instance for training/testing,
// which depends on Instance.
class MemInstance {
 public:
  MemInstance() {}
  explicit MemInstance(int32_t label_id) : label_id_(label_id) {}
  ~MemInstance() {}

  void Clear() {
    label_id_ = -1;
    features_.clear();
  }

  void set_label_id(int32_t label_id) {
    assert(label_id >= 0);
    label_id_ = label_id;
  }
  int32_t label_id() const { return label_id_; }

  void AddFeature(int32_t feature_name_id, double value) {
    assert(feature_name_id >= 0);
    features_.push_back(std::pair<int32_t, double>(feature_name_id, value));
  }

  // A const interator over all features in an instance.
  class ConstIterator {
   public:
    explicit ConstIterator(const MemInstance& mem_instance)
      : feature_idx_(0), mem_instance_(mem_instance) {}
    ~ConstIterator() {}

    // Returns true if we are doing iterater.
    bool Done() const { return feature_idx_ >= mem_instance_.features_.size(); }

    void Next() {
      assert(!Done());
      ++feature_idx_;
    }

    int32_t FeatureNameId() const {
      assert(!Done());
      return mem_instance_.features_[feature_idx_].first;
    }

    double FeatureValue() const {
      assert(!Done());
      return mem_instance_.features_[feature_idx_].second;
    }

    const std::pair<int32_t, double>& Feature() const {
      assert(!Done());
      return mem_instance_.features_[feature_idx_];
    }

    int32_t LabelId() const {
      assert(!Done());
      return mem_instance_.label_id_;
    }

   private:
    size_t feature_idx_;
    const MemInstance& mem_instance_;
  };

 private:
  int32_t label_id_;  // class id
  std::vector<std::pair<int32_t, double> > features_;  // vector of features
};

}  // namespace common
}  // namespace mltk

#endif  // MLTK_COMMON_MEM_INSTANCE_H_

