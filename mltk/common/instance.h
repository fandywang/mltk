// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Instance class.

#ifndef MLTK_COMMON_INSTANCE_H_
#define MLTK_COMMON_INSTANCE_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "common/base/string/algorithm.h"

namespace mltk {
namespace common {

// Instance is the user-oriented data format of instance for training/testing.
class Instance {
 public:
  Instance() {}
  explicit Instance(const std::string& label) : label_(label) {}
  ~Instance() {}

  // text format: class \t f1:v1 \t f2:v2 \t ...
  bool ParseFromText(const std::string& text) {
    features_.clear();

    std::vector<std::string> fields;
    ::common::SplitString(text, "\t", &fields);

    if (fields.size() < 2) {
      std::cerr << "Text format error. text: " << text << std::endl;
      return false;
    }

    label_ = fields[0];
    for (size_t i = 1; i < fields.size(); ++i) {
      std::vector<std::string> feature_info;
      ::common::SplitString(fields[i], ":", &feature_info);
      const std::string& feature_name = feature_info[0];
      double value = atof(feature_info[1].c_str());
      features_.push_back(std::pair<std::string, double>(feature_name, value));
    }

    return true;
  }

  void set_label(const std::string& label) { label_ = label; }
  std::string label() const { return label_; }

  // to add a real-valued feature
  void AddFeature(const std::string& feature_name, const double value) {
    features_.push_back(std::pair<std::string, double>(feature_name, value));
  }

  // A const interator over all features in an instance.
  class ConstIterator {
   public:
    explicit ConstIterator(const Instance& instance)
      : feature_idx_(0), instance_(instance) {}
    ~ConstIterator() {}

    // Returns true if we are doing iterater.
    bool Done() const { return feature_idx_ >= instance_.features_.size(); }

    void Next() {
      assert(!Done());
      ++feature_idx_;
    }

    const std::string& FeatureName() const {
      assert(!Done());
      return instance_.features_[feature_idx_].first;
    }

    double FeatureValue() const {
      assert(!Done());
      return instance_.features_[feature_idx_].second;
    }

    const std::pair<std::string, double>& Feature() const {
      assert(!Done());
      return instance_.features_[feature_idx_];
    }

    const std::string& Label() const {
      assert(!Done());
      return instance_.label_;
    }

   private:
    size_t feature_idx_;
    const Instance& instance_;
  };

 public:
  std::string label_;  // the label/class of the instance
  std::vector<std::pair<std::string, double> > features_;  // the real-valued
                                                           // features
};

}  // namespace common
}  // namespace mltk

#endif  // MLTK_COMMON_INSTANCE_H_

