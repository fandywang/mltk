// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Instance class.

#ifndef MLTK_COMMON_INSTANCE_H_
#define MLTK_COMMON_INSTANCE_H_

#include <string>
#include <utility>
#include <vector>

namespace mltk {
namespace common {

// data format of instance for training/testing
class Instance {
 public:
  Instance() {}
  explicit Instance(const std::string& label) : label_(label) {}
  ~Instance() {}

  void set_label(const std::string& label) { label_ = label; }
  std::string label() const { return label_; }

  // to add a real-valued feature
  void AddFeature(const std::string& feature_name, const double value) {
    features_.push_back(std::pair<std::string, double>(feature_name, value));
  }
  const std::vector<std::pair<std::string, double> >& GetFeatures() const {
    return features_;
  }

 public:
  std::string label_;  // the label/class of the instance
  std::vector<std::pair<std::string, double> > features_;  // the real-valued
                                                           // features
};

}  // namespace common
}  // namespace mltk

#endif  // MLTK_COMMON_INSTANCE_H_

