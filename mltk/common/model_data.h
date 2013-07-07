// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Implementations of model data.

#ifndef MLTK_COMMON_MODEL_DATA_H_
#define MLTK_COMMON_MODEL_DATA_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "mltk/common/feature_vocabulary.h"
#include "mltk/common/feature.h"
#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"
#include "mltk/common/vocabulary.h"

namespace mltk {
namespace common {

class ModelData {
 public:
  ModelData() {}
  ~ModelData() {}

  // Load model data from filename.
  bool Load(const std::string& filename);

  // Save model data to filename.
  bool Save(const std::string& filename) const;

  // Initialize with instances.
  void InitFromInstances(const std::vector<Instance>& instances);

  void Clear() {
    label_vocab_.Clear();
    featurename_vocab_.Clear();
    feature_vocab_.Clear();
    lambdas_.clear();
    all_features_.clear();
  }

  int32_t NumClasses() const { return label_vocab_.Size(); }
  int32_t NumFeatures() const { return feature_vocab_.Size(); }

  // Transfer from class Instance to class MemInstance.
  void FormatInstance(const Instance& instance,
                      MemInstance* mem_instances) const;

  int32_t FeatureNameId(const std::string& feature_name) const {
    return featurename_vocab_.Id(feature_name);
  }

  int32_t LabelId(const std::string& label) const {
    return label_vocab_.Id(label);
  }
  const std::string& Label(int32_t label_id) const {
    return label_vocab_.Str(label_id);
  }

  const Feature& FeatureAt(int32_t feature_id) const {
    return feature_vocab_.GetFeature(feature_id);
  }
  int32_t FeatureId(const Feature& feature) const {
    return feature_vocab_.FeatureId(feature);
  }

  const std::vector<int32_t>& FeatureIds(int32_t feature_name_id) const {
    assert(feature_name_id >= 0 && feature_name_id < featurename_vocab_.Size());
    return all_features_[feature_name_id];
  }

  const std::vector<double>& Lambdas() const { return lambdas_; }
  std::vector<double>* MutableLambdas() { return &lambdas_; }

  void UpdateLambdas(const std::vector<double>& lambdas) {
    assert(lambdas_.size() == lambdas.size());

    for (size_t i = 0; i < lambdas.size(); ++i) {
      lambdas_[i] = lambdas[i];
    }
  }

  double L1NormLambdas() const {
    double sum = 0.0;
    for (size_t i = 0; i < lambdas_.size(); ++i) { sum += abs(lambdas_[i]); }
    return sum;
  }

  int32_t NumActiveFeatures() const {
    int32_t num_active = 0;
    for (size_t i = 0; i < lambdas_.size(); ++i) {
      if (lambdas_[i] != 0) { ++num_active; }
    }
    return num_active;
  }

  int32_t CalcConditionalProbability(const MemInstance& mem_instance,
                                     std::vector<double>* prob_dist) const;

 private:
  void InitAllFeatures() {
    for (int32_t feature_name_id = 0;
         feature_name_id < featurename_vocab_.Size();
         ++feature_name_id) {
      all_features_.push_back(std::vector<int32_t>());
      std::vector<int32_t>& vi = all_features_.back();

      for (int32_t label_id = 0; label_id < label_vocab_.Size(); ++label_id) {
        int32_t feature_id = feature_vocab_.FeatureId(
            Feature(label_id, feature_name_id));
        if (feature_id >= 0) { vi.push_back(feature_id); }
      }
    }
  }

  void InitLambdas() {
    lambdas_.resize(feature_vocab_.Size());
    for (int32_t i = 0; i < feature_vocab_.Size(); ++i) { lambdas_[i] = 0.0; }
  }

  Vocabulary label_vocab_;  // label mapping, {y : id}

  Vocabulary featurename_vocab_;  // vocabulary of feature names, {x : id}

  FeatureVocabulary feature_vocab_;  // vocabulary of features, {f(x, y) : id}

  std::vector<double> lambdas_;  // vector of lambda, weight for feature f(x, y)
                                 // lambdas_.size() == feature_vocab_.size()

  // all possible features f(x, y), format:
  // [featurename_id, [feature1.id, feature2.id, ...]]
  std::vector<std::vector<int32_t> > all_features_;
};

}  // namespace common
}  // namespace mltk

#endif  // MLTK_COMMON_MODEL_DATA_H_

