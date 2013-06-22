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

  void Clear() {
    label_vocab_.Clear();
    featurename_vocab_.Clear();
    feature_vocab_.Clear();
    lambdas_.clear();
    all_features_.clear();
  }

  // Initialize with instances.
  void InitFromInstances(const std::vector<Instance>& instances);

  // Transfer from class Instance to class MemInstance.
  void FormatInstances(const std::vector<Instance>& instances,
                       std::vector<MemInstance>* mem_instances) const;

  int32_t NumClasses() const { return label_vocab_.Size(); }
  int32_t NumFeatures() const { return feature_vocab_.Size(); }

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
    lambdas_.resize(lambdas.size());
    for (size_t i = 0; i < lambdas.size(); ++i) {
      lambdas_[i] = lambdas[i];
    }
  }

  double L1NormLambdas() {
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

 private:
  Vocabulary label_vocab_;  // label mapping, {y : id}

  Vocabulary featurename_vocab_;  // feature name mapping, {x : id}

  FeatureVocabulary feature_vocab_;  // vocabulary of features, f(x, y)

  std::vector<double> lambdas_;  // vector of lambda, weight for feature f(x, y)
                                 // lambdas_.size() == feature_vocab_.size()

  // all possible features f(x, y), format:
  // [featurename_id, [feature1.id, feature2.id, ...]]
  std::vector<std::vector<int32_t> > all_features_;
};

bool ModelData::Load(const std::string& filename) {
  Clear();

  FILE* fp = fopen(filename.c_str(), "r");
  if (!fp) {
    std::cerr << "error: cannot open " << filename << "!" << std::endl;
    return false;
  }

  char buf[1024];
  while(fgets(buf, 1024, fp)) {
    std::string line(buf);
    std::string::size_type t1 = line.find_first_of('\t');
    std::string::size_type t2 = line.find_last_of('\t');
    std::string label_name = line.substr(0, t1);
    std::string feature_name = line.substr(t1 + 1, t2 - (t1 + 1));
    double lambda;
    std::string w = line.substr(t2 + 1);
    sscanf(w.c_str(), "%lf", &lambda);

    int32_t label_id = label_vocab_.Put(label_name);
    int32_t feature_name_id = featurename_vocab_.Put(feature_name);
    feature_vocab_.Put(Feature(label_id, feature_name_id));
    lambdas_.push_back(lambda);
  }
  fclose(fp);

  // TODO(fandywang): to be optimized
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

  return true;
}

bool ModelData::Save(const std::string& filename) const {
  FILE* fp = fopen(filename.c_str(), "w");
  if (!fp) {
    std::cerr << "error: cannot open " << filename << "!" << std::endl;
    return false;
  }

  for (StringMapType::const_iterator iter = featurename_vocab_.begin();
       iter != featurename_vocab_.end();
       ++iter) {
    for (int32_t label_id = 0; label_id < label_vocab_.Size(); ++label_id) {
      std::string label = label_vocab_.Str(label_id);
      int32_t id = feature_vocab_.FeatureId(Feature(label_id, iter->second));
      if (id < 0) continue;
      if (lambdas_[id] == 0) continue;  // ignore zero-weight features

      fprintf(fp, "%s\t%s\t%f\n",
              label.c_str(), iter->first.c_str(), lambdas_[id]);
    }
  }
  fclose(fp);

  return true;
}

void ModelData::Initialize(const std::vector<Instance>& instances) {
  Clear();

  for (size_t n = 0; n < instances.size(); ++n) {
    int32_t label_id = label_vocab_.Put(instances[n].label());
    if (label_id > Feature::MAX_LABEL_TYPES) {
      std::cerr << "error: too many types of labels." << std::endl;
      exit(1);
    }

    for (Instance::ConstIterator citer(instances[n]);
         !citer.Done(); citer.Next()) {
      int32_t feature_name_id = featurename_vocab_.Put(citer.FeatureName());
      feature_vocab_.Put(Feature(label_id, feature_name_id));
    }
  }

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

  lambdas_.resize(feature_vocab_.Size());
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    lambdas_[i] = 0.0;
  }
}

void ModelData::FormatInstances(const std::vector<Instance>& instances,
                                std::vector<MemInstance>* mem_instances) const {
  assert(mem_instances != NULL);
  mem_instances->clear();

  for (size_t n = 0; n < instances.size(); ++n) {
    const Instance& instance = instances[n];
    mem_instances->push_back(MemInstance());
    MemInstance& mem_instance = mem_instances->back();

    mem_instance.set_label(label_vocab_.Id(instance.label()));
    for (Instance::ConstIterator citer(instance);
         !citer.Done(); citer.Next()) {
      int32_t feature_name_id = featurename_vocab_.Id(citer.FeatureName());
      if (feature_name_id > 0) {
        mem_instance.AddFeature(feature_name_id, citer.FeatureValue());
      }
    }
  }
}

}  // namespace common
}  // namespace mltk

#endif  // MLTK_COMMON_MODEL_DATA_H_

