// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/model_data.h"

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

  InitAllFeatures();

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

void ModelData::InitFromInstances(const std::vector<Instance>& instances) {
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

  InitAllFeatures();
  InitLambdas();
}

void ModelData::FormatInstance(const Instance& instance,
                               MemInstance* mem_instance) const {
  assert(mem_instance != NULL);
  mem_instance->Clear();

  mem_instance->set_label_id(label_vocab_.Id(instance.label()));
  for (Instance::ConstIterator citer(instance);
       !citer.Done(); citer.Next()) {
    int32_t feature_name_id = featurename_vocab_.Id(citer.FeatureName());
    if (feature_name_id > 0) {
      mem_instance->AddFeature(feature_name_id, citer.FeatureValue());
    }
  }
}

int32_t ModelData::CalcConditionalProbability(
    const MemInstance& mem_instance, std::vector<double>* prob_dist) const {
  std::vector<double> powv(NumClasses(), 0.0);

  for (MemInstance::ConstIterator citer(mem_instance);
       !citer.Done(); citer.Next()) {
    const std::vector<int32_t>& feature_ids = FeatureIds(citer.FeatureNameId());
    for (size_t i = 0; i < feature_ids.size(); ++i) {
      const int32_t feature_id = feature_ids[i];
      powv[FeatureAt(feature_id).LabelId()]
          += lambdas_[feature_id] * citer.FeatureValue();
    }
  }

  std::vector<double>::const_iterator pmax
      = max_element(powv.begin(), powv.end());
  double sum = 0.0;
  double offset = std::max(0.0, *pmax - 700);  // to avoid overflow
  for (int32_t label_id = 0; label_id < NumClasses(); ++label_id) {
    double pow_value = powv[label_id] - offset;
    double prod = exp(pow_value);  // exp(w * x)
    assert(prod != 0);

    (*prob_dist)[label_id] = prod;
    sum += prod;
  }

  int32_t max_label = 0;
  if (sum > 0.0) {
    for (int32_t label_id = 0; label_id < NumClasses(); ++label_id) {
      (*prob_dist)[label_id] /= sum;
      if ((*prob_dist)[label_id] > (*prob_dist)[max_label]) {
        max_label = label_id;
      }
    }
  }
  assert(max_label >= 0);

  return max_label;
}

}  // namespace common
}  // namespace mltk

