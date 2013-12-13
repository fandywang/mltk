// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"
#include "mltk/maxent/optimizer.h"

namespace mltk {
namespace maxent {

using mltk::common::Instance;
using mltk::common::MemInstance;

bool MaxEnt::LoadModel(const std::string& filename) {
  model_data_.Clear();
  return model_data_.Load(filename);
}

bool MaxEnt::SaveModel(const std::string& filename) const {
  return model_data_.Save(filename);
}

bool MaxEnt::Train(const std::vector<Instance>& instances,
                   int32_t num_heldout,
                   int32_t feature_cutoff) {
  // parameter estimation
  std::cerr << "parameter estimation ..." << std::endl;
  model_data_.Clear();
  assert(optimizer_ != NULL);

  optimizer_->EstimateParamater(instances,
                                num_heldout,
                                feature_cutoff,
                                &model_data_);

  // count the number of active features
  std::cerr << "number of active features = " << model_data_.NumActiveFeatures()
      << std::endl;
  std::cerr << "parameter estimation done" << std::endl;

  return true;
}

std::vector<double> MaxEnt::Predict(Instance* instance) const {
  MemInstance mem_instance;
  model_data_.FormatInstance(*instance, &mem_instance);

  std::vector<double> prob_dist(model_data_.NumClasses());
  int32_t label_id = model_data_.CalcConditionalProbability(mem_instance,
                                                            &prob_dist);
  instance->set_label(model_data_.Label(label_id));

  return prob_dist;
}

}  // namespace maxent
}  // namespace mltk

