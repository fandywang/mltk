// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "mltk/common/double_vector.h"
#include "mltk/common/feature.h"
#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"

namespace mltk {
namespace maxent {

using mltk::common::Feature;
using mltk::common::Instance;
using mltk::common::MemInstance;

bool MaxEnt::LoadModel(const std::string& filename) {
  Clear();
  return model_data_.Load(filename);
}

bool MaxEnt::SaveModel(const std::string& filename) const {
  return model_data_.Save(filename);
}

bool MaxEnt::Train(const std::vector<Instance>& instances,
                   int32_t num_heldout) {
  std::cerr << "preparing for estimation..." << std::endl;

  // initialize model
  std::cerr << "initialize model data...";
  model_data_.InitFromInstances(instances);

  for (size_t n = 0; n < instances.size(); ++n) {
    mem_instances_.push_back(MemInstance());
    MemInstance& mem_instance = mem_instances_.back();
    model_data_.FormatInstance(instances[n], &mem_instance);
  }
  if (mem_instances_.size() == 0) {
    std::cerr << "error: no training data." << std::endl;
    return false;
  }
  std::cerr << "done" << std::endl;

  // preparing for heldout data
  if (num_heldout >= static_cast<int32_t>(mem_instances_.size())) {
    std::cerr << "error: too much heldout data. no training data is available."
        << std::endl;
    return false;
  }
  for (int32_t i = 0; i < num_heldout; ++i) {
    heldout_.push_back(mem_instances_.back());
    mem_instances_.pop_back();
  }

  std::cerr << "number of classes = " << model_data_.NumClasses() << std::endl;
  std::cerr << "number of features = " << model_data_.NumFeatures()
      << std::endl;
  std::cerr << "number of training instances = " << mem_instances_.size()
      << std::endl;
  std::cerr << "number of heldout instances = " << heldout_.size()
      << std::endl;

  // normalize l1 & l2 regularizer
  if (l1reg_ > 0) {
    l1reg_ /= mem_instances_.size();
    std::cerr << "L1 regularizer = " << l1reg_ << std::endl;
  }
  if (l2reg_ > 0) {
    l2reg_ /= mem_instances_.size();
    std::cerr << "L2 regularizer = " << l2reg_ << std::endl;
  }
  if (l1reg_ > 0 && l2reg_ > 0) {
    std::cerr << "error: L1 and L2 regularizers cannot be used simultaneously."
         << std::endl;
    return false;
  }

  // calc E_p1 (f), p1(x, y) = count(x, y) / N
  std::cerr << "calculating empirical expectation...";
  empirical_expectation_.resize(model_data_.NumFeatures());
  for (int32_t i = 0; i < model_data_.NumFeatures(); ++i) {
    empirical_expectation_[i] = 0;
  }
  for (size_t n = 0; n < mem_instances_.size(); ++n) {
    for (MemInstance::ConstIterator citer(mem_instances_[n]);
         !citer.Done(); citer.Next()) {
      const std::vector<int32_t> feature_ids
          = model_data_.FeatureIds(citer.FeatureNameId());
      for (size_t i = 0; i < feature_ids.size(); ++i) {
        if (model_data_.FeatureAt(feature_ids[i]).LabelId()
            == citer.LabelId()) {
          empirical_expectation_[feature_ids[i]] += citer.FeatureValue();
          break;
        }
      }
    }
  }
  for (int32_t i = 0; i < model_data_.NumFeatures(); ++i) {
    empirical_expectation_[i] /= mem_instances_.size();
  }
  std::cerr << "done" << std::endl;

  // parameter estimation
  std::cerr << "parameter estimation ..." << std::endl;
  if (optimization_method_ == SGD) {
    PerformSGD();
  } else {
    PerformQuasiNewton();
  }
  std::cerr << "done" << std::endl;

  // count the number of active features
  std::cerr << "number of active features = " << model_data_.NumActiveFeatures()
      << std::endl;

  return true;
}

std::vector<double> MaxEnt::Classify(Instance* instance) const {
  MemInstance mem_instance;
  // model_data_.FormatInstance(instance, &mem_instance);

  for (Instance::ConstIterator citer(*instance); !citer.Done(); citer.Next()) {
    int32_t feature_name_id = model_data_.FeatureNameId(citer.FeatureName());
    if (feature_name_id >= 0) {
      // only using the feature exists in training data
      mem_instance.AddFeature(feature_name_id, citer.FeatureValue());
    }
  }

  std::vector<double> prob_dist(model_data_.NumClasses());
  int32_t label_id = Classify(mem_instance, &prob_dist);
  instance->set_label(model_data_.Label(label_id));

  return prob_dist;
}

void MaxEnt::Clear() {
  model_data_.Clear();
  mem_instances_.clear();
  heldout_.clear();
  empirical_expectation_.clear();
  model_expectation_.clear();
}

void MaxEnt::PerformQuasiNewton() {
  const std::vector<double> lambdas = model_data_.Lambdas();
  assert(static_cast<int32_t>(lambdas.size()) == model_data_.NumFeatures());

  std::vector<double> x0(model_data_.NumFeatures());
  for (int32_t i = 0; i < model_data_.NumFeatures(); ++i) {
    x0[i] = lambdas[i];
  }

  std::vector<double> x;
  if (l1reg_ > 0 || optimization_method_ == OWLQN) {
    // NOTE(l1reg_ > 0): The LBFGS limited-memory quasi-Newton method is the
    // algorithm of choice for optimizing the parameters of large-scale
    // log-linear models with L2-regularization, but it cannot be used for an
    // L1-regularized loss due to its non-diﬀerentiability whenever some
    // parameter is zero.
    std::cerr << "performing OWLQN" << std::endl;
    x = PerformOWLQN(x0, l1reg_);
  } else {
    std::cerr << "performing LBFGS" << std::endl;
    x = PerformLBFGS(x0);
  }

  model_data_.UpdateLambdas(x);
}

double MaxEnt::FunctionGradient(const std::vector<double>& x,
                                std::vector<double>* grad) {
  assert(static_cast<size_t>(model_data_.NumFeatures()) == x.size());

  model_data_.UpdateLambdas(x);

  double score = UpdateModelExpectation();

  // update gradient
  if (l2reg_ == 0) {
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = model_expectation_[i] - empirical_expectation_[i];
    }
  } else {
    const double c = l2reg_ * 2;
    const std::vector<double>& lambdas = model_data_.Lambdas();
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = model_expectation_[i] - empirical_expectation_[i]
                   + c * lambdas[i];
    }
  }

  return -score;
}

double MaxEnt::UpdateModelExpectation() {
  double logl = 0;
  int32_t ncorrect = 0;

  model_expectation_.resize(model_data_.NumFeatures());
  for (int i = 0; i < model_data_.NumFeatures(); ++i) {
    model_expectation_[i] = 0;
  }

  for (size_t n = 0; n < mem_instances_.size(); ++n) {
    std::vector<double> prob_dist(model_data_.NumClasses());
    int32_t max_label = CalcConditionalProbability(mem_instances_[n],
                                                   &prob_dist);

    logl += log(prob_dist[mem_instances_[n].label_id()]);
    if (max_label == mem_instances_[n].label_id()) { ++ncorrect; }

    // model_expectation
    for (MemInstance::ConstIterator citer(mem_instances_[n]);
         !citer.Done(); citer.Next()) {
      const std::vector<int32_t>& feature_ids
          = model_data_.FeatureIds(citer.FeatureNameId());
      for (size_t i = 0; i < feature_ids.size(); ++i) {
        const int32_t feature_id = feature_ids[i];
        model_expectation_[feature_id]
          += prob_dist[model_data_.FeatureAt(feature_id).LabelId()]
             * citer.FeatureValue();
      }
    }
  }

  const std::vector<double>& lambdas = model_data_.Lambdas();
  for (int32_t i = 0; i < model_data_.NumFeatures(); ++i) {
    model_expectation_[i] /= mem_instances_.size();
    if (l2reg_ > 0) { logl -= lambdas[i] * lambdas[i] * l2reg_; }
  }

  train_accuracy_ = static_cast<double>(ncorrect) / mem_instances_.size();
  logl /= mem_instances_.size();

  return logl;
}

double MaxEnt::CalcHeldoutLikelihood() {
  double logl = 0;
  int32_t ncorrect = 0;

  for (std::vector<MemInstance>::const_iterator citer = heldout_.begin();
       citer != heldout_.end();
       ++citer) {
    std::vector<double> prob_dist(model_data_.NumClasses());
    int32_t label_id = Classify(*citer, &prob_dist);
    logl += log(prob_dist[citer->label_id()]);
    if (label_id == citer->label_id()) { ++ncorrect; }
  }

  heldout_accuracy_ = static_cast<double>(ncorrect) / heldout_.size();

  return logl /= heldout_.size();
}

// p(y | x)
int32_t MaxEnt::Classify(const MemInstance& mem_instance,
                         std::vector<double>* prob_dist) const {
  assert(model_data_.NumClasses() == static_cast<int32_t>(prob_dist->size()));

  CalcConditionalProbability(mem_instance, prob_dist);

  int32_t max_label = 0;
  double max_prob = 0.0;
  for (int32_t i = 0; i < static_cast<int32_t>(prob_dist->size()); ++i) {
    if ((*prob_dist)[i] > max_prob) {
      max_label = i;
      max_prob = (*prob_dist)[i];
    }
  }

  return max_label;
}

int32_t MaxEnt::CalcConditionalProbability(
    const MemInstance& mem_instance, std::vector<double>* prob_dist) const {
  const std::vector<double>& lambdas = model_data_.Lambdas();
  std::vector<double> powv(model_data_.NumClasses(), 0.0);

  for (MemInstance::ConstIterator citer(mem_instance);
       !citer.Done(); citer.Next()) {
    const std::vector<int32_t>& feature_ids
        = model_data_.FeatureIds(citer.FeatureNameId());
    for (size_t i = 0; i < feature_ids.size(); ++i) {
      const int32_t feature_id = feature_ids[i];
      powv[model_data_.FeatureAt(feature_id).LabelId()]
          += lambdas[feature_id] * citer.FeatureValue();
    }
  }

  std::vector<double>::const_iterator pmax
      = max_element(powv.begin(), powv.end());
  double sum = 0.0;
  double offset = std::max(0.0, *pmax - 700);  // to avoid overflow
  for (int32_t label_id = 0; label_id < model_data_.NumClasses(); ++label_id) {
    double pow_value = powv[label_id] - offset;
    double prod = exp(pow_value);  // exp(w * x)
    assert(prod != 0);

    (*prob_dist)[label_id] = prod;
    sum += prod;
  }

  int32_t max_label = 0;
  if (sum > 0.0) {
    for (int32_t label_id = 0; label_id < model_data_.NumClasses();
         ++label_id) {
      (*prob_dist)[label_id] /= sum;
      if ((*prob_dist)[label_id] > (*prob_dist)[max_label]) {
        max_label = label_id;
      }
    }
  }
  assert(max_label >= 0);

  return max_label;
}

}  // namespace maxent
}  // namespace mltk
