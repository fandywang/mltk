// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/optimizer.h"

#include <vector>

#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"
#include "mltk/common/model_data.h"

namespace mltk {
namespace maxent {

using mltk::common::Instance;
using mltk::common::MemInstance;
using mltk::common::ModelData;

bool Optimizer::InitFromInstances(const std::vector<Instance>& instances,
                                  int32_t num_heldout,
                                  int32_t feature_cutoff,
                                  ModelData* model_data) {
  std::cerr << "preparing for estimation..." << std::endl;

  // initialize model
  std::cerr << "initialize model data...";
  assert(model_data != NULL);
  model_data_ = model_data;  // for convient
  model_data_->InitFromInstances(instances, feature_cutoff);

  // mapping common::Instance to common::MemInstance
  for (size_t n = 0; n < instances.size(); ++n) {
    train_data_.push_back(MemInstance());
    MemInstance& mem_instance = train_data_.back();

    model_data_->FormatInstance(instances[n], &mem_instance);
  }
  if (train_data_.size() == 0) {
    std::cerr << "error: no training data." << std::endl;
    return false;
  }
  std::cerr << "done" << std::endl;

  // preparing for heldout data
  if (num_heldout >= static_cast<int32_t>(train_data_.size())) {
    std::cerr << "error: too much heldout data. no training data is available."
        << std::endl;
    return false;
  }
  for (int32_t i = 0; i < num_heldout; ++i) {
    heldout_data_.push_back(train_data_.back());
    train_data_.pop_back();
  }

  std::cerr << "number of classes = " << model_data_->NumClasses() << std::endl;
  std::cerr << "number of features = " << model_data_->NumFeatures()
      << std::endl;
  std::cerr << "number of training instances = " << train_data_.size()
      << std::endl;
  std::cerr << "number of heldout instances = " << heldout_data_.size()
      << std::endl;

  // normalize l1 & l2 regularizer
  if (l1reg_ > 0) {
    l1reg_ /= train_data_.size();
    std::cerr << "L1 regularizer = " << l1reg_ << std::endl;
  }
  if (l2reg_ > 0) {
    l2reg_ /= train_data_.size();
    std::cerr << "L2 regularizer = " << l2reg_ << std::endl;
  }
  if (l1reg_ > 0 && l2reg_ > 0) {
    std::cerr << "error: L1 and L2 regularizers cannot be used simultaneously."
         << std::endl;
    return false;
  }

  InitEmpiricalExpection();

  return true;
}

void Optimizer::InitEmpiricalExpection() {
  // calc E_p1 (f), p1(x, y) = count(x, y) / N
  std::cerr << "calculating empirical expectation...";

  empirical_expectation_.resize(model_data_->NumFeatures());
  for (int32_t i = 0; i < model_data_->NumFeatures(); ++i) {
    empirical_expectation_[i] = 0;
  }

  for (size_t n = 0; n < train_data_.size(); ++n) {
    for (MemInstance::ConstIterator citer(train_data_[n]);
         !citer.Done(); citer.Next()) {
      const std::vector<int32_t> feature_ids
          = model_data_->FeatureIds(citer.FeatureNameId());
      for (size_t i = 0; i < feature_ids.size(); ++i) {
        if (model_data_->FeatureAt(feature_ids[i]).LabelId()
            == citer.LabelId()) {
          empirical_expectation_[feature_ids[i]] += citer.FeatureValue();
          break;
        }
      }
    }
  }

  for (int32_t i = 0; i < model_data_->NumFeatures(); ++i) {
    empirical_expectation_[i] /= train_data_.size();
  }
  std::cerr << "done" << std::endl;
}

double Optimizer::FunctionGradient(const std::vector<double>& x,
                                   std::vector<double>* grad) {
  assert(static_cast<size_t>(model_data_->NumFeatures()) == x.size());

  model_data_->UpdateLambdas(x);
  double score = UpdateModelExpectation();

  // update gradient
  if (l2reg_ == 0) {
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = model_expectation_[i] - empirical_expectation_[i];
    }
  } else {
    const double c = l2reg_ * 2;
    const std::vector<double>& lambdas = model_data_->Lambdas();
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = model_expectation_[i] - empirical_expectation_[i]
                   + c * lambdas[i];
    }
  }

  return -score;
}

double Optimizer::UpdateModelExpectation() {
  double logl = 0;
  int32_t ncorrect = 0;

  model_expectation_.resize(model_data_->NumFeatures());
  for (int i = 0; i < model_data_->NumFeatures(); ++i) {
    model_expectation_[i] = 0;
  }

  for (size_t n = 0; n < train_data_.size(); ++n) {
    std::vector<double> prob_dist(model_data_->NumClasses());
    int32_t max_label = model_data_->CalcConditionalProbability(
        train_data_[n], &prob_dist);

    logl += log(prob_dist[train_data_[n].label_id()]);
    if (max_label == train_data_[n].label_id()) { ++ncorrect; }

    // model_expectation
    for (MemInstance::ConstIterator citer(train_data_[n]);
         !citer.Done(); citer.Next()) {
      const std::vector<int32_t>& feature_ids
          = model_data_->FeatureIds(citer.FeatureNameId());
      for (size_t i = 0; i < feature_ids.size(); ++i) {
        const int32_t feature_id = feature_ids[i];
        model_expectation_[feature_id]
          += prob_dist[model_data_->FeatureAt(feature_id).LabelId()]
             * citer.FeatureValue();
      }
    }
  }

  const std::vector<double>& lambdas = model_data_->Lambdas();
  for (int32_t i = 0; i < model_data_->NumFeatures(); ++i) {
    model_expectation_[i] /= train_data_.size();
    if (l2reg_ > 0) { logl -= lambdas[i] * lambdas[i] * l2reg_; }
  }

  train_accuracy_ = static_cast<double>(ncorrect) / train_data_.size();

  return logl / train_data_.size();
}

double Optimizer::CalcHeldoutLikelihood() {
  double logl = 0;
  int32_t ncorrect = 0;

  for (std::vector<MemInstance>::const_iterator citer = heldout_data_.begin();
       citer != heldout_data_.end(); ++citer) {
    std::vector<double> prob_dist(model_data_->NumClasses());
    int32_t label_id = model_data_->CalcConditionalProbability(*citer,
                                                              &prob_dist);
    logl += log(prob_dist[citer->label_id()]);
    if (label_id == citer->label_id()) { ++ncorrect; }
  }

  heldout_accuracy_ = static_cast<double>(ncorrect) / heldout_data_.size();

  return logl / heldout_data_.size();
}

}  // namespace maxent
}  // namespace mltk

