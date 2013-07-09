// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/sgd.h"

#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "mltk/common/feature.h"
#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"
#include "mltk/common/model_data.h"

namespace mltk {
namespace maxent {

using mltk::common::Feature;
using mltk::common::Instance;
using mltk::common::MemInstance;
using mltk::common::ModelData;

const static double ALPHA = 0.85;  // the constant for learning rate
                                   // exponential delay.
                                   // eta_k = eta_0 * alpha^(-k / N)

void SGD::EstimateParamater(const std::vector<Instance>& instances,
                            int32_t num_heldout,
                            int32_t feature_cutoff,
                            ModelData* model_data) {
  std::cerr << "performing SGD" << std::endl;
  if (l2reg_ > 0) {
    std::cerr << "error: L2 regularization is currently not supported in SGD."
        << std::endl;
    exit(1);
  }

  InitFromInstances(instances, num_heldout, feature_cutoff, model_data);
  PerformSGD();
}

void SGD::PerformSGD() {
  assert(ALPHA < 1.0 && ALPHA > 0.0);
  std::cerr << "learning_rate = " << learning_rate_
      << ", alpha = " << ALPHA << std::endl;


  std::vector<int32_t> instance_ids(train_data_.size());
  for (size_t i = 0; i < instance_ids.size(); ++i) { instance_ids[i] = i; }

  const double l1param = l1reg_;
  double u = 0;  // u_k = C/N * sum_{t=1}^k {eta_t}
  std::vector<double> q(model_data_->NumFeatures(), 0);  // q_i^k = sum_{t=1}^k {w_i^(t+1) - w_i^(t+1/2)}
  int32_t iter_sample = 0;  // the number of iter sample
  std::vector<double>* lambdas = model_data_->MutableLambdas();

  for (int32_t iter = 0; iter < num_iter_; ++iter) {
    int32_t ncorrect = 0;
    double logl = 0.0;

    random_shuffle(instance_ids.begin(), instance_ids.end());

    // batch size is 1, which is the extreme case.
    for (size_t i = 0; i < train_data_.size(); ++i, ++iter_sample) {
      const MemInstance& mem_instance = train_data_[instance_ids[i]];

      std::vector<double> prob_dist(model_data_->NumClasses());
      const int32_t max_label =
          model_data_->CalcConditionalProbability(mem_instance, &prob_dist);
      logl += log(prob_dist[mem_instance.label_id()]);
      if (max_label == mem_instance.label_id()) { ++ncorrect; }

      // learning rate : exponential decay
      const double eta = learning_rate_ *
          pow(ALPHA, static_cast<double>(iter_sample) / train_data_.size());
      u += eta * l1param;

      // update weight/lambdas according to current sampled instance
      for (MemInstance::ConstIterator citer(mem_instance);
           !citer.Done(); citer.Next()) {
        const std::vector<int32_t>& feature_ids
            = model_data_->FeatureIds(citer.FeatureNameId());
        for (size_t i = 0; i < feature_ids.size(); ++i) {
          const int32_t feature_id = feature_ids[i];
          const Feature& feature = model_data_->FeatureAt(feature_id);
          const double me = prob_dist[feature.LabelId()];
          const double ee = (feature.LabelId() == citer.LabelId() ? 1.0 : 0);
          const double grad = (me - ee) * citer.FeatureValue();
          (*lambdas)[feature_id] -= eta * grad;  // GD

          ApplyL1Penalty(feature_id, u, lambdas, q);
        }
      }
    }

    logl /= train_data_.size();
    double f = - logl;
    if (l1param > 0) {
      const double l1 = model_data_->L1NormLambdas();
      f += l1param * l1;
    }

    std::cerr << "iter = " << iter + 1 << ", obj(err) = " << f
        << ", accuracy = "
        << static_cast<double>(ncorrect) / train_data_.size() << std::endl;

    if (heldout_data_.size() > 0) {
      double heldout_logl = CalcHeldoutLikelihood();
      std::cerr << "\t heldout_logl(err) = " << -1 * heldout_logl
          << ", accuracy = " << heldout_accuracy_ << std::endl;
    }
  }
}

void SGD::ApplyL1Penalty(const size_t id,
                         const double u,
                         std::vector<double>* lambdas,
                         std::vector<double>& q) {
  double& w = (*lambdas)[id];
  const double z = w;
  if (w > 0) {
    w = std::max(0.0, w - (u + q[id]));
  } else if (w < 0) {
    w = std::min(0.0, w + (u - q[id]));
  }
  q[id] += w - z;
}

}  // namespace maxent
}  // namespace mltk
