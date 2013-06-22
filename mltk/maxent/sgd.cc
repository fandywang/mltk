// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Implementation of Stochastic Gradient Descent (SGD) algorithm
//
// Pls refer to 'Yoshimasa Tsuruoka, Jun'ichi Tsujii, and Sophia Ananiadou.
// 2009. Stochastic Gradient Descent Training for L1-regularized Log-linear
// Models with Cumulative Penalty, In Proceedings of ACL-IJCNLP'

#include "mltk/maxent/maxent.h"

#include <math.h>
#include <iostream>
#include <vector>

#include "mltk/common/mem_instance.h"

namespace mltk {
namespace maxent {

using mltk::common::MemInstance;

const static double SGD_ITER = 50;  // the total number of iterater
const static double SGD_ETA0 = 1;  // learning rate eta_0
const static double SGD_ALPHA = 0.85;  // the constant for learning rate
                                       // exponential delay.
                                       // eta_k = eta_0 * alpha^(-k/N)

inline void ApplyL1Penalty(const size_t id,
                           const double u,
                           std::vector<double>& lambas,
                           std::vector<double>& q) {
  double& w = lambas[id];
  const double z = w;
  if (w > 0) {
    w = std::max(0.0, w - (u + q[id]));
  } else if (w < 0) {
    w = std::min(0.0, w + (u - q[id]));
  }
  q[id] += w - z;
}

static double L1Norm(const std::vector<double>& v) {
  double sum = 0.0;
  for (size_t i = 0; i < v.size(); ++i) { sum += abs(v[i]); }
  return sum;
}

int32_t MaxEnt::PerformSGD() {
  std::cerr << "performing SGD" << std::endl;
  if (l2reg_ > 0) {
    std::cerr << "error: L2 regularization is currently not supported in SGD."
        << std::endl;
    exit(1);
  }
  assert(SGD_ALPHA < 1.0 && SGD_ALPHA > 0.0);
  std::cerr << "eta0 = " << SGD_ETA0 << ", alpha = " << SGD_ALPHA << std::endl;

  std::vector<int32_t> instance_ids(mem_instances_.size());
  for (size_t i = 0; i < instance_ids.size(); ++i) { instance_ids[i] = i; }

  const double l1param = l1reg_;
  double u = 0;  // u_k = C/N * sum_{t=1}^k {eta_t}
  std::vector<double> q(feature_vocab_.Size(), 0);  // q_i^k = sum_{t=1}^k {w_i^(t+1) - w_i^(t+1/2)}
  int32_t iter_sample = 0;  // the number of iter sample

  for (int32_t iter = 0; iter < SGD_ITER; ++iter) {
    int32_t ncorrect = 0;
    double logl = 0.0;

    random_shuffle(instance_ids.begin(), instance_ids.end());

    // batch size is 1, which is the extrem case.
    for (size_t i = 0; i < mem_instances_.size(); ++i, ++iter_sample) {
      const MemInstance& mem_instance = mem_instances_[instance_ids[i]];

      std::vector<double> prob_dist(NumClasses());
      const int32_t max_label = CalcConditionalProbability(mem_instance,
                                                           &prob_dist);
      logl += log(prob_dist[mem_instance.label()]);
      if (max_label == mem_instance.label()) { ++ncorrect; }

      // learning rate : exponential decay
      const double eta = SGD_ETA0 *
          pow(SGD_ALPHA,
              static_cast<double>(iter_sample) / mem_instances_.size());
      u += eta * l1param;

      // update weight/lambdas according to current sampled instance
      for (MemInstance::ConstIterator citer(mem_instance);
           !citer.Done(); citer.Next()) {
        for (size_t i = 0; i < all_me_features_[citer.FeatureId()].size(); ++i) {
          const int32_t id = all_me_features_[citer.FeatureId()][i];
          const double me
            = prob_dist[feature_vocab_.GetFeature(id).LabelId()];
          const double ee = (feature_vocab_.GetFeature(id).LabelId()
                             == citer.LabelId() ? 1.0 : 0);
          const double grad = (me - ee) * citer.FeatureValue();
          lambdas_[id] -= eta * grad;  // GD

          ApplyL1Penalty(id, u, lambdas_, q);
        }
      }
    }

    logl /= mem_instances_.size();
    double f = - logl;
    if (l1param > 0) {
      const double l1 = L1Norm(lambdas_);
      f += l1param * l1;
    }

    std::cerr << "iter = " << iter + 1 << ", obj(err) = " << f
        << ", accuracy = "
        << static_cast<double>(ncorrect) / mem_instances_.size() << std::endl;

    if (num_heldout_ > 0) {
      double heldout_logl = CalcHeldoutLikelihood();
      std::cerr << "\t heldout_logl(err) = " << -1 * heldout_logl
          << ", accuracy = " << heldout_accuracy_ << std::endl;
    }
  }

  return 0;
}

}  // namespace maxent
}  // namespace mltk
