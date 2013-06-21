// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// the implementation of Stochastic Gradient Descent (SGD) algorithm
//
// Pls refer to 'Yoshimasa Tsuruoka, Jun'ichi Tsujii, and Sophia Ananiadou.
// 2009. Stochastic Gradient Descent Training for L1-regularized Log-linear
// Models with Cumulative Penalty, In Proceedings of ACL-IJCNLP'

#include "mltk/maxent/maxent.h"

#include <math.h>
#include <iostream>
#include <vector>

#include "mltk/common/instance.h"

namespace mltk {
namespace maxent {

using mltk::common::MemInstance;

const static double SGD_ITER = 50;  // the total number of iterater
const static double SGD_ETA0 = 1;  // learning rate eta_0
const static double SGD_ALPHA = 0.85;  // the constant for learning rate
                                       // exponential delay.
                                       // eta_k = eta_0 * alpha^(-k/N)

inline void ApplyL1Penalty(const int32_t i,
                           const double u,
                           std::vector<double>& lambas,
                           std::vector<double>& q) {
  double& w = lambas[i];
  const double z = w;
  if (w > 0) {
    w = std::max(0.0, w - (u + q[i]));
  } else if (w < 0) {
    w = std::min(0.0, w + (u - q[i]));
  }
  q[i] += w - z;
}

static double L1Norm(const std::vector<double>& v) {
  double sum = 0;
  for (size_t i = 0; i < v.size(); ++i) { sum += abs(v[i]); }
  return sum;
}

int32_t MaxEnt::PerformSGD() {
  if (l2reg_ > 0) {
    std::cerr << "error: L2 regularization is currently not supported in SGD."
        << std::endl;
    exit(1);
  }
  std::cerr << "performing SGD" << std::endl;
  assert(SGD_ALPHA < 1.0 && SGD_ALPHA > 0.0);
  std::cerr << "eta0 = " << SGD_ETA0 << ", alpha = " << SGD_ALPHA << std::endl;

  std::vector<int32_t> instance_indexs(me_instances_.size());
  for (size_t i = 0; i < instance_indexs.size(); ++i) {
    instance_indexs[i] = i;
  }

  const double l1param = l1reg_;
  double u = 0;  // u_k = C/N * sum_{t=1}^k {eta_t}
  std::vector<double> q(feature_vocab_.Size(), 0);  // q_i^k = sum_{t=1}^k {w_i^(t+1) - w_i^(t+1/2)}
  int32_t iter_sample = 0;  // the number of iter sample

  for (int32_t iter = 0; iter < SGD_ITER; ++iter) {
    int32_t ncorrect = 0;
    double logl = 0.0;

    random_shuffle(instance_indexs.begin(), instance_indexs.end());

    // batch size is 1, which is the extrem case.
    for (size_t i = 0; i < me_instances_.size(); ++i, ++iter_sample) {
      const MemInstance& me_instance = me_instances_[instance_indexs[i]];

      std::vector<double> prob_dist(NumClasses());
      const int32_t max_label = CalcConditionalProbability(me_instance,
                                                           &prob_dist);
      logl += log(prob_dist[me_instance.label]);
      if (max_label == me_instance.label) { ++ncorrect; }

      // learning rate : exponential decay
      const double eta = SGD_ETA0 *
          pow(SGD_ALPHA,
              static_cast<double>(iter_sample) / me_instances_.size());
      u += eta * l1param;

      // update weight/lambdas according to current sampled instance
      for (std::vector<std::pair<int32_t, double> >::const_iterator j
           = me_instance.features.begin();
           j != me_instance.features.end(); ++j) {
        for (std::vector<int32_t>::const_iterator k
             = all_me_features_[j->first].begin();
             k != all_me_features_[j->first].end(); ++k) {
          const double me = prob_dist[feature_vocab_.GetFeature(*k).LabelId()];
          const double ee = (feature_vocab_.GetFeature(*k).LabelId()
                             == me_instance.label ? 1.0 : 0);
          const double grad = (me - ee) * j->second;
          lambdas_[*k] -= eta * grad;  // GD

          ApplyL1Penalty(*k, u, lambdas_, q);
        }
      }
    }

    logl /= me_instances_.size();
    double f = logl;
    if (l1param > 0) {
      const double l1 = L1Norm(lambdas_);
      f -= l1param * l1;
    }

    std::cerr << "iter = " << iter + 1 << ", obj(err) = " << f
        << ", accuracy = "
        << static_cast<double>(ncorrect) / me_instances_.size() << std::endl;

    if (num_heldout_ > 0) {
      double heldout_logl = CalcHeldoutLikelihood();
      std::cerr << "\t heldout_logl(err) = " << heldout_logl
          << ", accuracy = " << heldout_accuracy_ << std::endl;
    }
  }

  return 0;
}

}  // namespace maxent
}  // namespace mltk
