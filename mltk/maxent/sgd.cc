// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>

namespace mltk {
namespace maxent {

const double SGD_ETA0 = 1;
const double SGD_ITER = 30;
const double SGD_ALPHA = 0.85;

inline void ApplyL1Penalty(const int32_t i,
                           const double u,
                           std::vector<double>& lambas,
                           std::vector<double>& q) {
  double& w = lambas[i];
  const double z = w;
  double& qi = q[i];
  if (w > 0) {
    w = std::max(0.0, w - (u + qi));
  } else if (w < 0) {
    w = std::min(0.0, w + (u - qi));
  }
  qi += w - z;
}

static double L1Norm(const std::vector<double>& v) {
  double sum = 0;
  for (size_t i = 0; i < v.size(); ++i) sum += abs(v[i]);
  return sum;
}

int32_t MaxEnt::PerformSGD() {
  if (l2reg_ > 0) {
    std::cerr << "error: L2 regularization is currently not supported in SGD."
        << std::endl;
    exit(1);
  }
  std::cerr << "performing SGD" << std::endl;

  const double l1param = l1reg_;
  const int32_t d = feature_vocab_.Size();

  std::vector<int32_t> ri(me_instances_.size());
  for (size_t i = 0; i < ri.size(); ++i) ri[i] = i;

  std::vector<double> grad(d);
  int32_t iter_sample = 0;
  const double eta0 = SGD_ETA0;

  std::cerr << "eta0 = " << eta0 << " alpha = " << SGD_ALPHA << std::endl;

  double u = 0;
  std::vector<double> q(d, 0);
  std::vector<int> last_updated(d, 0);
  std::vector<double> sum_eta;
  sum_eta.push_back(0);

  for (int32_t iter = 0; iter < SGD_ITER; ++iter) {
    double logl = 0;
    int32_t ncorrect = 0;
    int32_t ntotal = 0;

    random_shuffle(ri.begin(), ri.end());
    for (size_t i = 0; i < me_instances_.size(); ++i, ++ntotal, ++iter_sample) {
      const MaxEntInstance& me_instance = me_instances_[ri[i]];

      std::vector<double> prob_dist(num_classes_);
      const int32_t max_label = CalcConditionalProbability(me_instance,
                                                           &prob_dist);

      const double eta = eta0 *
          pow(SGD_ALPHA, static_cast<double>(iter_sample) / me_instances_.size()); // exponential decay
      u += eta * l1param;

      sum_eta.push_back(sum_eta.back() + eta * l1param);

      logl += log(prob_dist[me_instance.label]);
      if (max_label == me_instance.label) { ++ncorrect; }

      // real-valued features
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
          lambdas_[*k] -= eta * grad;

          ApplyL1Penalty(*k, u, lambdas_, q);
        }
      }
    }
    logl /= me_instances_.size();

    double f = logl;
    if (l1param > 0) {
      const double l1 = L1Norm(lambdas_); // this is not accurate when lazy update is used
      f -= l1param * l1;
      int32_t nonzero = 0;
      for (int32_t j = 0; j < d; ++j) if (lambdas_[j] != 0) ++nonzero;
    }

    fprintf(stderr, "%3d  obj(err) = %f (%6.4f)",
            iter + 1, f, 1 - static_cast<double>(ncorrect) / ntotal);
    if (num_heldout_ > 0) {
      double heldout_logl = CalcHeldoutLikelihood();
      fprintf(stderr, "\theldout_logl(err) = %f (%6.4f)",
              heldout_logl, heldout_error_);
    }
    fprintf(stderr, "\n");
  }

  return 0;
}

}  // namespace maxent
}  // namespace mltk
