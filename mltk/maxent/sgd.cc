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

inline void ApplyL1Penalty(const int i,
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

  //  cerr << "l1param = " << l1param << endl;
  std::cerr << "eta0 = " << eta0 << " alpha = " << SGD_ALPHA << std::endl;

  double u = 0;
  std::vector<double> q(d, 0);
  std::vector<int> last_updated(d, 0);
  std::vector<double> sum_eta;
  sum_eta.push_back(0);

  for (int32_t iter = 0; iter < SGD_ITER; ++iter) {
    random_shuffle(ri.begin(), ri.end());

    double logl = 0;
    int32_t ncorrect = 0, ntotal = 0;
    for (size_t i = 0; i < me_instances_.size(); ++i, ++ntotal, ++iter_sample) {
      const MaxEntInstance& me_instance = me_instances_[ri[i]];

      std::vector<double> membp(num_classes_);
      const int32_t max_label = CalcConditionalProbability(me_instance, &membp);

      const double eta = eta0 * pow(SGD_ALPHA,
                                    (double)iter_sample / me_instances_.size()); // exponential decay
      //      const double eta = eta0 / (1.0 + (double)iter_sample / me_instances_.size());

      //      if (iter_sample % me_instances_.size() == 0) cerr << "eta = " << eta << endl;
      u += eta * l1param;

      sum_eta.push_back(sum_eta.back() + eta * l1param);

      logl += log(membp[me_instance.label]);
      if (max_label == me_instance.label) { ++ncorrect; }

      // real-valued features
      for (std::vector<std::pair<int32_t, double> >::const_iterator j
           = me_instance.features.begin();
           j != me_instance.features.end(); ++j) {
        for (std::vector<int32_t>::const_iterator k
             = all_me_features_[j->first].begin();
             k != all_me_features_[j->first].end(); ++k) {
          const double me = membp[feature_vocab_.GetFeature(*k).LabelId()];
          const double ee = (feature_vocab_.GetFeature(*k).LabelId()
                             == me_instance.label ? 1.0 : 0);
          const double grad = (me - ee) * j->second;
          lambdas_[*k] -= eta * grad;

          ApplyL1Penalty(*k, u, lambdas_, q);
        }
      }
    }
    logl /= me_instances_.size();
    //    fprintf(stderr, "%4d logl = %8.3f acc = %6.4f ", iter, logl, (double)ncorrect / ntotal);

    double f = logl;
    if (l1param > 0) {
      const double l1 = L1Norm(lambdas_); // this is not accurate when lazy update is used
      //      cerr << "f0 = " <<  update_model_expectation() - l1param * l1 << " ";
      f -= l1param * l1;
      int nonzero = 0;
      for (int32_t j = 0; j < d; ++j) if (lambdas_[j] != 0) ++nonzero;
      //      cerr << " f = " << f << " l1 = " << l1 << " nonzero_features = " << nonzero << endl;
    }
    //    fprintf(stderr, "%4d  obj = %7.3f acc = %6.4f", iter+1, f, (double)ncorrect/ntotal);
    //    fprintf(stderr, "%4d  obj = %f", iter+1, f);
    fprintf(stderr, "%3d  obj(err) = %f (%6.4f)",
            iter + 1, f, 1 - static_cast<double>(ncorrect) / ntotal);

    if (num_heldout_ > 0) {
      double heldout_logl = CalcHeldoutLikelihood();
      //      fprintf(stderr, "  heldout_logl = %f  acc = %6.4f\n", heldout_logl, 1 - _heldout_error);
      fprintf(stderr, "  heldout_logl(err) = %f (%6.4f)",
              heldout_logl, heldout_error_);
    }
    fprintf(stderr, "\n");
  }

  return 0;
}

}  // namespace maxent
}  // namespace mltk
