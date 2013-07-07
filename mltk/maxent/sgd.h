// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Implementation of Stochastic Gradient Descent (SGD) algorithm
//
// Pls refer to 'Yoshimasa Tsuruoka, Jun'ichi Tsujii, and Sophia Ananiadou.
// 2009. Stochastic Gradient Descent Training for L1-regularized Log-linear
// Models with Cumulative Penalty, In Proceedings of ACL-IJCNLP'

#ifndef MLTK_MAXENT_SGD_H_
#define MLTK_MAXENT_SGD_H_

#include "mltk/maxent/optimizer.h"

#include <vector>

namespace mltk {

namespace common {
class Instance;
class ModelData;
}  // namespace common

namespace maxent {

class SGD : public Optimizer {
 public:
  SGD() {}
  virtual ~SGD() {}

  virtual void EstimateParamater(const std::vector<common::Instance>& instances,
                                 int32_t num_heldout,
                                 common::ModelData* model_data);

 private:
  void PerformSGD();

  void ApplyL1Penalty(const size_t id,
                      const double u,
                      std::vector<double>* lambdas,
                      std::vector<double>& q);
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_SGD_H_

