// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Implementation of OWLQN algorithm.
//
// Pls refer to 'Galen Andrew and Jianfeng Gao, Scalable training of
// L1-regularized log-linear models, in ICML 2007.'

#ifndef MLTK_MAXENT_OWLQN_H_
#define MLTK_MAXENT_OWLQN_H_

#include "mltk/maxent/optimizer.h"

#include <vector>

namespace mltk {

namespace common {
class DoubleVector;
class Instance;
class ModelData;
}  // namespace common

namespace maxent {

class OWLQN : public Optimizer {
 public:
  OWLQN(int32_t num_iter = 300, int32_t m = 10) : num_iter_(num_iter), m_(m) {}
  virtual ~OWLQN() {}

  virtual void EstimateParamater(const std::vector<common::Instance>& instances,
                                 int32_t num_heldout,
                                 common::ModelData* model_data);

 private:
  std::vector<double> PerformOWLQN();

  double RegularizedFuncGrad(const double C,
                             const common::DoubleVector& x,
                             common::DoubleVector& grad);

  common::DoubleVector PseudoGradient(const common::DoubleVector& x,
                                      const common::DoubleVector& grad0,
                                      const double C);

  common::DoubleVector ApproximateHg(const int32_t iter,
                                     const common::DoubleVector& grad,
                                     const common::DoubleVector* s,
                                     const common::DoubleVector* y,
                                     const double* z);

  double ConstrainedLineSearch(double C,
                               const common::DoubleVector& x0,
                               const common::DoubleVector& grad0,
                               const double f0,
                               const common::DoubleVector& dx,
                               common::DoubleVector& x,
                               common::DoubleVector& grad1);

  int32_t num_iter_;  // the total iterations
  int32_t m_;
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_OWLQN_H_

