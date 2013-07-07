// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Implementation of LBFGS algorithm.
//
// Pls refer to 'Jorge Nocedal, "Updating Quasi-Newton Matrices With Limited
// Storage", Mathematics of Computation, 1980.'

#ifndef MLTK_MAXENT_LBFGS_H_
#define MLTK_MAXENT_LBFGS_H_

#include "mltk/maxent/optimizer.h"

#include <vector>

namespace mltk {

namespace common {
class DoubleVector;
class Instance;
class ModelData;
}  // namespace common

namespace maxent {

class LBFGS : public Optimizer {
 public:
  LBFGS() {}
  virtual ~LBFGS() {}

  virtual void EstimateParamater(const std::vector<common::Instance>& instances,
                                 int32_t num_heldout,
                                 common::ModelData* model_data);

 private:
  std::vector<double> PerformLBFGS(const std::vector<double>& x0);

  common::DoubleVector ApproximateHg(const int32_t iter,
                                     const common::DoubleVector& grad,
                                     const common::DoubleVector s[],
                                     const common::DoubleVector y[],
                                     const double z[]);

  double BacktrackingLineSearch(const common::DoubleVector& x0,
                                const common::DoubleVector& grad0,
                                const double f0,
                                const common::DoubleVector& dx,
                                common::DoubleVector* x,
                                common::DoubleVector* grad1);
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_LBFGS_H_

