// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The base class of optimization algorithm.

#ifndef MLTK_MAXENT_OPTIMIZER_H_
#define MLTK_MAXENT_OPTIMIZER_H_

#include <vector>

#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"
#include "mltk/common/model_data.h"

namespace mltk {
namespace maxent {

class Optimizer {
 public:
  Optimizer() : model_data_(NULL), l1reg_(0.0), l2reg_(0.0) {}
  virtual ~Optimizer() {}

  void UseL1Reg(double l1reg) { l1reg_ = l1reg; }
  void UseL2Reg(double l2reg) { l2reg_ = l2reg; }

  // paramater estimation
  virtual void EstimateParamater(const std::vector<common::Instance>& instances,
                                 int32_t num_heldout,
                                 common::ModelData* model_data) = 0;

 protected:
  void Clear() {
    train_data_.clear();
    heldout_data_.clear();
    model_data_ = NULL;
    l1reg_ = 0.0;
    l2reg_ = 0.0;
    empirical_expectation_.clear();
    model_expectation_.clear();
  }

  bool InitFromInstances(const std::vector<common::Instance>& instances,
                         int32_t num_heldout,
                         common::ModelData* model_data);

  // Calculate empirical expection based on training data.
  void InitEmpiricalExpection();

  double FunctionGradient(const std::vector<double>& x,
                          std::vector<double>* grad);

  // Update E_p (f), formula: E_p (f) = sum_x,y P1(x)P(y|x)f(x, y)
  double UpdateModelExpectation();

  // Calculate p(y|x)
  int32_t CalcConditionalProbability(const common::MemInstance& mem_instance,
                                     std::vector<double>* prob_dist) const;
  double CalcHeldoutLikelihood();

 protected:
  std::vector<common::MemInstance> train_data_;  // training data
  double train_accuracy_;  // current accuracy on the training data

  std::vector<common::MemInstance> heldout_data_;  // heldout data
  double heldout_accuracy_;  // current accuracy on the heldout data

  common::ModelData* model_data_;  // the maxent model

  double l1reg_;  // L1-regularization
  double l2reg_;  // L2-regularization

  // E_p1(f), which is the expected value of f(x,y) with respect to the
  // empirical distribution p1(x,y).
  //
  // E_p1 (f) = sum_x,y P1(x, y)f(x, y)
  std::vector<double> empirical_expectation_;

  // E_p(f), which is the expected value of f(x,y) with respect to the
  // model p(y|x) and the expirical distribution p1(x).
  //
  // E_p (f) = sum_x,y P1(x)P(y|x)f(x, y)
  std::vector<double> model_expectation_;
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_OPTIMIZER_H_

