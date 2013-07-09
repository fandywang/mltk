// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/lbfgs.h"

#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "mltk/common/double_vector.h"
#include "mltk/common/instance.h"
#include "mltk/common/model_data.h"

namespace mltk {
namespace maxent {

using mltk::common::DoubleVector;
using mltk::common::Instance;
using mltk::common::ModelData;

const static double line_search_alpha_ = 0.1;
const static double line_search_beta_ = 0.5;
// stopping criteria
const static double MIN_GRAD_NORM = 0.0001;

void LBFGS::EstimateParamater(const std::vector<Instance>& instances,
                              int32_t num_heldout,
                              ModelData* model_data) {
  std::cerr << "performing LBFGS" << std::endl;
  if (l1reg_ > 0) {
    std::cerr << "error: L1 regularization is not supported in LBFGS,"
        << "you can use OWLQN method instead." << std::endl;
    exit(1);
  }

  InitFromInstances(instances, num_heldout, model_data);

  std::vector<double> x = PerformLBFGS();
  model_data_->UpdateLambdas(x);
}

std::vector<double> LBFGS::PerformLBFGS() {
  const std::vector<double> lambdas = model_data_->Lambdas();
  assert(static_cast<int32_t>(lambdas.size()) == model_data_->NumFeatures());

  std::vector<double> x0(lambdas.size());
  for (int32_t i = 0; i < lambdas.size(); ++i) { x0[i] = lambdas[i]; }

  DoubleVector x(x0);
  DoubleVector grad(lambdas.size());
  double f = FunctionGradient(x.STLVector(), &(grad.STLVector()));

  DoubleVector* s = new DoubleVector[m_];
  DoubleVector* y = new DoubleVector[m_];
  double* z = new double[m_];  // rho

  for (int32_t iter = 0; iter < num_iter_; ++iter) {  // stopping criteria 1
    std::cerr << "iter = " << iter + 1
        << ", obj(err) = " << f
        << ", accuracy = " << train_accuracy_ << std::endl;

    if (heldout_data_.size() > 0) {
      const double heldout_logl = CalcHeldoutLikelihood();
      std::cerr << "\theldout_logl(err) = " << -1 * heldout_logl
          << ", accuracy = " << heldout_accuracy_ << std::endl;
    }

    // stopping criteria 2
    if (sqrt(DotProduct(grad, grad)) < MIN_GRAD_NORM) { break; }

    DoubleVector dx = -1 * ApproximateHg(iter, grad, s, y, z);

    DoubleVector x1(lambdas.size()), grad1(lambdas.size());
    f = BacktrackingLineSearch(x, grad, f, dx, &x1, &grad1);

    s[iter % m_] = x1 - x;
    y[iter % m_] = grad1 - grad;
    z[iter % m_] = 1.0 / DotProduct(y[iter % m_], s[iter % m_]);
    x = x1;
    grad = grad1;
  }
  delete[] s;
  delete[] y;
  delete[] z;

  return x.STLVector();
}

DoubleVector LBFGS::ApproximateHg(const int32_t iter,
                                  const DoubleVector& grad,
                                  const DoubleVector* s,
                                  const DoubleVector* y,
                                  const double* z) {
  int32_t offset, bound;
  if (iter <= m_) {
    offset = 0;
    bound = iter;
  } else {
    offset = iter - m_;
    bound = m_;
  }

  DoubleVector q = grad;
  double alpha[m_], beta[m_];
  for (int32_t i = bound - 1; i >= 0; --i) {
    const int32_t j = (i + offset) % m_;
    alpha[i] = z[j] * DotProduct(s[j], q);
    q += -alpha[i] * y[j];
  }
  if (iter > 0) {
    const int32_t j = (iter - 1) % m_;
    const double gamma = ((1.0 / z[j]) / DotProduct(y[j], y[j]));
    q *= gamma;
  }
  for (int32_t i = 0; i <= bound - 1; ++i) {
    const int32_t j = (i + offset) % m_;
    beta[i] = z[j] * DotProduct(y[j], q);
    q += s[j] * (alpha[i] - beta[i]);
  }

  return q;
}

double LBFGS::BacktrackingLineSearch(const DoubleVector& x0,
                                     const DoubleVector& grad0,
                                     const double f0,
                                     const DoubleVector& dx,
                                     DoubleVector* x,
                                     DoubleVector* grad1) {
  double t = 1.0 / line_search_beta_;
  double f;

  do {
    t *= line_search_beta_;
    *x = x0 + t * dx;
    f = FunctionGradient(x->STLVector(), &(grad1->STLVector()));
  } while (f > f0 + line_search_alpha_ * t * DotProduct(dx, grad0));

  return f;
}

}  // namespace maxent
}  // namespace mltk
