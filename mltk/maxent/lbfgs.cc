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

const static int32_t LBFGS_M = 10;
const static double LINE_SEARCH_ALPHA = 0.1;
const static double LINE_SEARCH_BETA = 0.5;

// stopping criteria
const static int32_t LBFGS_MAX_ITER = 300;
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

  const std::vector<double> lambdas = model_data_->Lambdas();
  std::vector<double> x0(model_data_->NumFeatures());
  for (int32_t i = 0; i < model_data_->NumFeatures(); ++i) {
    x0[i] = lambdas[i];
  }

  std::vector<double> x = PerformLBFGS(x0);
  model_data_->UpdateLambdas(x);
}

std::vector<double> LBFGS::PerformLBFGS(const std::vector<double>& x0) {
  const size_t dim = x0.size();
  DoubleVector x(x0);
  DoubleVector grad(dim), dx(dim);

  double f = FunctionGradient(x.STLVector(), &(grad.STLVector()));

  DoubleVector s[LBFGS_M];
  DoubleVector y[LBFGS_M];
  double z[LBFGS_M];  // rho

  for (int32_t iter = 0; iter < LBFGS_MAX_ITER; ++iter) {  // stopping criteria 1
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

    dx = -1 * ApproximateHg(iter, grad, s, y, z);

    DoubleVector x1(dim), grad1(dim);
    f = BacktrackingLineSearch(x, grad, f, dx, &x1, &grad1);

    s[iter % LBFGS_M] = x1 - x;
    y[iter % LBFGS_M] = grad1 - grad;
    z[iter % LBFGS_M] = 1.0 / DotProduct(y[iter % LBFGS_M], s[iter % LBFGS_M]);
    x = x1;
    grad = grad1;
  }

  return x.STLVector();
}

DoubleVector LBFGS::ApproximateHg(const int32_t iter,
                                  const DoubleVector& grad,
                                  const DoubleVector s[],
                                  const DoubleVector y[],
                                  const double z[]) {
  int32_t offset, bound;
  if (iter <= LBFGS_M) {
    offset = 0;
    bound = iter;
  }
  else {
    offset = iter - LBFGS_M;
    bound = LBFGS_M;
  }

  DoubleVector q = grad;
  double alpha[LBFGS_M], beta[LBFGS_M];
  for (int32_t i = bound - 1; i >= 0; --i) {
    const int32_t j = (i + offset) % LBFGS_M;
    alpha[i] = z[j] * DotProduct(s[j], q);
    q += -alpha[i] * y[j];
  }
  if (iter > 0) {
    const int32_t j = (iter - 1) % LBFGS_M;
    const double gamma = ((1.0 / z[j]) / DotProduct(y[j], y[j]));
    q *= gamma;
  }
  for (int32_t i = 0; i <= bound - 1; ++i) {
    const int32_t j = (i + offset) % LBFGS_M;
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
  double t = 1.0 / LINE_SEARCH_BETA;
  double f;

  do {
    t *= LINE_SEARCH_BETA;
    *x = x0 + t * dx;
    f = FunctionGradient(x->STLVector(), &(grad1->STLVector()));
  } while (f > f0 + LINE_SEARCH_ALPHA * t * DotProduct(dx, grad0));

  return f;
}

}  // namespace maxent
}  // namespace mltk
