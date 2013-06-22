// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Implementation of LBFGS algorithm.
//
// Pls refer to 'Jorge Nocedal, "Updating Quasi-Newton Matrices With Limited
// Storage", Mathematics of Computation, 1980.'

#include "mltk/maxent/maxent.h"

#include <math.h>
#include <iostream>
#include <vector>

#include "mltk/common/double_vector.h"

namespace mltk {
namespace maxent {

using mltk::common::DoubleVector;

const static int32_t M = 10;
const static double LINE_SEARCH_ALPHA = 0.1;
const static double LINE_SEARCH_BETA = 0.5;

// stopping criteria
const static int32_t LBFGS_MAX_ITER = 300;
const static double MIN_GRAD_NORM = 0.0001;

DoubleVector ApproximateHg(const int32_t iter,
                           const DoubleVector& grad,
                           const DoubleVector s[],
                           const DoubleVector y[],
                           const double z[]) {
  int32_t offset, bound;
  if (iter <= M) {
    offset = 0;
    bound = iter;
  }
  else {
    offset = iter - M;
    bound = M;
  }

  DoubleVector q = grad;
  double alpha[M], beta[M];
  for (int32_t i = bound - 1; i >= 0; --i) {
    const int32_t j = (i + offset) % M;
    alpha[i] = z[j] * DotProduct(s[j], q);
    q += -alpha[i] * y[j];
  }
  if (iter > 0) {
    const int32_t j = (iter - 1) % M;
    const double gamma = ((1.0 / z[j]) / DotProduct(y[j], y[j]));
    q *= gamma;
  }
  for (int32_t i = 0; i <= bound - 1; ++i) {
    const int32_t j = (i + offset) % M;
    beta[i] = z[j] * DotProduct(y[j], q);
    q += s[j] * (alpha[i] - beta[i]);
  }

  return q;
}

std::vector<double> MaxEnt::PerformLBFGS(const std::vector<double>& x0) {
  const size_t dim = x0.size();
  DoubleVector x(x0);
  DoubleVector grad(dim), dx(dim);

  double f = FunctionGradient(x.STLVector(), &(grad.STLVector()));

  DoubleVector s[M];
  DoubleVector y[M];
  double z[M];  // rho

  for (int32_t iter = 0; iter < LBFGS_MAX_ITER; ++iter) {  // stopping criteria 1
    std::cerr << "iter = " << iter + 1
        << ", obj(err) = " << f
        << ", accuracy = " << train_accuracy_ << std::endl;

    if (heldout_.size() > 0) {
      const double heldout_logl = CalcHeldoutLikelihood();
      std::cerr << "\theldout_logl(err) = " << -1 * heldout_logl
          << ", accuracy = " << heldout_accuracy_ << std::endl;
    }

    // stopping criteria 2
    if (sqrt(DotProduct(grad, grad)) < MIN_GRAD_NORM) { break; }

    dx = -1 * ApproximateHg(iter, grad, s, y, z);

    DoubleVector x1(dim), grad1(dim);
    f = BacktrackingLineSearch(x, grad, f, dx, &x1, &grad1);

    s[iter % M] = x1 - x;
    y[iter % M] = grad1 - grad;
    z[iter % M] = 1.0 / DotProduct(y[iter % M], s[iter % M]);
    x = x1;
    grad = grad1;
  }

  return x.STLVector();
}

double MaxEnt::BacktrackingLineSearch(const DoubleVector& x0,
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
