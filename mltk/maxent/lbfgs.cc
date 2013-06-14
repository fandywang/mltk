// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
#include "mltk/maxent/maxent.h"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "mltk/maxent/double_vector.h"

namespace mltk {
namespace maxent {

const static int32_t M = 10;
const static double LINE_SEARCH_ALPHA = 0.1;
const static double LINE_SEARCH_BETA = 0.5;

// stopping criteria
int32_t LBFGS_MAX_ITER = 300;
const static double MIN_GRAD_NORM = 0.0001;

//
// Jorge Nocedal, "Updating Quasi-Newton Matrices With Limited Storage",
// Mathematics of Computation, Vol. 35, No. 151, pp. 773-782, 1980.
//
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
    const int j = (i + offset) % M;
    alpha[i] = z[j] * DotProduct(s[j], q);
    q += -alpha[i] * y[j];
  }
  if (iter > 0) {
    const int32_t j = (iter - 1) % M;
    const double gamma = ((1.0 / z[j]) / DotProduct(y[j], y[j]));
    q *= gamma;
  }
  for (int32_t i = 0; i <= bound - 1; ++i) {
    const int j = (i + offset) % M;
    beta[i] = z[j] * DotProduct(y[j], q);
    q += s[j] * (alpha[i] - beta[i]);
  }

  return q;
}

std::vector<double> MaxEnt::PerformLBFGS(const std::vector<double>& x0) {
  const size_t dim = x0.size();
  DoubleVector x(x0);

  DoubleVector grad(dim), dx(dim);
  double f = FunctionGradient(x.STLVector(), grad.STLVector());

  DoubleVector s[M];
  DoubleVector y[M];
  double z[M];  // rho

  for (int32_t iter = 0; iter < LBFGS_MAX_ITER; ++iter) {  // 终止条件 1
    fprintf(stderr, "%3d  obj(err) = %f (%6.4f)", iter + 1, -f, train_error_);
    if (num_heldout_ > 0) {
      const double heldout_logl = CalcHeldoutLikelihood();
      fprintf(stderr, "  heldout_logl(err) = %f (%6.4f)",
              heldout_logl, heldout_error_);
    }
    fprintf(stderr, "\n");

    // 终止条件 2
    if (sqrt(DotProduct(grad, grad)) < MIN_GRAD_NORM) break;

    dx = -1 * ApproximateHg(iter, grad, s, y, z);

    DoubleVector x1(dim), grad1(dim);
    f = BacktrackingLineSearch(x, grad, f, dx, x1, grad1);

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
                                      DoubleVector& x,
                                      DoubleVector& grad1) {
  double t = 1.0 / LINE_SEARCH_BETA;
  double f;

  do {
    t *= LINE_SEARCH_BETA;
    x = x0 + t * dx;
    f = FunctionGradient(x.STLVector(), grad1.STLVector());
  } while (f > f0 + LINE_SEARCH_ALPHA * t * DotProduct(dx, grad0));

  return f;
}

}  // namespace maxent
}  // namespace mltk
