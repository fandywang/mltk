// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/owlqn.h"

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

const static double LINE_SEARCH_ALPHA = 0.1;
const static double LINE_SEARCH_BETA = 0.5;

// stopping criteria
const static double MIN_GRAD_NORM = 0.0001;

inline int32_t Sign(double x) {
  if (x > 0) { return 1; }
  if (x < 0) { return -1; }
  return 0;
};

void OWLQN::EstimateParamater(const std::vector<Instance>& instances,
                              int32_t num_heldout,
                              ModelData* model_data) {
  // NOTE(l1reg_ > 0): The LBFGS limited-memory quasi-Newton method is the
  // algorithm of choice for optimizing the parameters of large-scale
  // log-linear models with L2-regularization, but it cannot be used for an
  // L1-regularized loss due to its non-diﬀerentiability whenever some
  // parameter is zero.
  std::cerr << "performing OWLQN" << std::endl;
  if (l2reg_ > 0) {
    std::cerr << "error: L2 regularization is not supported in OWLQN,"
        << "you can use LBFGS method instead." << std::endl;
    exit(1);
  }

  InitFromInstances(instances, num_heldout, model_data);

  std::vector<double> x = PerformOWLQN();
  model_data_->UpdateLambdas(x);
}

std::vector<double> OWLQN::PerformOWLQN() {
  const std::vector<double> lambdas = model_data_->Lambdas();
  assert(static_cast<int32_t>(lambdas.size()) == model_data_->NumFeatures());

  std::vector<double> x0(lambdas.size());
  for (int32_t i = 0; i < lambdas.size(); ++i) { x0[i] = lambdas[i]; }

  DoubleVector x(x0);
  DoubleVector grad(lambdas.size());
  double f = RegularizedFuncGrad(l1reg_, x, grad);

  DoubleVector* s = new DoubleVector[m_];
  DoubleVector* y = new DoubleVector[m_];
  double* z = new double[m_];  // rho

  for (int32_t iter = 0; iter < num_iter_; ++iter) {  // stopping criteria 1
    DoubleVector pg = PseudoGradient(x, grad, l1reg_);

    std::cerr << "iter = " << iter + 1
        << ", obj(err) = " << f
        << ", accuracy = " << train_accuracy_ << std::endl;
    if (heldout_data_.size() > 0) {
      const double heldout_logl = CalcHeldoutLikelihood();
      std::cerr << "\theldout_logl(err) = " << -1 * heldout_logl
          << ", accuracy = " << heldout_accuracy_ << std::endl;
    }

    // stopping criteria 2
    if (sqrt(DotProduct(pg, pg)) < MIN_GRAD_NORM) { break; }

    DoubleVector dx = -1 * ApproximateHg(iter, pg, s, y, z);
    if (DotProduct(dx, pg) >= 0) { dx.Project(-1 * pg); }

    DoubleVector x1(lambdas.size()), grad1(lambdas.size());
    f = ConstrainedLineSearch(l1reg_, x, pg, f, dx, x1, grad1);

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

double OWLQN::RegularizedFuncGrad(const double C,
                                  const DoubleVector& x,
                                  DoubleVector& grad) {
  double f = FunctionGradient(x.STLVector(), &(grad.STLVector()));
  for (size_t i = 0; i < x.Size(); i++) {
    f += C * fabs(x[i]);
  }
  return f;
}

DoubleVector OWLQN::PseudoGradient(const DoubleVector& x,
                                   const DoubleVector& grad0,
                                   const double C) {
  DoubleVector grad = grad0;
  for (size_t i = 0; i < x.Size(); i++) {
    if (x[i] != 0) {
      grad[i] += C * Sign(x[i]);
      continue;
    }
    const double gm = grad0[i] - C;
    if (gm > 0) {
      grad[i] = gm;
      continue;
    }
    const double gp = grad0[i] + C;
    if (gp < 0) {
      grad[i] = gp;
      continue;
    }
    grad[i] = 0;
  }

  return grad;
}

DoubleVector OWLQN::ApproximateHg(const int32_t iter,
                                  const DoubleVector& grad,
                                  const DoubleVector* s,
                                  const DoubleVector* y,
                                  const double* z) {
  int32_t offset, bound;
  if (iter <= m_) {
    offset = 0;
    bound = iter;
  }
  else {
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

double OWLQN::ConstrainedLineSearch(double C,
                                    const DoubleVector& x0,
                                    const DoubleVector& grad0,
                                    const double f0,
                                    const DoubleVector& dx,
                                    DoubleVector& x,
                                    DoubleVector& grad1) {
  // compute the orthant to explore
  DoubleVector orthant = x0;
  for (size_t i = 0; i < orthant.Size(); i++) {
    if (orthant[i] == 0) orthant[i] = -grad0[i];
  }

  double t = 1.0 / LINE_SEARCH_BETA;

  double f;
  do {
    t *= LINE_SEARCH_BETA;
    x = x0 + t * dx;
    x.Project(orthant);
    f = RegularizedFuncGrad(C, x, grad1);
  } while (f > f0 + LINE_SEARCH_ALPHA * DotProduct(x - x0, grad0));

  return f;
}

}  // namespace maxent
}  // namespace mltk
