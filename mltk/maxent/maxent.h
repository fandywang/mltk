// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// MaxEnt implements the maximum entropy model, also named multi-nominal
// logistic regression model, which is usually used in Natural Language
// Processing.
//
// Specifically, MaxEnt is mainly reconstructed and optimizated based on
// 'http://www-tsujii.is.s.u-tokyo.ac.jp/~tsuruoka/maxent/', yet anothor simple
// C++ library for maximum entropy classification.
//
// Maximum Entropy Principle:
//
//    To select a model from a set C of allowed probability distributions,
//    choose the model p^\star \in C with maximum entropy H(p):
//
//         p^\star = argmax_{p \in C} H(p)
//
//    Where H(p) = - \sum_{x,y} {p1(x) * p(y|x) log p(y|x)}
//          C = {p \in P | E_p(f_i) = E_p1(f_i), i = 1,2,...,n}, constraint.
//          p1(f) = \sum_{x,y} {p1(x,y) * f(x,y)}
//          p(f) = \sum_{x,y} {p1(x) * p(y|x) * f(x,y)}, f is a feature function
//
//    Obviously, the maximum entropy principle presents us with a problem in
//    constrained optimization.
//
// Maximum Likelihood:
//
//    Refer to 'Adam L. Berger, Stephen A.Della Pietra, Vincent J. Della Pietra.
//    1996. A Maximum Entropy Approach to Natural Language Processing. ACL',
//    we know that the model p^\star \in C with maximum entropy is the model
//    in the parametric family p_\lambda (y|x) that maximizes the likelihood of
//    the training sample p1.
//
//    The log-likelihood L_p1 (p) of the empirical distribution p1(x,y) as
//    predicted by a model p(y|x) is defined by
//
//       L_p1 (p) = log {\prod_{x,y} {p(y|x)^p1(x,y)}}
//                = \sum_{x,y} {p1(x,y) * log p(y|x)}
//                = \sum_{x,y} {p1(x,y) * \sum_i {\lambda_i * f_i (x,y)}}
//                  - \sum_x,y {p1(x,y) * log Z_\lambda(x)}
//                = \sum_x,y {p1(x,y) *\ sum_i {\lambda_i * f_i(x,y)}}
//                  - \sum_x {p1(x) * log Z_\lambda(x)}
//
//    Where p(y|x) = exp (\sum_i {\lambda_i * f_i(x,y)}) / Z_\lambda (x)
//          Z_\lambda (x) = \sum_y (exp (\sum_i {\lambda_i * f_i(x,y)}))
//
//    The most important practical consequence of this result is that any
//    algorithm for finding the maximum \lambda^\star of L_p1 (p) can be used
//    to find the maximum p^\star of H(p) for p \in C.
//
//    Finally, we pose the unconstrained optimization problem:
//
//         Find \lambda^\star = argmax_\lambda L_p1(p)
//                            = argmin_\lambda -L_p1(p)
//
// Parameter Estimation:
//
//    For all but the most simple problems, the \lambda^\star that maximize
//    L_p1(p) cannot be found analytically. Instead, we must resort to mumerical
//    methods, like IIS, GIS, GD, SGD, Newton's Methods, Quasi-Newton Methods,
//    etc. So far, MaxEnt implements three fast and effective algorithms,
//    LBFGS, OWLQN and SGD.
//
//    Refer to:
//      Jorge Nocedal. 1980. Updating Quasi-Newton Matrices with Limited
//      Storage, Mathematics of Computation.
//
//      Galen Andrew and Jianfeng Gao. 2007. Scalable training of L1-regularized
//      log-linear models, In Proceedings of ICML.
//
//      Yoshimasa Tsuruoka, Jun'ichi Tsujii, and Sophia Ananiadou. 2009.
//      Stochastic Gradient Descent Training for L1-regularized Log-linear
//      Models with Cumulative Penalty, In Proceedings of ACL-IJCNLP.
//
// Regularization:
//
//    Log-linear models are used in a variety of forms in maching learning,
//    and the parameters of such models are typically trained to minimize an
//    objective function
//
//        f(\lambda) = l(\lambda) + r(\lambda)
//
//    where l is the negative log-probability of a labelled training samples
//    according to the model, and r is a regularization term that favors
//    simpler models. It is well-known that the use of regularization is
//    necessary to achieve a model that generalizes well to unseen data,
//    particularly if the number of parameters is very high relative to the
//    amount of training data.
//
//    We focus on two kinds of commonly used regularizations, L1-Regularization
//    and L2-Regularization. They defined by
//
//       L1-Reg: r(\lambda) = a * ||\lambda||_1 = a * sum_i |\lambda_i|, a > 0
//       L2-Reg: r(\lambda) = a * ||\lambda||_2 = a * sum_i \lambda_i^2, a > 0
//
// Features:
//    1. supporting real-valued features.
//    2. supporting three fast and effective parameter estimation algorithms,
//       including LBFGS, OWLQN and SGD.
//
// TODO:
//    1. add apps, like [hierarchial] text classification and part-of-speech
//       tagging (POS).

#ifndef MLTK_MAXENT_MAXENT_H_
#define MLTK_MAXENT_MAXENT_H_

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "mltk/common/feature_vocabulary.h"
#include "mltk/common/vocabulary.h"

namespace mltk {

namespace common {
class DoubleVector;
class Feature;
class Instance;
}  // namespace common

namespace maxent {

typedef struct {
  int32_t label;  // class id

  std::vector<std::pair<int32_t, double> > features;  // vector of features
} MaxEntInstance;

class MaxEnt {
 public:
  MaxEnt() : num_heldout_(0), feature_freq_threshold_(0),
     optimization_method_(LBFGS), l1reg_(0), l2reg_(0) {}
  ~MaxEnt() { Clear(); }

  // Load model from file.
  //
  // Line format: class \t feature \t weight(lambda)
  bool LoadModel(const std::string& filename);

  // Save model to file.
  bool SaveModel(const std::string& filename) const;

  int32_t NumClasses() const { return label_vocab_.Size(); }

  std::string GetClassLabel(int32_t id) const { return label_vocab_.Str(id); }

  int32_t GetClassId(const std::string& label) const {
    return label_vocab_.Id(label);
  }

  // Training
  void AddInstance(const mltk::common::Instance& instance);
  bool Train();

  bool Train(const std::vector<mltk::common::Instance>& instances);

  void SetNumHeldout(const int32_t num_heldout) { num_heldout_ = num_heldout; }

  void SetFeatureFreqThreshold(const int32_t freq_threshold) {
      feature_freq_threshold_ = freq_threshold;
  }

  void UseLBFGS() { optimization_method_ = LBFGS; }
  void UseOWLQN() { optimization_method_ = OWLQN; }
  void UseSGD() { optimization_method_ = SGD; }

  void UseL1Reg(const double reg) { l1reg_ = reg; }
  void UseL2Reg(const double reg) { l2reg_ = reg; }

  // Classify
  std::vector<double> Classify(mltk::common::Instance* instance) const;

 private:
  void Clear();

  void InitFeatureVocabulary();

  // enumerate all possible features, class_name * feature_name
  void InitAllMEFeatures();

  // parameter estimation: Quasi-Newton's Methods, including LBFGS and OWLQN.
  int32_t PerformQuasiNewton();
  // parameter estimation: Stochastic gradient descent (SGD)
  int32_t PerformSGD();

  double FunctionGradient(const std::vector<double>& x,
                          std::vector<double>* grad);

  std::vector<double> PerformLBFGS(const std::vector<double>& x0);
  double BacktrackingLineSearch(const mltk::common::DoubleVector& x0,
                                const mltk::common::DoubleVector& grad0,
                                const double f0,
                                const mltk::common::DoubleVector& dx,
                                mltk::common::DoubleVector* x,
                                mltk::common::DoubleVector* grad1);

  // update E_p (f), formula: E_p (f) = sum_x,y P1(x)P(y|x)f(x, y)
  double UpdateModelExpectation();

  std::vector<double> PerformOWLQN(const std::vector<double>& x0,
                                   double C);
  double RegularizedFuncGrad(const double C,
                             const mltk::common::DoubleVector& x,
                             mltk::common::DoubleVector& grad);
  double ConstrainedLineSearch(double C,
                               const mltk::common::DoubleVector& x0,
                               const mltk::common::DoubleVector& grad0,
                               const double f0,
                               const mltk::common::DoubleVector& dx,
                               mltk::common::DoubleVector& x,
                               mltk::common::DoubleVector& grad1);

  int32_t Classify(const MaxEntInstance& me_instance,
                   std::vector<double>* prob_dist) const;

  // calculate p(y|x)
  int32_t CalcConditionalProbability(const MaxEntInstance& me_instance,
                                     std::vector<double>* prob_dist) const;

  double CalcHeldoutLikelihood();

 private:
  std::vector<MaxEntInstance> me_instances_;  // training data
  double train_error_;  // current error rate on the training data

  int32_t num_heldout_;
  std::vector<MaxEntInstance> heldout_;  // heldout data
  double heldout_error_;  // current error rate on the heldout data

  mltk::common::Vocabulary featurename_vocab_;  // featurename mapping, {x : id}
  mltk::common::Vocabulary label_vocab_;  // labelname mapping, {y : id}
  mltk::common::FeatureVocabulary feature_vocab_;  // vocabulary of features,
                                                   // f(x, y)

  int32_t feature_freq_threshold_;  // the threshold of feature frequency, which
                                    // is used as a simple feature selection
                                    // strategy.

  std::vector<double> lambdas_;  // vector of lambda, weight for feature f(x, y)

  // all possible features f(x, y), format:
  // [featurename_id, [feature_id1, feature_id2, ...]]
  std::vector<std::vector<int> > all_me_features_;

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

  // Note: OWLQN and SGD are available only for L1-regularization
  enum OPTIMIZATION_METHOD { LBFGS, OWLQN, SGD } optimization_method_;
  double l1reg_;  // L1-regularization
  double l2reg_;  // L2-regularization
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_MAXENT_H_
