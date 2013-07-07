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
//    Where H(p) = - \sum_{x,y} {p1(x) * p(y|x) log p(y|x)},
//          ( H(p) = H(Y|X), H(XY) = H(X|Y) + H(X) )
//          C = {p \in P | E_p(f_i) = E_p1(f_i), i = 1,2,...,n}, constraint.
//          E_p1(f) = \sum_{x,y} {p1(x,y) * f(x,y)}, is the probability of
//          feature f(x, y) in the training data.
//          E_p(f) = \sum_{x,y} {p1(x) * p(y|x) * f(x,y)}, is the probability in
//          the model.
//
//          p1(x,y) is the probability of (x, y) in the training data.
//          p1(x) is the probability of (x) in the training data.
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
//        f(\lambda) = l(\lambda) - r(\lambda)
//
//    where l is the negative log-probability of a labelled training samples
//    according to the model, and r is a regularization term that favors
//    simpler models. It is well-known that the use of regularization is
//    necessary to achieve a model that generalizes well to unseen data,
//    particularly if the number of parameters is very high relative to the
//    amount of training data.
//
//    We focus on the two most common regularization methods, called
//    L1 and L2 regularization. They defined by
//
//       L1-Reg: r(\lambda) = a * ||\lambda||_1 = a * sum_i |\lambda_i|, a > 0
//       L2-Reg: r(\lambda) = a * ||\lambda||_2 = a * sum_i \lambda_i^2, a > 0
//
// Data-Format:
//
//    MLTK.MaxEnt uses the popular sparse data format.
//
//       <class-label>\t<feature-key>:<feature-value>\t...\t<feature-key>:<feature-value>\n
//
//    The feature-key's are expected to be in numeric or string value, and the
//    class label for test data is required but not used; it's okay to put in a
//    dummy placeholder value such as 0 for test data.
//
// Features:
//    1. supporting real-valued features.
//    2. supporting three fast and effective parameter estimation algorithms,
//       including LBFGS, OWLQN and SGD.
//    3. supporting text-format feature name.
//
// TODO:
//    1. add apps, like [hierarchial] text classification and part-of-speech
//       tagging (POS).

#ifndef MLTK_MAXENT_MAXENT_H_
#define MLTK_MAXENT_MAXENT_H_

#include <string>
#include <vector>

#include "mltk/common/mem_instance.h"
#include "mltk/common/model_data.h"

namespace mltk {

namespace common {
class DoubleVector;
class Feature;
class Instance;
}  // namespace common

namespace maxent {

class Optimizer;

class MaxEnt {
 public:
  MaxEnt() : optimizer_(NULL) {}
  explicit MaxEnt(Optimizer* optimizer) : optimizer_(optimizer) {}
  ~MaxEnt() {}

  // Load model from file.
  //
  // Line format: label_name \t feature_name \t weight(lambda)
  bool LoadModel(const std::string& filename);

  // Save model to file.
  bool SaveModel(const std::string& filename) const;

  int32_t NumClasses() const { return model_data_.NumClasses(); }

  const common::ModelData& GetModelData() const { return model_data_; }

  const std::string& GetClassLabel(int32_t label_id) const {
    return model_data_.Label(label_id);
  }

  int32_t GetClassId(const std::string& label) const {
    return model_data_.LabelId(label);
  }

  // Training
  bool Train(const std::vector<common::Instance>& instances,
             int32_t num_heldout = 0);

  // Predict
  std::vector<double> Predict(common::Instance* instance) const;

 private:
  Optimizer* optimizer_;  // the optimization algorithm

  common::ModelData model_data_;  // the maxent model
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_MAXENT_H_

