// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// MaxEnt implements the maximum entropy model, also named multi-nominal
// logistic regression model, which is usually used in Natural Language
// Processing.

#ifndef MLTK_MAXENT_MAXENT_H_
#define MLTK_MAXENT_MAXENT_H_

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "mltk/maxent/double_vector.h"
#include "mltk/maxent/feature.h"
#include "mltk/maxent/instance.h"
#include "mltk/maxent/vocabulary.h"

namespace mltk {
namespace maxent {

typedef struct {
  // 类别 id
  int32_t label;

  // 特征向量
  std::vector<std::pair<int32_t, double> > features;

  // 先验分布, 比如人工指定, 或者在已有模型基础上支持增量训练.
  std::vector<double> ref_prob_dist;  // reference probability distribution
} MaxEntInstance;

// Features:
//   1. supporting real-valued features.
//   2. supporting parameter estimation algorithms, including LBFGS, OWLQN
//      and SGD.
//   3. supporting incremental learning.
//
// TODO(fandywang):
//   1. add unittest
//   2. add apps
//   3. supporting cross validation
class MaxEnt {
 public:
  MaxEnt() : optimization_method_(LBFGS), l1reg_(0), l2reg_(0) {}
  ~MaxEnt() { Clear(); }

  // Load model from file.
  //
  // Line format: class \t feature \t weight(lambda)
  bool LoadModel(const std::string& filename);

  // Save model to file.
  //
  // threshold: cut off low_weight features
  bool SaveModel(const std::string& filename, const double threshold = 0) const;

  // to support incremental learning.
  void SetReferenceModel(const MaxEnt& ref_model) { ref_model_ = &ref_model; }

  void UseLBFGS() { optimization_method_ = LBFGS; }
  void UseSGD() { optimization_method_ = SGD; }

  void UseL1Regularizer(const double reg) { l1reg_ = reg; }
  void UseL2Regularizer(const double reg) { l2reg_ = reg; }

  void SetHeldout(const int32_t num_heldout) { num_heldout_ = num_heldout; }

  // Training
  void AddInstance(const Instance& instance);
  int32_t Train();

  int32_t Train(const std::vector<Instance>& instances);

  // Classify
  std::vector<double> Classify(Instance* instance) const;

  int32_t NumClasses() const { return num_classes_; }

  std::string GetClassLabel(int32_t id) const { return label_vocab_.Str(id); }

  int32_t GetClassId(const std::string& label) const {
    return label_vocab_.Id(label);
  }

 private:
  void Clear();

  // 为样本 me_instance 设置先验分布结果, 先验分布根据先验模型
  // reference_distribution 预估得到.
  void SetRefProbDist(MaxEntInstance* me_instance) const;

  // 特征选择: 过滤掉出现次数少于 cutoff 的特征.
  void InitFeatureVocabulary(const int cutoff);

  // 枚举所有可能的 feature, class_name * feature_name
  void InitAllMEFeatures();

  // 参数估计: 拟牛顿法
  int32_t PerformQuasiNewton();
  // 参数估计: 梯度下降法
  int32_t PerformSGD();

  std::vector<double> PerformLBFGS(const std::vector<double>& x0);
  std::vector<double> PerformOWLQN(const std::vector<double>& x0,
                                   const double C);

  double FunctionGradient(const std::vector<double>& x,
                          std::vector<double>& grad);

  // 更新 E_p (f)
  // E_p (f) = sum_x,y P1(x)P(y|x)f(x, y)
  double UpdateModelExpectation();

  // 线性搜索
  double BacktrackingLineSearch(const DoubleVector& x0,
                                const DoubleVector& grad0,
                                const double f0,
                                const DoubleVector& dx,
                                DoubleVector& x,
                                DoubleVector& grad1);
  double RegularizedFuncGrad(const double C,
                             const DoubleVector& x,
                             DoubleVector& grad);
  double ConstrainedLineSearch(double C,
                               const DoubleVector& x0,
                               const DoubleVector& grad0,
                               const double f0,
                               const DoubleVector& dx,
                               DoubleVector& x,
                               DoubleVector& grad1);

  // 预估输入样本 me_instance 的类别分布.
  int32_t Classify(const MaxEntInstance& me_instance,
                   std::vector<double>* prob_dist) const;

  // 预估样本 me_instance 的条件概率分布 prob_dist: p(y|x)
  int32_t CalcConditionalProbability(const MaxEntInstance& me_instance,
                                     std::vector<double>* prob_dist) const;

  double CalcHeldoutLikelihood();

 private:
  int32_t num_classes_;

  std::vector<MaxEntInstance> me_instances_;  // vector of training samples

  Vocabulary featurename_vocab_;  // x:feature_name <--> id

  Vocabulary label_vocab_;  // y:class label <--> id

  FeatureVocabulary feature_vocab_;  // f(x, y)

  std::vector<double> lambdas_;  // vector of lambda, 即 f(x, y) 对应的 weight

  const MaxEnt* ref_model_;  // reference model

  // 特征函数 f(x, y) 关于经验分布 P1(X, Y) 的期望值, 用 E_p1 (f) 表示.
  //
  // E_p1 (f) = sum_x,y P1(x, y)f(x, y)
  std::vector<double> empirical_expectation_;

  // 特征函数 f(x, y) 关于模型 P(Y|X) 与经验分布 P1(X) 的期望值, 用 E_p (f) 表示.
  //
  // E_p (f) = sum_x,y P1(x)P(y|x)f(x, y)
  std::vector<double> model_expectation_;

  // 所有可能的 Feature f(x, y), 格式为:
  // [feaure_name_id, [class1_feature_id, class2_feature_id, ...]]
  std::vector<std::vector<int> > all_me_features_;

  // heldout data
  int32_t num_heldout_;
  std::vector<MaxEntInstance> heldout_;

  // Note: OWLQN and SGD are available only for L1-regularization
  enum OPTIMIZATION_METHOD { LBFGS, OWLQN, SGD } optimization_method_;

  // 正则化因子
  double l1reg_;
  double l2reg_;

  double train_error_;  // current error rate on the training data
  double heldout_error_;  // current error rate on the heldout data
};

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_MAXENT_H_
