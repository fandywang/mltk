// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "mltk/common/double_vector.h"
#include "mltk/common/feature_vocabulary.h"
#include "mltk/common/feature.h"
#include "mltk/common/instance.h"
#include "mltk/common/vocabulary.h"

namespace mltk {
namespace maxent {

using mltk::common::Feature;
using mltk::common::Instance;
using mltk::common::MemInstance;

bool MaxEnt::LoadModel(const std::string& filename) {
  Clear();

  FILE* fp = fopen(filename.c_str(), "r");
  if (!fp) {
    std::cerr << "error: cannot open " << filename << "!" << std::endl;
    return false;
  }

  char buf[1024];
  while(fgets(buf, 1024, fp)) {
    std::string line(buf);
    std::string::size_type t1 = line.find_first_of('\t');
    std::string::size_type t2 = line.find_last_of('\t');
    std::string label_name = line.substr(0, t1);
    std::string feature_name = line.substr(t1 + 1, t2 - (t1 + 1));
    float lambda;
    std::string w = line.substr(t2 + 1);
    sscanf(w.c_str(), "%f", &lambda);

    int32_t label = label_vocab_.Put(label_name);
    int32_t feature = featurename_vocab_.Put(feature_name);
    feature_vocab_.Put(Feature(label, feature));
    lambdas_.push_back(lambda);
  }
  fclose(fp);

  InitAllMEFeatures();

  return true;
}

bool MaxEnt::SaveModel(const std::string& filename) const {
  FILE* fp = fopen(filename.c_str(), "w");
  if (!fp) {
    std::cerr << "error: cannot open " << filename << "!" << std::endl;
    return false;
  }

  for (common::StringMapType::const_iterator iter = featurename_vocab_.begin();
       iter != featurename_vocab_.end();
       ++iter) {
    for (int32_t label_id = 0; label_id < label_vocab_.Size(); ++label_id) {
      std::string label = label_vocab_.Str(label_id);
      int32_t id = feature_vocab_.FeatureId(Feature(label_id, iter->second));
      if (id < 0) continue;
      if (lambdas_[id] == 0) continue;  // ignore zero-weight features

      fprintf(fp, "%s\t%s\t%f\n",
              label.c_str(), iter->first.c_str(), lambdas_[id]);
    }
  }
  fclose(fp);

  return true;
}

bool MaxEnt::Train(const std::vector<Instance>& instances) {
  mem_instances_.clear();
  for (std::vector<Instance>::const_iterator citer = instances.begin();
       citer != instances.end(); ++citer) {
    AddInstance(*citer);
  }

  return Train();
}

void MaxEnt::AddInstance(const Instance& instance) {
  mem_instances_.push_back(MemInstance());
  MemInstance& mem_instance = mem_instances_.back();

  mem_instance.set_label(label_vocab_.Put(instance.label()));
  if (mem_instance.label() > Feature::MAX_LABEL_TYPES) {
    std::cerr << "error: too many types of labels." << std::endl;
    exit(1);
  }

  for (Instance::ConstIterator citer(instance); !citer.Done(); citer.Next()) {
    mem_instance.AddFeature(featurename_vocab_.Put(citer.FeatureName()),
                            citer.FeatureValue());
  }

  // mem_instance.AddFeature(featurename_vocab_.Put("BIAS"), 1.0);
}

bool MaxEnt::Train() {
  if (l1reg_ > 0 && l2reg_ > 0) {
    std::cerr << "error: L1 and L2 regularizers cannot be used simultaneously."
         << std::endl;
    return false;
  }

  if (mem_instances_.size() == 0) {
    std::cerr << "error: no training data." << std::endl;
    return false;
  }

  if (num_heldout_ >= static_cast<int32_t>(mem_instances_.size())) {
    std::cerr << "error: too much heldout data. no training data is available."
        << std::endl;
    return false;
  }
  for (int32_t i = 0; i < num_heldout_; ++i) {
    heldout_.push_back(mem_instances_.back());
    mem_instances_.pop_back();
  }

  if (feature_freq_threshold_ > 0) {
    std::cerr << "the threshold of feature frequency = "
        << feature_freq_threshold_ << std::endl;
  }

  if (l1reg_ > 0) { std::cerr << "L1 regularizer = " << l1reg_ << std::endl; }
  if (l2reg_ > 0) { std::cerr << "L2 regularizer = " << l2reg_ << std::endl; }

  // normalize
  l1reg_ /= mem_instances_.size();
  l2reg_ /= mem_instances_.size();

  std::cerr << "preparing for estimation...";
  InitFeatureVocabulary();
  InitAllMEFeatures();
  std::cerr << "done" << std::endl;

  std::cerr << "number of classes = " << label_vocab_.Size() << std::endl;
  std::cerr << "number of instances = " << mem_instances_.size() << std::endl;
  std::cerr << "number of features = " << feature_vocab_.Size() << std::endl;

  // calc E_p1 (f), p1(x, y) = count(x, y) / N
  std::cerr << "calculating empirical expectation...";
  empirical_expectation_.resize(feature_vocab_.Size());
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    empirical_expectation_[i] = 0;
  }
  for (int32_t n = 0; n < static_cast<int32_t>(mem_instances_.size()); ++n) {
    for (MemInstance::ConstIterator citer(mem_instances_[n]);
         !citer.Done(); citer.Next()) {
      for (size_t i = 0; i < all_me_features_[citer.FeatureId()].size(); ++i) {
        const int32_t id = all_me_features_[citer.FeatureId()][i];
        if (feature_vocab_.GetFeature(id).LabelId() == citer.LabelId()) {
          empirical_expectation_[id] += citer.FeatureValue();
          break;
        }
      }
    }
  }
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    empirical_expectation_[i] /= mem_instances_.size();
  }
  std::cerr << "done" << std::endl;

  // initialize model
  lambdas_.resize(feature_vocab_.Size());
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) { lambdas_[i] = 0.0; }

  // parameter estimation
  if (optimization_method_ == SGD) {
    PerformSGD();
  } else {
    PerformQuasiNewton();
  }

  // count the number of active features
  int32_t num_active = 0;
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    if (lambdas_[i] != 0) { ++num_active; }
  }
  std::cerr << "number of active features = " << num_active << std::endl;

  return true;
}

std::vector<double> MaxEnt::Classify(Instance* instance) const {
  MemInstance mem_instance;
  for (Instance::ConstIterator citer(*instance); !citer.Done(); citer.Next()) {
    int32_t id = featurename_vocab_.Id(citer.FeatureName());
    if (id >= 0) {  // only using the feature exists in training data
      mem_instance.AddFeature(id, citer.FeatureValue());
    }
  }

  std::vector<double> prob_dist(label_vocab_.Size());
  int32_t label = Classify(mem_instance, &prob_dist);
  instance->set_label(GetClassLabel(label));

  return prob_dist;
}

void MaxEnt::Clear() {
  label_vocab_.Clear();
  featurename_vocab_.Clear();
  feature_vocab_.Clear();
  lambdas_.clear();
  all_me_features_.clear();
  empirical_expectation_.clear();
  model_expectation_.clear();
  mem_instances_.clear();
  heldout_.clear();
}

void MaxEnt::InitFeatureVocabulary() {
  // count the occurrences of features
  typedef std::map<uint32_t, int32_t> FeatureCountType;
  FeatureCountType feature_count;

  if (feature_freq_threshold_ > 0) {
    for (size_t n = 0; n < mem_instances_.size(); ++n) {
      for (MemInstance::ConstIterator citer(mem_instances_[n]);
           !citer.Done(); citer.Next()) {
        feature_count[Feature(citer.LabelId(), citer.FeatureId()).Body()]++;
      }
    }
  }

  for (size_t n = 0; n < mem_instances_.size(); ++n) {
    for (MemInstance::ConstIterator citer(mem_instances_[n]);
         !citer.Done(); citer.Next()) {
      const Feature feature(citer.LabelId(), citer.FeatureId());
      if (feature_freq_threshold_ > 0
          && feature_count[feature.Body()] <= feature_freq_threshold_) {
        continue;
      }
      feature_vocab_.Put(feature);
    }
  }
}

void MaxEnt::InitAllMEFeatures() {
  all_me_features_.clear();

  // feature = feature_name * class
  for (int32_t f = 0; f < featurename_vocab_.Size(); ++f) {
    all_me_features_.push_back(std::vector<int32_t>());
    std::vector<int32_t>& vi = all_me_features_.back();

    for (int32_t c = 0; c < label_vocab_.Size(); ++c) {
      int32_t id = feature_vocab_.FeatureId(Feature(c, f));
      if (id >= 0) {
        vi.push_back(id);
      }
    }
  }
}

void MaxEnt::PerformQuasiNewton() {
  const int32_t dim = feature_vocab_.Size();
  std::vector<double> x0(dim);
  for (int32_t i = 0; i < dim; ++i) { x0[i] = lambdas_[i]; }

  std::vector<double> x;
  if (l1reg_ > 0 || optimization_method_ == OWLQN) {
    // NOTE(l1reg_ > 0): The LBFGS limited-memory quasi-Newton method is the
    // algorithm of choice for optimizing the parameters of large-scale
    // log-linear models with L2-regularization, but it cannot be used for an
    // L1-regularized loss due to its non-diﬀerentiability whenever some
    // parameter is zero.
    std::cerr << "performing OWLQN" << std::endl;
    x = PerformOWLQN(x0, l1reg_);
  } else {
    std::cerr << "performing LBFGS" << std::endl;
    x = PerformLBFGS(x0);
  }

  for (int32_t i = 0; i < dim; ++i) { lambdas_[i] = x[i]; }
}

double MaxEnt::FunctionGradient(const std::vector<double>& x,
                                std::vector<double>* grad) {
  assert(static_cast<size_t>(feature_vocab_.Size()) == x.size());

  for (size_t i = 0; i < x.size(); ++i) { lambdas_[i] = x[i]; }

  double score = UpdateModelExpectation();

  // update gradient
  if (l2reg_ == 0) {
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = model_expectation_[i] - empirical_expectation_[i];
    }
  } else {
    const double c = l2reg_ * 2;
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = model_expectation_[i] - empirical_expectation_[i]
                   + c * lambdas_[i];
    }
  }

  return -score;
}

double MaxEnt::UpdateModelExpectation() {
  double logl = 0;
  int32_t ncorrect = 0;

  model_expectation_.resize(feature_vocab_.Size());
  for (int i = 0; i < feature_vocab_.Size(); ++i) { model_expectation_[i] = 0; }

  for (size_t n = 0; n < mem_instances_.size(); ++n) {
    std::vector<double> prob_dist(label_vocab_.Size());
    int32_t max_label = CalcConditionalProbability(mem_instances_[n],
                                                   &prob_dist);

    logl += log(prob_dist[mem_instances_[n].label()]);
    if (max_label == mem_instances_[n].label()) { ++ncorrect; }

    // model_expectation
    for (MemInstance::ConstIterator citer(mem_instances_[n]);
         !citer.Done(); citer.Next()) {
      for (size_t i = 0; i < all_me_features_[citer.FeatureId()].size(); ++i) {
        const int32_t id = all_me_features_[citer.FeatureId()][i];
        model_expectation_[id]
          += prob_dist[feature_vocab_.GetFeature(id).LabelId()]
             * citer.FeatureValue();
      }
    }
  }

  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    model_expectation_[i] /= mem_instances_.size();
    if (l2reg_ > 0) { logl -= lambdas_[i] * lambdas_[i] * l2reg_; }
  }

  train_accuracy_ = static_cast<double>(ncorrect) / mem_instances_.size();
  logl /= mem_instances_.size();

  return logl;
}

double MaxEnt::CalcHeldoutLikelihood() {
  double logl = 0;
  int32_t ncorrect = 0;

  for (std::vector<MemInstance>::const_iterator citer = heldout_.begin();
       citer != heldout_.end();
       ++citer) {
    std::vector<double> prob_dist(label_vocab_.Size());
    int32_t label = Classify(*citer, &prob_dist);
    logl += log(prob_dist[citer->label()]);
    if (label == citer->label()) { ++ncorrect; }
  }

  heldout_accuracy_ = static_cast<double>(ncorrect) / heldout_.size();

  return logl /= heldout_.size();
}

// p(y | x)
int32_t MaxEnt::Classify(const MemInstance& mem_instance,
                         std::vector<double>* prob_dist) const {
  assert(label_vocab_.Size() == static_cast<int32_t>(prob_dist->size()));

  CalcConditionalProbability(mem_instance, prob_dist);

  int32_t max_label = 0;
  double max_prob = 0.0;
  for (int32_t i = 0; i < static_cast<int32_t>(prob_dist->size()); ++i) {
    if ((*prob_dist)[i] > max_prob) {
      max_label = i;
      max_prob = (*prob_dist)[i];
    }
  }

  return max_label;
}

int32_t MaxEnt::CalcConditionalProbability(
    const MemInstance& mem_instance, std::vector<double>* prob_dist) const {
  std::vector<double> powv(label_vocab_.Size(), 0.0);

  for (MemInstance::ConstIterator citer(mem_instance);
       !citer.Done(); citer.Next()) {
    for (size_t i = 0; i < all_me_features_[citer.FeatureId()].size(); ++i) {
      const int32_t id = all_me_features_[citer.FeatureId()][i];
      powv[feature_vocab_.GetFeature(id).LabelId()]
          += lambdas_[id] * citer.FeatureValue();
    }
  }

  std::vector<double>::const_iterator pmax
      = max_element(powv.begin(), powv.end());
  // std::cerr << "pmax: " << *pmax << std::endl;
  double sum = 0.0;
  double offset = std::max(0.0, *pmax - 700);  // to avoid overflow
  for (int32_t label = 0; label < label_vocab_.Size(); ++label) {
    double pow_value = powv[label] - offset;
    // std::cerr << "powv : " << pow_value << ", label: " << label << std::endl;
    double prod = exp(pow_value);  // exp(w * x)
    assert(prod != 0);

    (*prob_dist)[label] = prod;
    sum += prod;
  }

  int32_t max_label = 0;
  if (sum > 0.0) {
    for (int32_t label = 0; label < label_vocab_.Size(); ++label) {
      (*prob_dist)[label] /= sum;
      if ((*prob_dist)[label] > (*prob_dist)[max_label]) { max_label = label; }
    }
  }
  assert(max_label >= 0);

  return max_label;
}

}  // namespace maxent
}  // namespace mltk
