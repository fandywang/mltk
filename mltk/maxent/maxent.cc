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
  me_instances_.clear();
  for (std::vector<Instance>::const_iterator citer = instances.begin();
       citer != instances.end(); ++citer) {
    AddInstance(*citer);
  }

  return Train();
}

void MaxEnt::AddInstance(const Instance& instance) {
  me_instances_.push_back(MaxEntInstance());
  MaxEntInstance& me_instance = me_instances_.back();

  me_instance.label = label_vocab_.Put(instance.label());
  if (me_instance.label > Feature::MAX_LABEL_TYPES) {
    std::cerr << "error: too many types of labels." << std::endl;
    exit(1);
  }

  const std::vector<std::pair<std::string, double> >& features
      = instance.GetFeatures();
  for (std::vector<std::pair<std::string, double> >::const_iterator citer
       = features.begin();
       citer != features.end();
       ++citer) {
    me_instance.features.push_back(
        std::pair<int32_t, double>(featurename_vocab_.Put(citer->first),
                                   citer->second));
  }
}

bool MaxEnt::Train() {
  if (l1reg_ > 0 && l2reg_ > 0) {
    std::cerr << "error: L1 and L2 regularizers cannot be used simultaneously."
         << std::endl;
    return false;
  }

  if (me_instances_.size() == 0) {
    std::cerr << "error: no training data." << std::endl;
    return false;
  }

  if (num_heldout_ >= static_cast<int32_t>(me_instances_.size())) {
    std::cerr << "error: too much heldout data. no training data is available."
        << std::endl;
    return false;
  }
  for (int32_t i = 0; i < num_heldout_; ++i) {
    heldout_.push_back(me_instances_.back());
    me_instances_.pop_back();
  }

  if (feature_freq_threshold_ > 0) {
    std::cerr << "the threshold of feature frequency = "
      << feature_freq_threshold_ << std::endl;
  }

  if (l1reg_ > 0) { std::cerr << "L1 regularizer = " << l1reg_ << std::endl; }
  if (l2reg_ > 0) { std::cerr << "L2 regularizer = " << l2reg_ << std::endl; }

  // normalize
  l1reg_ /= me_instances_.size();
  l2reg_ /= me_instances_.size();

  std::cerr << "preparing for estimation...";
  InitFeatureVocabulary();
  InitAllMEFeatures();
  std::cerr << "done" << std::endl;

  std::cerr << "number of classes = " << label_vocab_.Size() << std::endl;
  std::cerr << "number of instances = " << me_instances_.size() << std::endl;
  std::cerr << "number of features = " << feature_vocab_.Size() << std::endl;

  // calc E_p1 (f), p1(x, y) = count(x, y) / N
  std::cerr << "calculating empirical expectation...";
  empirical_expectation_.resize(feature_vocab_.Size());
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    empirical_expectation_[i] = 0;
  }
  for (int32_t n = 0; n < static_cast<int32_t>(me_instances_.size()); ++n) {
    const MaxEntInstance* me_instance = &me_instances_[n];
    for (std::vector<std::pair<int32_t, double> >::const_iterator citer
         = me_instance->features.begin();
         citer != me_instance->features.end();
         ++citer) {
      for (std::vector<int32_t>::const_iterator k
           = all_me_features_[citer->first].begin();
           k != all_me_features_[citer->first].end();
           ++k) {
        if (feature_vocab_.GetFeature(*k).LabelId() == me_instance->label) {
          empirical_expectation_[*k] += citer->second;
          break;
        }
      }
    }
  }
  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    empirical_expectation_[i] /= me_instances_.size();
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
  MaxEntInstance me_instance;
  const std::vector<std::pair<std::string, double> >& features
      = instance->GetFeatures();
  for (std::vector<std::pair<std::string, double> >::const_iterator citer
       = features.begin();
       citer != features.end();
       ++citer) {
    int32_t id = featurename_vocab_.Id(citer->first);
    if (id >= 0) {  // only using the feature exists in training data
      me_instance.features.push_back(
          std::pair<int32_t, double>(id, citer->second));
    }
  }

  std::vector<double> prob_dist(label_vocab_.Size());
  int32_t label = Classify(me_instance, &prob_dist);
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
  me_instances_.clear();
  heldout_.clear();
}

void MaxEnt::InitFeatureVocabulary() {
  // count the occurrences of features
  typedef std::map<uint32_t, int32_t> FeatureCountType;
  FeatureCountType feature_count;

  if (feature_freq_threshold_ > 0) {
    for (std::vector<MaxEntInstance>::const_iterator citer1
         = me_instances_.begin();
         citer1 != me_instances_.end();
         ++citer1) {
      for (std::vector<std::pair<int32_t, double> >::const_iterator citer2
           = citer1->features.begin();
           citer2 != citer1->features.end();
           ++citer2) {
        feature_count[Feature(citer1->label, citer2->first).Body()]++;
      }
    }
  }

  for (std::vector<MaxEntInstance>::const_iterator citer1
       = me_instances_.begin();
       citer1 != me_instances_.end();
       ++citer1) {
    for (std::vector<std::pair<int32_t, double> >::const_iterator citer2
         = citer1->features.begin();
         citer2 != citer1->features.end();
         ++citer2) {
      const Feature feature(citer1->label, citer2->first);
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

int32_t MaxEnt::PerformQuasiNewton() {
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

  return 0;
}

double MaxEnt::FunctionGradient(const std::vector<double>& x,
                                std::vector<double>* grad) {
  assert(static_cast<size_t>(feature_vocab_.Size()) == x.size());

  for (size_t i = 0; i < x.size(); ++i) { lambdas_[i] = x[i]; }

  double score = UpdateModelExpectation();

  // update gradient
  if (l2reg_ == 0) {
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = -(empirical_expectation_[i] - model_expectation_[i]);
    }
  } else {
    const double c = l2reg_ * 2;
    for (size_t i = 0; i < x.size(); ++i) {
      (*grad)[i] = -(empirical_expectation_[i] - model_expectation_[i]
                    - c * lambdas_[i]);
    }
  }

  return -score;
}

double MaxEnt::UpdateModelExpectation() {
  double logl = 0;
  int32_t ncorrect = 0;

  model_expectation_.resize(feature_vocab_.Size());
  for (int i = 0; i < feature_vocab_.Size(); ++i) { model_expectation_[i] = 0; }

  for (std::vector<MaxEntInstance>::const_iterator citer
       = me_instances_.begin();
       citer != me_instances_.end();
       ++citer) {
    std::vector<double> prob_dist(label_vocab_.Size());
    int32_t max_label = CalcConditionalProbability(*citer, &prob_dist);

    logl += log(prob_dist[citer->label]);
    if (max_label == citer->label) { ++ncorrect; }

    // model_expectation
    for (std::vector<std::pair<int32_t, double> >::const_iterator j
         = citer->features.begin();
         j != citer->features.end();
         ++j) {
      for (std::vector<int32_t>::const_iterator k
           = all_me_features_[j->first].begin();
           k != all_me_features_[j->first].end();
           ++k) {
        model_expectation_[*k]
            += prob_dist[feature_vocab_.GetFeature(*k).LabelId()] * j->second;
      }
    }
  }

  for (int32_t i = 0; i < feature_vocab_.Size(); ++i) {
    model_expectation_[i] /= me_instances_.size();
    if (l2reg_ > 0) { logl -= lambdas_[i] * lambdas_[i] * l2reg_; }
  }

  train_error_ = 1 - static_cast<double>(ncorrect) / me_instances_.size();
  logl /= me_instances_.size();

  return logl;
}

double MaxEnt::CalcHeldoutLikelihood() {
  double logl = 0;
  int32_t ncorrect = 0;

  for (std::vector<MaxEntInstance>::const_iterator citer = heldout_.begin();
       citer != heldout_.end();
       ++citer) {
    std::vector<double> prob_dist(label_vocab_.Size());
    int32_t label = Classify(*citer, &prob_dist);
    logl += log(prob_dist[citer->label]);
    if (label == citer->label) { ++ncorrect; }
  }

  heldout_error_ = 1 - static_cast<double>(ncorrect) / heldout_.size();

  return logl /= heldout_.size();
}

// p(y | x)
int32_t MaxEnt::Classify(const MaxEntInstance& me_instance,
                         std::vector<double>* prob_dist) const {
  assert(label_vocab_.Size() == static_cast<int32_t>(prob_dist->size()));

  CalcConditionalProbability(me_instance, prob_dist);

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
    const MaxEntInstance& me_instance, std::vector<double>* prob_dist) const {
  std::vector<double> powv(label_vocab_.Size(), 0.0);

  for (std::vector<std::pair<int32_t, double> >::const_iterator citer
       = me_instance.features.begin();
       citer != me_instance.features.end();
       ++citer) {
    for (std::vector<int32_t>::const_iterator k
         = all_me_features_[citer->first].begin();
         k != all_me_features_[citer->first].end();
         ++k) {
      powv[feature_vocab_.GetFeature(*k).LabelId()]
          += lambdas_[*k] * citer->second;
    }
  }

  std::vector<double>::const_iterator pmax
      = max_element(powv.begin(), powv.end());
  double sum = 0.0;
  double offset = std::max(0.0, *pmax - 700);  // to avoid overflow
  for (int32_t label = 0; label < label_vocab_.Size(); ++label) {
    double pow_value = powv[label] - offset;
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
