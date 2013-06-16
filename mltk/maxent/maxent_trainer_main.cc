// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/base/string/algorithm.h"
#include "mltk/common/instance.h"

DEFINE_string(train_data_file, "", "the filename of training data.");
DEFINE_string(model_file, "", "the filename of maxent model.");
DEFINE_string(optim_method, "LBFGS",
              "the optimization method, LBFGS, OWLQN, or SGD.");
DEFINE_double(l1_reg, 0.0, "the L1 regularization.");
DEFINE_double(l2_reg, 0.0, "the L2 regularization.");
DEFINE_int32(num_heldout, 0, "the number of heldout data.");
DEFINE_int32(feature_freq_threshold, 1, "the threshold of feature frequency.");

int main(int argc, char** argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Initialize MaxEnt.";
  mltk::maxent::MaxEnt maxent;
  if (FLAGS_optim_method == "LBFGS") {
    maxent.UseLBFGS();
  } else if (FLAGS_optim_method == "OWLQN") {
    maxent.UseOWLQN();
  } else if (FLAGS_optim_method == "SGD") {
    maxent.UseSGD();
  } else {
    LOG(FATAL) << "Invalid optimization method : " << FLAGS_optim_method;
  }
  if (FLAGS_l1_reg > 0.0) {
    maxent.UseL1Reg(FLAGS_l1_reg);
  }
  if (FLAGS_l2_reg > 0.0) {
    maxent.UseL2Reg(FLAGS_l2_reg);
  }
  maxent.SetHeldout(FLAGS_num_heldout);
  maxent.SetFeatureFreqThreshold(FLAGS_feature_freq_threshold);

  LOG(INFO) << "Load training data from " << FLAGS_train_data_file;
  std::ifstream fin(FLAGS_train_data_file.c_str());
  if (!fin) {
    LOG(ERROR) << "Can't open train_data file '" << FLAGS_train_data_file
        << "'";
    return -1;
  }

  std::string line;
  while (std::getline(fin, line)) {
    std::vector<std::string> fields;
    common::SplitString(line, "\t", &fields);

    if (fields.size() < 2) {
      LOG(WARNING) << "Line format error. line: " << line;
      continue;
    }

    mltk::common::Instance instance;
    instance.set_label(fields[0]);
    for (size_t i = 1; i < fields.size(); ++i) {
      std::vector<std::string> feature_info;
      common::SplitString(fields[i], ":", &feature_info);
      const std::string& feature_name = feature_info[0];
      double value = atof(feature_info[1].c_str());
      instance.AddFeature(feature_name, value);
    }

    maxent.AddInstance(instance);
  }
  fin.close();

  LOG(INFO) << "MaxEnt model training.";
  maxent.Train();

  LOG(INFO) << "Save model to " << FLAGS_model_file;
  maxent.SaveModel(FLAGS_model_file);

  return 0;
}
