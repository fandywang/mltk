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

DEFINE_string(test_data_file, "", "the filename of test data.");
DEFINE_string(model_file, "", "the filename of maxent model.");

int main(int argc, char** argv) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  mltk::maxent::MaxEnt maxent;
  CHECK(maxent.LoadModel(FLAGS_model_file));

  int32_t ncorrect = 0;
  int32_t ntotal = 0;

  std::ifstream fin(FLAGS_test_data_file.c_str());
  if (!fin) {
    LOG(ERROR) << "Can't open test_data file '" << FLAGS_test_data_file
        << "'";
    return -1;
  }

  std::string line;
  while (std::getline(fin, line)) {
    mltk::common::Instance instance;
    if (instance.ParseFromText(line)) {
      const std::string& true_label = instance.label();
      maxent.Classify(&instance);
      if (instance.label() == true_label) { ++ncorrect; }
      ++ntotal;
    }
  }
  fin.close();

  LOG(ERROR) << "accuracy(" << ncorrect << " / " << ntotal << "): "
      << static_cast<double>(ncorrect) / ntotal;

  return 0;
}

