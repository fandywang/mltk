// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/logging.h"

#include <gtest/gtest.h>

TEST(Logging, CHECK) {
  bool flag = true;
  int i = 1;
  double f = 9.8;
  std::string str = "hello mltk.";

  CHECK(flag);
  CHECK(i > 0);
  CHECK(f >= 9.8);

  CHECK_EQ(true, flag);
  CHECK_EQ(1, i);
  CHECK_EQ(9.8, f);
  CHECK_EQ("hello mltk.", str);

  CHECK_GT(2, i);
  CHECK_GT(10, f);
  CHECK_GT("hello nltk", str);

  CHECK_LT(0, i);
  CHECK_LT(9.5, f);
  CHECK_LT("hello", str);

  CHECK_GE(1, i);
  CHECK_GE(9.8, f);
  CHECK_GE("hello mltk.", str);
  CHECK_GE(2, i);
  CHECK_GE(10, f);
  CHECK_GE("hello nltk", str);

  CHECK_LE(1, i);
  CHECK_LE(9.8, f);
  CHECK_LE("hello mltk.", str);
  CHECK_LE(0, i);
  CHECK_LE(9.5, f);
  CHECK_LE("hello", str);
}

TEST(Logger, LOG) {
  int i = 1;
  double f = 9.8;
  std::string str = "hello mltk.";

  LOG(INFO) << i << "\t" << f << "\t" << str << std::endl;
  LOG(WARNING) << i << "\t" << f << "\t" << str << std::endl;
  LOG(ERROR) << i << "\t" << f << "\t" << str << std::endl;
//  LOG(FATAL) << i << "\t" << f << "\t" << str << std::endl;
}
