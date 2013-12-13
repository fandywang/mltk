// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/string_algorithm.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(StringAlgorithm, SplitString) {
  std::string str = "    hello world! hello mltk! ";

  std::vector<std::string> tokens;
  SplitString(str, " ", &tokens);
  EXPECT_EQ(4, tokens.size());
  EXPECT_EQ("hello", tokens[0]);
  EXPECT_EQ("world!", tokens[1]);
  EXPECT_EQ("hello", tokens[2]);
  EXPECT_EQ("mltk!", tokens[3]);

  str = "1111144444444555555555566666667777778884444666666";
  SplitString(str, "44", &tokens);
  EXPECT_EQ(3, tokens.size());
  EXPECT_EQ("11111", tokens[0]);
  EXPECT_EQ("55555555556666666777777888", tokens[1]);
  EXPECT_EQ("666666", tokens[2]);

  str = "";
  SplitString(str, "44", &tokens);
  EXPECT_EQ(0, tokens.size());
}
