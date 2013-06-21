// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/instance.h"

#include <string>

#include <gtest/gtest.h>

using mltk::common::Instance;

TEST(Instance, Label) {
  Instance instance;

  instance.set_label("IT");
  EXPECT_EQ("IT", instance.label());
  instance.set_label("Finance");
  EXPECT_EQ("Finance", instance.label());
}

TEST(Instance, Feature) {
  Instance instance("IT");

  instance.AddFeature("Apple", 0.65);
  instance.AddFeature("Microsoft", 0.8);
  instance.AddFeature("ipad", 0.45);
  instance.AddFeature("Google glass", 0.6);

  Instance::ConstIterator citer(instance);
  EXPECT_EQ("Apple", citer.FeatureName());
  EXPECT_EQ(0.65, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ("Microsoft", citer.FeatureName());
  EXPECT_EQ(0.8, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ("ipad", citer.FeatureName());
  EXPECT_EQ(0.45, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ("Google glass", citer.FeatureName());
  EXPECT_EQ(0.6, citer.FeatureValue());
  ASSERT_TRUE(citer.Done());
}

