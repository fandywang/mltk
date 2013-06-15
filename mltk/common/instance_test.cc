// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/instance.h"

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using mltk::common::Instance;

TEST(Instance, Label) {
  Instance instance1;

  instance1.set_label("IT");
  EXPECT_EQ("IT", instance1.label());
  instance1.set_label("Finance");
  EXPECT_EQ("Finance", instance1.label());

  Instance instance2("IT");
  EXPECT_EQ("IT", instance2.label());
  instance2.set_label("Finance");
  EXPECT_EQ("Finance", instance2.label());
}

TEST(Instance, AddFeature) {
  Instance instance("IT");
  ASSERT_EQ(0, instance.GetFeatures().size());

  instance.AddFeature("Apple", 0.65);
  instance.AddFeature("Microsoft", 0.8);
  instance.AddFeature("ipad", 0.45);
  instance.AddFeature("Google glass", 0.6);

  const std::vector<std::pair<std::string, double> >& features
      = instance.GetFeatures();
  ASSERT_EQ(4, features.size());

  EXPECT_EQ("Apple", features[0].first);
  EXPECT_EQ(0.65, features[0].second);
  EXPECT_EQ("Microsoft", features[1].first);
  EXPECT_EQ(0.8, features[1].second);
  EXPECT_EQ("ipad", features[2].first);
  EXPECT_EQ(0.45, features[2].second);
  EXPECT_EQ("Google glass", features[3].first);
  EXPECT_EQ(0.6, features[3].second);
}

