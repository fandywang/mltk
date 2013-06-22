// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/feature.h"

#include <gtest/gtest.h>

using mltk::common::Feature;

TEST(Feature, FeatureFuncs) {
  Feature feature1(5, 50);

  EXPECT_EQ(5, feature1.LabelId());
  EXPECT_EQ(50, feature1.FeatureNameId());
  EXPECT_EQ(12805, feature1.Body());

  Feature feature2(100, 55);

  EXPECT_EQ(100, feature2.LabelId());
  EXPECT_EQ(55, feature2.FeatureNameId());
  EXPECT_EQ(14180, feature2.Body());
}
