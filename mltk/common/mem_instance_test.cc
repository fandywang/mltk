// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/mem_instance.h"

#include <gtest/gtest.h>

using mltk::common::MemInstance;

TEST(MemInstance, Label) {
  MemInstance mem_instance;

  mem_instance.set_label_id(1);
  EXPECT_EQ(1, mem_instance.label_id());
  mem_instance.set_label_id(2);
  EXPECT_EQ(2, mem_instance.label_id());
}

TEST(MemInstance, Feature) {
  MemInstance mem_instance(1);

  mem_instance.AddFeature(1, 0.65);
  mem_instance.AddFeature(2, 0.8);
  mem_instance.AddFeature(3, 0.45);
  mem_instance.AddFeature(4, 0.6);

  MemInstance::ConstIterator citer(mem_instance);
  EXPECT_EQ(1, citer.FeatureNameId());
  EXPECT_EQ(0.65, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ(2, citer.FeatureNameId());
  EXPECT_EQ(0.8, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ(3, citer.FeatureNameId());
  EXPECT_EQ(0.45, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ(4, citer.FeatureNameId());
  EXPECT_EQ(0.6, citer.FeatureValue());
  citer.Next();
  ASSERT_TRUE(citer.Done());
}

