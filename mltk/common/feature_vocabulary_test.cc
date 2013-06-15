// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/feature_vocabulary.h"

#include <gtest/gtest.h>

using mltk::common::Feature;
using mltk::common::FeatureVocabulary;

TEST(FeatureVocabulary, PutAndGet) {
  FeatureVocabulary feature_vocab;
  ASSERT_EQ(0, feature_vocab.Size());

  EXPECT_EQ(0, feature_vocab.Put(Feature(10, 100)));
  EXPECT_EQ(1, feature_vocab.Put(Feature(20, 200)));
  EXPECT_EQ(2, feature_vocab.Put(Feature(30, 300)));
  EXPECT_EQ(3, feature_vocab.Put(Feature(40, 400)));
  EXPECT_EQ(0, feature_vocab.Put(Feature(10, 100)));
  EXPECT_EQ(2, feature_vocab.Put(Feature(30, 300)));
  ASSERT_EQ(4, feature_vocab.Size());

  EXPECT_EQ(0, feature_vocab.FeatureId(Feature(10, 100)));
  EXPECT_EQ(3, feature_vocab.FeatureId(Feature(40, 400)));
  EXPECT_EQ(-1, feature_vocab.FeatureId(Feature(10, 400)));

  EXPECT_EQ(10, feature_vocab.GetFeature(0).LabelId());
  EXPECT_EQ(100, feature_vocab.GetFeature(0).FeatureId());
  EXPECT_EQ(40, feature_vocab.GetFeature(3).LabelId());
  EXPECT_EQ(400, feature_vocab.GetFeature(3).FeatureId());

  feature_vocab.Clear();
  ASSERT_EQ(0, feature_vocab.Size());
}

