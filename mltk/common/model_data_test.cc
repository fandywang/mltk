// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/model_data.h"

#include <vector>

#include <gtest/gtest.h>
#include "mltk/common/feature.h"
#include "mltk/common/instance.h"
#include "mltk/common/mem_instance.h"

using mltk::common::Feature;
using mltk::common::Instance;
using mltk::common::MemInstance;
using mltk::common::ModelData;

TEST(ModelData, SaveAndLoad) {
  ModelData model_data1;

  ASSERT_TRUE(model_data1.Load("testdata/test.model"));
  EXPECT_EQ(2, model_data1.NumClasses());
  EXPECT_EQ(167, model_data1.NumFeatures());
  EXPECT_EQ(167, model_data1.NumActiveFeatures());

  ASSERT_TRUE(model_data1.Save("testdata/test_bak.model"));

  ModelData model_data2;
  ASSERT_TRUE(model_data2.Load("testdata/test_bak.model"));

  EXPECT_EQ(model_data1.NumClasses(), model_data2.NumClasses());
  EXPECT_EQ(model_data1.NumFeatures(), model_data2.NumFeatures());
  EXPECT_EQ(model_data1.NumActiveFeatures(), model_data2.NumActiveFeatures());
}

TEST(ModelData, InitFromInstances) {
  std::vector<Instance> instances;

  Instance instance1("IT");
  instance1.AddFeature("Apple", 0.65);
  instance1.AddFeature("Microsoft", 0.8);
  instance1.AddFeature("ipad", 0.45);
  instance1.AddFeature("Google glass", 0.6);
  instances.push_back(instance1);
  instances.push_back(instance1);

  Instance instance2("Finance");
  instance2.AddFeature("Stock", 0.8);
  instance2.AddFeature("Wall Street", 0.9);
  instances.push_back(instance2);
  instances.push_back(instance2);

  ModelData model_data;
  model_data.InitFromInstances(instances);
  EXPECT_EQ(2, model_data.NumClasses());
  EXPECT_EQ(6, model_data.NumFeatures());
  EXPECT_EQ(0, model_data.NumActiveFeatures());

  ASSERT_TRUE(model_data.Save("testdata/test_bak.model"));
  ASSERT_TRUE(model_data.Load("testdata/test_bak.model"));
}

class ModelDataTest : public ::testing::Test {
 public:
  void SetUp() {
    ASSERT_TRUE(model_data_.Load("testdata/test.model"));
  }

  void TearDown() {}

  ModelData model_data_;
};

TEST_F(ModelDataTest, FormatInstances) {
  Instance instance;
  instance.set_label("-1");
  instance.AddFeature("100", 0.5);
  instance.AddFeature("119", 0.9);
  instance.AddFeature("NULL", 0.9);

  MemInstance mem_instance;

  model_data_.FormatInstance(instance, &mem_instance);

  EXPECT_EQ(0, mem_instance.label_id());

  MemInstance::ConstIterator citer(mem_instance);
  EXPECT_EQ(2, citer.FeatureNameId());
  EXPECT_EQ(0.5, citer.FeatureValue());
  citer.Next();
  EXPECT_EQ(13, citer.FeatureNameId());
  EXPECT_EQ(0.9, citer.FeatureValue());
  citer.Next();
  EXPECT_TRUE(citer.Done());
}

TEST_F(ModelDataTest, Label) {
  EXPECT_EQ(0, model_data_.LabelId("-1"));
  EXPECT_EQ(1, model_data_.LabelId("+1"));
  EXPECT_EQ(-1, model_data_.LabelId("NULL"));

  EXPECT_EQ("-1", model_data_.Label(0));
  EXPECT_EQ("+1", model_data_.Label(1));
}

TEST_F(ModelDataTest, Feature) {
  EXPECT_EQ(4, model_data_.FeatureId(Feature(0, 2)));
  EXPECT_EQ(19, model_data_.FeatureId(Feature(0, 13)));

  EXPECT_EQ(0, model_data_.FeatureAt(0).FeatureNameId());
  EXPECT_EQ(1, model_data_.FeatureAt(2).FeatureNameId());

  const std::vector<int32_t>& feature_ids = model_data_.FeatureIds(2);
  EXPECT_EQ(2, feature_ids.size());
}

TEST_F(ModelDataTest, Lambdas) {
  const std::vector<double>& lambdas = model_data_.Lambdas();
  ASSERT_EQ(167, lambdas.size());

  EXPECT_EQ(0.614847, lambdas[0]);
  EXPECT_EQ(-0.420031, lambdas[2]);

  std::vector<double>* lambdas_ptr = model_data_.MutableLambdas();
  (*lambdas_ptr)[0] = 0.1;
  const std::vector<double>& lambdas1 = model_data_.Lambdas();
  ASSERT_EQ(167, lambdas1.size());
  EXPECT_EQ(0.1, lambdas1[0]);

  std::vector<double> new_lambdas(lambdas.size(), 0);
  model_data_.UpdateLambdas(new_lambdas);

  const std::vector<double>& lambdas2 = model_data_.Lambdas();
  ASSERT_EQ(167, lambdas2.size());

  for (size_t i = 0; i < lambdas2.size(); ++i) {
    EXPECT_EQ(0, lambdas2[i]);
  }
}

TEST_F(ModelDataTest, CalcConditionalProbability) {
  Instance instance;
  instance.set_label("-1");
  instance.AddFeature("100", 0.5);
  instance.AddFeature("119", 0.9);
  MemInstance mem_instance;
  model_data_.FormatInstance(instance, &mem_instance);

  std::vector<double> prob_dist(model_data_.NumClasses());
  int32_t max_label_id = model_data_.CalcConditionalProbability(mem_instance,
                                                                &prob_dist);
  EXPECT_EQ(2, prob_dist.size());
  EXPECT_EQ(0, max_label_id);
  EXPECT_EQ(.94365970390140463, prob_dist[0]);
  EXPECT_EQ(.05634029609859529, prob_dist[1]);
}

