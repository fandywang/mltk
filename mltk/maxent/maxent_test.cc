// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <string>

#include <gtest/gtest.h>
#include "mltk/common/instance.h"

using mltk::common::Instance;
using mltk::maxent::MaxEnt;

const static std::string kModelFile = "maxent.model";

TEST(MaxEnt, Train) {
  MaxEnt maxent;
  EXPECT_EQ(0, maxent.NumClasses());

  Instance instance1("IT");
  instance1.AddFeature("Apple", 0.68);
  instance1.AddFeature("ipad", 0.5);
  maxent.AddInstance(instance1);
  maxent.AddInstance(instance1);

  Instance instance2("IT");
  instance2.AddFeature("Macbook Air", 0.8);
  instance2.AddFeature("iphone 4s", 0.9);
  maxent.AddInstance(instance2);
  maxent.AddInstance(instance2);

  Instance instance3("Finance");
  instance3.AddFeature("Wall Street", 0.8);
  instance3.AddFeature("QE", 0.9);
  instance3.AddFeature("stock", 0.88);
  instance1.AddFeature("Apple", 0.2);
  maxent.AddInstance(instance3);
  maxent.AddInstance(instance3);

  EXPECT_EQ(2, maxent.NumClasses());

  EXPECT_EQ(0, maxent.GetClassId("IT"));
  EXPECT_EQ(1, maxent.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent.GetClassLabel(1));

  ASSERT_TRUE(maxent.Train());

  ASSERT_TRUE(maxent.SaveModel(kModelFile));

  MaxEnt maxent1;
  ASSERT_TRUE(maxent1.LoadModel(kModelFile));

  EXPECT_EQ(2, maxent1.NumClasses());

  EXPECT_EQ(0, maxent1.GetClassId("IT"));
  EXPECT_EQ(1, maxent1.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent1.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent1.GetClassLabel(1));
}

const static double kEpsilon = 1E-6;
TEST(MaxEnt, Classify) {
  MaxEnt maxent;
  ASSERT_TRUE(maxent.LoadModel(kModelFile));
  EXPECT_EQ(2, maxent.NumClasses());

  Instance instance1;
  instance1.AddFeature("Macbook Air", 0.5);
  instance1.AddFeature("iphone 4s", 0.8);
  instance1.AddFeature("iphone", 0.8);

  std::vector<double> prob_dist1 = maxent.Classify(&instance1);
  EXPECT_EQ(maxent.NumClasses(), static_cast<int32_t>(prob_dist1.size()));
  EXPECT_EQ("IT", instance1.label());
  EXPECT_NEAR(0.999135, prob_dist1[maxent.GetClassId(instance1.label())],
              kEpsilon);

  Instance instance2;
  instance2.AddFeature("Wall Street", 0.8);
  instance2.AddFeature("QE", 0.9);

  std::vector<double> prob_dist2 = maxent.Classify(&instance2);
  EXPECT_EQ(maxent.NumClasses(), static_cast<int32_t>(prob_dist2.size()));
  EXPECT_EQ("Finance", instance2.label());
  EXPECT_NEAR(0.997498, prob_dist2[maxent.GetClassId(instance2.label())],
              kEpsilon);
}

