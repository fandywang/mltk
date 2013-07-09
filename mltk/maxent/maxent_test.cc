// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/maxent/maxent.h"

#include <string>

#include <gtest/gtest.h>
#include "mltk/common/instance.h"
#include "mltk/maxent/lbfgs.h"
#include "mltk/maxent/optimizer.h"
#include "mltk/maxent/owlqn.h"
#include "mltk/maxent/sgd.h"

using mltk::common::Instance;
using mltk::maxent::LBFGS;
using mltk::maxent::MaxEnt;
using mltk::maxent::Optimizer;
using mltk::maxent::OWLQN;
using mltk::maxent::SGD;

const static std::string kModelFile = "maxent.model";

TEST(MaxEnt, TrainUsingSGD) {
  Optimizer* optim = new SGD(50, 1);
  optim->UseL1Reg(0.1);

  MaxEnt maxent(optim);
  EXPECT_EQ(0, maxent.NumClasses());

  std::vector<Instance> instances;
  Instance instance1("IT");
  instance1.AddFeature("Apple", 0.68);
  instance1.AddFeature("ipad", 0.5);
  instances.push_back(instance1);
  instances.push_back(instance1);

  Instance instance2("IT");
  instance2.AddFeature("Macbook Air", 0.8);
  instance2.AddFeature("iphone 4s", 0.9);
  instances.push_back(instance2);
  instances.push_back(instance2);

  Instance instance3("Finance");
  instance3.AddFeature("Wall Street", 0.8);
  instance3.AddFeature("QE", 0.9);
  instance3.AddFeature("stock", 0.88);
  instance1.AddFeature("Apple", 0.2);
  instances.push_back(instance3);
  instances.push_back(instance3);

  ASSERT_TRUE(maxent.Train(instances));

  EXPECT_EQ(2, maxent.NumClasses());

  EXPECT_EQ(0, maxent.GetClassId("IT"));
  EXPECT_EQ(1, maxent.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent.GetClassLabel(1));

  ASSERT_TRUE(maxent.SaveModel(kModelFile));

  MaxEnt maxent1;
  ASSERT_TRUE(maxent1.LoadModel(kModelFile));

  EXPECT_EQ(2, maxent1.NumClasses());

  EXPECT_EQ(0, maxent1.GetClassId("IT"));
  EXPECT_EQ(1, maxent1.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent1.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent1.GetClassLabel(1));

  delete optim;
}

TEST(MaxEnt, TrainUsingOWLQN) {
  Optimizer* optim = new OWLQN(300, 10);
  optim->UseL1Reg(0.1);

  MaxEnt maxent(optim);
  EXPECT_EQ(0, maxent.NumClasses());

  std::vector<Instance> instances;
  Instance instance1("IT");
  instance1.AddFeature("Apple", 0.68);
  instance1.AddFeature("ipad", 0.5);
  instances.push_back(instance1);
  instances.push_back(instance1);

  Instance instance2("IT");
  instance2.AddFeature("Macbook Air", 0.8);
  instance2.AddFeature("iphone 4s", 0.9);
  instances.push_back(instance2);
  instances.push_back(instance2);

  Instance instance3("Finance");
  instance3.AddFeature("Wall Street", 0.8);
  instance3.AddFeature("QE", 0.9);
  instance3.AddFeature("stock", 0.88);
  instance1.AddFeature("Apple", 0.2);
  instances.push_back(instance3);
  instances.push_back(instance3);

  ASSERT_TRUE(maxent.Train(instances));

  EXPECT_EQ(2, maxent.NumClasses());

  EXPECT_EQ(0, maxent.GetClassId("IT"));
  EXPECT_EQ(1, maxent.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent.GetClassLabel(1));

  ASSERT_TRUE(maxent.SaveModel(kModelFile));

  MaxEnt maxent1;
  ASSERT_TRUE(maxent1.LoadModel(kModelFile));

  EXPECT_EQ(2, maxent1.NumClasses());
  // reestablish a mapping table
  EXPECT_EQ(0, maxent1.GetClassId("Finance"));
  EXPECT_EQ(1, maxent1.GetClassId("IT"));
  EXPECT_EQ("Finance", maxent1.GetClassLabel(0));
  EXPECT_EQ("IT", maxent1.GetClassLabel(1));

  delete optim;
}

TEST(MaxEnt, TrainUsingLBFGS) {
  Optimizer* optim = new LBFGS(300, 10);
  optim->UseL2Reg(0.1);

  MaxEnt maxent(optim);
  EXPECT_EQ(0, maxent.NumClasses());

  std::vector<Instance> instances;
  Instance instance1("IT");
  instance1.AddFeature("Apple", 0.68);
  instance1.AddFeature("ipad", 0.5);
  instances.push_back(instance1);
  instances.push_back(instance1);

  Instance instance2("IT");
  instance2.AddFeature("Macbook Air", 0.8);
  instance2.AddFeature("iphone 4s", 0.9);
  instances.push_back(instance2);
  instances.push_back(instance2);

  Instance instance3("Finance");
  instance3.AddFeature("Wall Street", 0.8);
  instance3.AddFeature("QE", 0.9);
  instance3.AddFeature("stock", 0.88);
  instance1.AddFeature("Apple", 0.2);
  instances.push_back(instance3);
  instances.push_back(instance3);

  ASSERT_TRUE(maxent.Train(instances));

  EXPECT_EQ(2, maxent.NumClasses());

  EXPECT_EQ(0, maxent.GetClassId("IT"));
  EXPECT_EQ(1, maxent.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent.GetClassLabel(1));

  ASSERT_TRUE(maxent.SaveModel(kModelFile));

  MaxEnt maxent1;
  ASSERT_TRUE(maxent1.LoadModel(kModelFile));

  EXPECT_EQ(2, maxent1.NumClasses());

  EXPECT_EQ(0, maxent1.GetClassId("IT"));
  EXPECT_EQ(1, maxent1.GetClassId("Finance"));
  EXPECT_EQ("IT", maxent1.GetClassLabel(0));
  EXPECT_EQ("Finance", maxent1.GetClassLabel(1));

  delete optim;
}

const static double kEpsilon = 1E-6;
TEST(MaxEnt, Predict) {
  MaxEnt maxent;
  ASSERT_TRUE(maxent.LoadModel(kModelFile));
  EXPECT_EQ(2, maxent.NumClasses());

  Instance instance1("IT");
  instance1.AddFeature("Macbook Air", 0.5);
  instance1.AddFeature("iphone 4s", 0.8);
  instance1.AddFeature("iphone", 0.8);

  std::vector<double> prob_dist1 = maxent.Predict(&instance1);
  EXPECT_EQ(maxent.NumClasses(), static_cast<int32_t>(prob_dist1.size()));
  EXPECT_EQ("IT", instance1.label());
  EXPECT_NEAR(.990316, prob_dist1[maxent.GetClassId(instance1.label())],
              kEpsilon);

  Instance instance2("IT");
  instance2.AddFeature("Wall Street", 0.8);
  instance2.AddFeature("QE", 0.9);

  std::vector<double> prob_dist2 = maxent.Predict(&instance2);
  EXPECT_EQ(maxent.NumClasses(), static_cast<int32_t>(prob_dist2.size()));
  EXPECT_EQ("Finance", instance2.label());
  EXPECT_NEAR(.997918, prob_dist2[maxent.GetClassId(instance2.label())],
              kEpsilon);
}

