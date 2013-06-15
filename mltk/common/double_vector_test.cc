// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/double_vector.h"

#include <gtest/gtest.h>

using mltk::common::DoubleVector;

TEST(DoubleVector, Ctor) {
  DoubleVector vec1;
  ASSERT_EQ(0, vec1.Size());
  ASSERT_EQ(0, vec1.STLVector().size());

  DoubleVector vec2(10);
  ASSERT_EQ(10, vec2.Size());
  ASSERT_EQ(10, vec2.STLVector().size());
  for (size_t i = 0; i < vec2.Size(); ++i) {
    EXPECT_EQ(0, vec2[i]);
    EXPECT_EQ(0, vec2.STLVector()[i]);
  }

  DoubleVector vec3(10, 1);
  ASSERT_EQ(10, vec3.Size());
  ASSERT_EQ(10, vec3.STLVector().size());
  for (size_t i = 0; i < vec3.Size(); ++i) {
    EXPECT_EQ(1, vec3[i]);
    EXPECT_EQ(1, vec3.STLVector()[i]);
  }

  DoubleVector vec4 = vec3;
  ASSERT_EQ(10, vec4.Size());
  ASSERT_EQ(10, vec4.STLVector().size());
  for (size_t i = 0; i < vec4.Size(); ++i) {
    EXPECT_EQ(1, vec4[i]);
    EXPECT_EQ(1, vec4.STLVector()[i]);
  }
}

TEST(DoubleVector, Operator) {
  DoubleVector vec1(10, 2);
  DoubleVector vec2(10, 5);

  vec1 += vec2;
  ASSERT_EQ(10, vec1.Size());
  ASSERT_EQ(10, vec1.STLVector().size());
  for (size_t i = 0; i < vec1.Size(); ++i) {
    EXPECT_EQ(7, vec1[i]);
    EXPECT_EQ(7, vec1.STLVector()[i]);
  }

  vec1 *= 5;
  ASSERT_EQ(10, vec1.Size());
  ASSERT_EQ(10, vec1.STLVector().size());
  for (size_t i = 0; i < vec1.Size(); ++i) {
    EXPECT_EQ(35, vec1[i]);
    EXPECT_EQ(35, vec1.STLVector()[i]);
  }
}

TEST(DoubleVector, Project) {
  DoubleVector vec1(10, 2);
  DoubleVector vec2(10, 5);

  vec1.Project(vec2);
  ASSERT_EQ(10, vec1.Size());
  ASSERT_EQ(10, vec1.STLVector().size());
  for (size_t i = 0; i < vec1.Size(); ++i) {
    EXPECT_EQ(2, vec1[i]);
    EXPECT_EQ(2, vec1.STLVector()[i]);
  }

  DoubleVector vec3(10, -5);
  vec1.Project(vec3);
  ASSERT_EQ(10, vec1.Size());
  ASSERT_EQ(10, vec1.STLVector().size());
  for (size_t i = 0; i < vec1.Size(); ++i) {
    EXPECT_EQ(0, vec1[i]);
    EXPECT_EQ(0, vec1.STLVector()[i]);
  }
}

TEST(DoubleVector, Utils) {
  DoubleVector vec1(10, 2);
  DoubleVector vec2(10, 5);

  EXPECT_EQ(100, DotProduct(vec1, vec2));

  DoubleVector vec3 = vec1 + vec2;
  ASSERT_EQ(10, vec3.Size());
  ASSERT_EQ(10, vec3.STLVector().size());
  for (size_t i = 0; i < vec3.Size(); ++i) {
    EXPECT_EQ(7, vec3[i]);
    EXPECT_EQ(7, vec3.STLVector()[i]);
  }

  DoubleVector vec4 = vec1 - vec2;
  ASSERT_EQ(10, vec4.Size());
  ASSERT_EQ(10, vec4.STLVector().size());
  for (size_t i = 0; i < vec4.Size(); ++i) {
    EXPECT_EQ(-3, vec4[i]);
    EXPECT_EQ(-3, vec4.STLVector()[i]);
  }

  DoubleVector vec5 = vec1 * 5;
  ASSERT_EQ(10, vec5.Size());
  ASSERT_EQ(10, vec5.STLVector().size());
  for (size_t i = 0; i < vec5.Size(); ++i) {
    EXPECT_EQ(10, vec5[i]);
    EXPECT_EQ(10, vec5.STLVector()[i]);
  }

  DoubleVector vec6 = 5 * vec1;
  ASSERT_EQ(10, vec6.Size());
  ASSERT_EQ(10, vec6.STLVector().size());
  for (size_t i = 0; i < vec6.Size(); ++i) {
    EXPECT_EQ(10, vec6[i]);
    EXPECT_EQ(10, vec6.STLVector()[i]);
  }
}

