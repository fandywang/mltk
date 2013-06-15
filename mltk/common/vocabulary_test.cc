// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/common/vocabulary.h"

#include <gtest/gtest.h>

using mltk::common::StringMapType;
using mltk::common::Vocabulary;

TEST(Vocabulary, PutAndGet) {
  Vocabulary vocab;
  ASSERT_EQ(0, vocab.Size());
  ASSERT_TRUE(vocab.begin() == vocab.end());

  EXPECT_EQ(0, vocab.Put("Apple"));
  EXPECT_EQ(1, vocab.Put("Microsoft"));
  EXPECT_EQ(2, vocab.Put("ipad"));
  EXPECT_EQ(3, vocab.Put("Google glass"));
  EXPECT_EQ(0, vocab.Put("Apple"));
  EXPECT_EQ(2, vocab.Put("ipad"));
  ASSERT_EQ(4, vocab.Size());

  EXPECT_EQ(0, vocab.Id("Apple"));
  EXPECT_EQ(3, vocab.Id("Google glass"));
  EXPECT_EQ(-1, vocab.Id("Macbook Pro"));

  EXPECT_EQ("Apple", vocab.Str(0));
  EXPECT_EQ("Google glass", vocab.Str(3));

  for (StringMapType::const_iterator citer = vocab.begin();
       citer != vocab.end();
       ++citer) {
    EXPECT_TRUE(citer->second >= 0 && citer->second < vocab.Size());
  }

  vocab.Clear();
  ASSERT_EQ(0, vocab.Size());
}

