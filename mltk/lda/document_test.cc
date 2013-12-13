// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/lda/document.h"

#include <gtest/gtest.h>

#include "mltk/common/vocabulary.h"

using mltk::common::Vocabulary;
using mltk::lda::Document;
using mltk::lda::DenseTopicHistogram;
using mltk::lda::Word;

TEST(Document, Ctor) {
  Document doc(10);

  EXPECT_EQ(10, doc.NumTopics());
  EXPECT_EQ(0, doc.NumWords());
}

TEST(Document, ParseFromTokens) {
  Document doc(10);
  std::string text = "apple pie unknown";
  Vocabulary vocab;
  vocab.Put("apple");
  vocab.Put("pie");
  vocab.Put("macbook");
  vocab.Put("macbook air");
  vocab.Put("macbook pro");

  doc.ParseFromTokens(text, vocab);

  EXPECT_EQ(2, doc.NumWords());
  EXPECT_EQ(10, doc.NumTopics());
}

TEST(Document, WordOccurrenceIterator) {
  Document doc(10);
  std::string text = "apple pie unknown";
  Vocabulary vocab;
  vocab.Put("apple");
  vocab.Put("pie");
  vocab.Put("macbook");
  vocab.Put("macbook air");
  vocab.Put("macbook pro");

  doc.ParseFromTokens(text, vocab);

  Document::WordOccurrenceIterator iter(&doc);
  EXPECT_EQ(0, iter.Word());
  EXPECT_EQ(3, iter.Topic());
  iter.Next();
  EXPECT_EQ(1, iter.Word());
  EXPECT_EQ(6, iter.Topic());
  iter.Next();
  EXPECT_TRUE(iter.Done());

  const std::vector<Word>& words = doc.Words();
  EXPECT_EQ(2, words.size());
  EXPECT_EQ(0, words[0].first);
  EXPECT_EQ(3, words[0].second);
  EXPECT_EQ(1, words[1].first);
  EXPECT_EQ(6, words[1].second);

  const DenseTopicHistogram& topic_hist = doc.DocTopicHistogram();
  EXPECT_EQ(10, topic_hist.size());
  EXPECT_EQ(0, topic_hist[0]);
  EXPECT_EQ(0, topic_hist[1]);
  EXPECT_EQ(0, topic_hist[2]);
  EXPECT_EQ(1, topic_hist[3]);
  EXPECT_EQ(1, topic_hist[6]);
}

