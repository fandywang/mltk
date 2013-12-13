// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// Copy and Modify from Google's plda (https://code.google.com/p/plda).
// document.h
//
// Copyright 2008 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Stores a document as a bag of words and provides methods for interacting
// with Gibbs LDA models.

#ifndef MLTK_LDA_DOCUMENT_H_
#define MLTK_LDA_DOCUMENT_H_

#include <list>
#include <string>
#include <utility>
#include <vector>

#include "mltk/common/logging.h"

namespace mltk {

namespace common {
class Vocabulary;
}  // namespace common

namespace lda {

typedef std::pair<int32_t, int32_t> Word;
typedef std::vector<int64_t> DenseTopicDistribution;

class Document {
 public:
  Document(int32_t num_topics) { doc_topic_dist_.resize(num_topics, 0); }
  ~Document() {}

  // An iterator over all of the word occurrences in a document.
  class WordOccurrenceIterator {
   public:
    // Intialize the WordOccurrenceIterator for a document.
    explicit WordOccurrenceIterator(Document* doc)
        : doc_(doc), word_index_(0) {
      CHECK(doc_ != NULL);
    }
    ~WordOccurrenceIterator() {}

    // Returns true if we are done iterating.
    bool Done() {
      CHECK(word_index_ <= doc_->words_.size());
      return word_index_ == doc_->words_.size();
    }

    // Advances to the next word occurrence.
    void Next() {
      CHECK(!Done());
      ++word_index_;
    }

    // Returns the word of the current occurrence.
    int32_t Word() {
      CHECK(!Done());
      return doc_->words_[word_index_].first;
    }

    // Returns the topic of the current occurrence.
    int32_t Topic() {
      CHECK(!Done());
      return doc_->words_[word_index_].second;
    }

    // Changes the topic of the current occurrence.
    void SetTopic(int32_t new_topic) {
      CHECK(new_topic >= 0 && new_topic < doc_->doc_topic_dist_.size());

      --doc_->doc_topic_dist_[Topic()];
      ++doc_->doc_topic_dist_[new_topic];
      doc_->words_[word_index_].second = new_topic;
    }

   private:
    Document* doc_;
    int32_t word_index_;
  };

  friend class WordOccurrenceIterator;

  // Initialize the document with text which separated by a whitespace.
  bool ParseFromTokens(const std::string& text,
                       const common::Vocabulary& vocab);

  // Returns the document's length.
  size_t NumWords() const { return words_.size(); }

  // Returns the document's words.
  const std::vector<Word>& Words() const { return words_; }

  // Returns the total number of topics.
  int32_t NumTopics() const { return doc_topic_dist_.size(); }

  // Returns the document's topic occurrence counts - topic dist.
  const DenseTopicDistribution& DocTopicDistribution() const {
    return doc_topic_dist_;
  }

  std::string DebugString() const;

 private:
  std::vector<Word> words_;  // word occurrences
  DenseTopicDistribution doc_topic_dist_;  // the document's topic distribution
};

typedef std::list<Document*> LDACorpus;

}  // namespace lda
}  // namespace mltk

#endif  // MLTK_LDA_DOCUMENT_H_

