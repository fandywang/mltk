// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
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
typedef std::vector<int64_t> DenseTopicHistogram;

class Document {
 public:
  Document(int32_t num_topics) { doc_topic_histogram_.resize(num_topics, 0); }
  ~Document() {}

  // An iterator over all of the word occurrences in a document.
  class WordOccurrenceIterator {
   public:
    // Intialize the WordOccurrenceIterator for a document.
    explicit WordOccurrenceIterator(Document* parent)
        : parent_(parent), word_index_(0) {
      CHECK(parent_ != NULL);
    }
    ~WordOccurrenceIterator() {}

    // Returns true if we are done iterating.
    bool Done() {
      CHECK(word_index_ <= parent_->words_.size());
      return word_index_ == parent_->words_.size();
    }

    // Advances to the next word occurrence.
    void Next() {
      CHECK(!Done());
      ++word_index_;
    }

    // Returns the word of the current occurrence.
    int32_t Word() {
      CHECK(!Done());
      return parent_->words_[word_index_].first;
    }

    // Returns the topic of the current occurrence.
    int32_t Topic() {
      CHECK(!Done());
      return parent_->words_[word_index_].second;
    }

    // Changes the topic of the current occurrence.
    void SetTopic(int32_t new_topic) {
      CHECK(new_topic >= 0 && new_topic < parent_->doc_topic_histogram_.size());

      --parent_->doc_topic_histogram_[Topic()];
      ++parent_->doc_topic_histogram_[new_topic];
      parent_->words_[word_index_].second = new_topic;
    }

   private:
    Document* parent_;
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
  int32_t NumTopics() const { return doc_topic_histogram_.size(); }

  // Returns the document's topic occurrence counts - topic histogram.
  const DenseTopicHistogram& DocTopicHistogram() const {
    return doc_topic_histogram_;
  }

  std::string DebugString() const;

 private:
  std::vector<Word> words_;  // word occurrences
  DenseTopicHistogram doc_topic_histogram_;  // the document's topic distribution
};

typedef std::list<Document*> LDACorpus;

}  // namespace lda
}  // namespace mltk

#endif  // MLTK_LDA_DOCUMENT_H_

