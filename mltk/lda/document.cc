// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#include "mltk/lda/document.h"

#include <stdlib.h>
#include <string>
#include <vector>

#include "mltk/common/city.h"
#include "mltk/common/string_algorithm.h"
#include "mltk/common/vocabulary.h"

namespace mltk {
namespace lda {

using mltk::common::Vocabulary;

bool Document::ParseFromTokens(const std::string& text,
                               const Vocabulary& vocab) {
  std::vector<std::string> tokens;
  SplitString(text, " ", &tokens);

  srand(::CityHash32(text.c_str(), text.length()));

  for (size_t i = 0; i < tokens.size(); ++i) {
    int32_t word_id = vocab.Id(tokens[i]);
    if (word_id != -1) {
      int32_t topic = static_cast<int32_t>(
          rand() / static_cast<double>(RAND_MAX) * doc_topic_histogram_.size());
      words_.push_back(Word(word_id, topic));
    }
  }

  for (WordOccurrenceIterator iter(this); !iter.Done(); iter.Next()) {
    ++doc_topic_histogram_[iter.Topic()];
  }

  return true;
}

std::string Document::DebugString() const {
  std::string str;

  for (size_t i = 0; i < words_.size(); ++i) {
    char buf[100];
    snprintf(buf, sizeof(buf), "%d %d", words_[i].first, words_[i].second);
    str.append(buf);
    str.append(" ");
  }
  str.append("####");
  for (size_t i = 0; i < doc_topic_histogram_.size(); ++i) {
    char buf[100];
    snprintf(buf, sizeof(buf), "%ld %lld", i, doc_topic_histogram_[i]);
    str.append(buf);
    str.append(" ");
  }

  return str;
}

}  // namespace lda
}  // namespace mltk

