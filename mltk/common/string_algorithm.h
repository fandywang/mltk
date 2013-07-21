// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)

#ifndef MLTK_COMMON_STRING_TOKENIZER_H_
#define MLTK_COMMON_STRING_TOKENIZER_H_

#include <string>
#include <vector>

#include "mltk/common/logging.h"

void SplitString(const std::string& str,
                 const std::string& seperators,
                 std::vector<std::string>* tokens) {
  CHECK(tokens != NULL);
  tokens->clear();

  int32_t n = str.length();
  int32_t start = str.find_first_not_of(seperators);
  int32_t stop;

  while (start >= 0 && start < n) {
    stop = str.find_first_of(seperators, start);
    if (stop < 0 || stop > n) { stop = n; }

    tokens->push_back(str.substr(start, stop - start));
    start = str.find_first_not_of(seperators, stop + 1);
  }
}

#endif  // MLTK_COMMON_STRING_TOKENIZER_H_

