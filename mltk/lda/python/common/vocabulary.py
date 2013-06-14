#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class Vocabulary(object):
    """Vocabulary implements the structure of words.
    """
    def __init__(self):
        self.word_map = {}  # item fmt: word -> index
        self.words = []  # item fmt: word, default index

    def load(self, filename):
        """read words from filename.
        line fmt: word [\t count]
        """
        self.word_map.clear()
        self.words = []
        fp = open(filename, 'r')
        for line in fp.readlines():
            line = line.decode('gbk')
            fields = line.strip().split('\t')
            if len(fields) > 0 and fields[0] not in self.word_map:
                self.word_map[fields[0]] = len(self.words)
                self.words.append(fields[0])
        fp.close()

    def has_word(self, word):
        return (word in self.word_map)

    def word_index(self, word):
        return self.word_map.get(word, -1)

    def word(self, index):
        assert index >= 0 and index < len(self.words)
        return self.words[index]

    def size(self):
        return len(self.words)

