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

import unittest
from vocabulary import Vocabulary

class VocabularyTest(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary()
        self.vocabulary.load("../testdata/vocabulary.dat")

    def test_has_word(self):
        self.assertTrue(self.vocabulary.has_word('ipad'))
        self.assertTrue(self.vocabulary.has_word('iphone'))
        self.assertTrue(self.vocabulary.has_word('macbook'))
        self.assertFalse(self.vocabulary.has_word('nokia'))
        self.assertFalse(self.vocabulary.has_word('thinkpad'))

    def test_word_index(self):
        self.assertEqual(0, self.vocabulary.word_index('ipad'))
        self.assertEqual(1, self.vocabulary.word_index('iphone'))
        self.assertEqual(2, self.vocabulary.word_index('macbook'))
        self.assertEqual(-1, self.vocabulary.word_index('nokia'))
        self.assertEqual(-1, self.vocabulary.word_index('thinkpad'))

    def test_word(self):
        self.assertEqual('ipad', self.vocabulary.word(0))
        self.assertEqual('iphone', self.vocabulary.word(1))
        self.assertEqual('macbook', self.vocabulary.word(2))

    def test_size(self):
        self.assertEqual(17, self.vocabulary.size())

if __name__ == '__main__':
    unittest.main()
