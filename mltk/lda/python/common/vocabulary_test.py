#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

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
