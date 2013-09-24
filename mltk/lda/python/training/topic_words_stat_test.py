#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import unittest
import sys

sys.path.append('..')
from common.model import Model
from common.vocabulary import Vocabulary
from topic_words_stat import TopicWordsStat

class TopicWordsStatTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(20)
        self.model.load('../testdata/lda_model')
        self.vocabulary = Vocabulary()
        self.vocabulary.load('../testdata/vocabulary.dat')

        self.topic_words_stat = TopicWordsStat(self.model, self.vocabulary)

    def test_save(self):
        print self.topic_words_stat.save('../testdata/topic_top_words.dat', 0.8)

    def test_get_topic_top_words(self):
        print self.topic_words_stat.get_topic_top_words(0.8)

    def test_compute_topic_word_distribution(self):
        print self.topic_words_stat.compute_topic_word_distribution()

if __name__ == '__main__':
    unittest.main()

