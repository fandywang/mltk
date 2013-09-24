#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import logging
import random
import unittest

from document import Document
from model import Model
from vocabulary import Vocabulary

class DocumentTest(unittest.TestCase):

    def setUp(self):
        self.document = Document(20)
        self.vocabulary = Vocabulary()
        self.vocabulary.load("../testdata/vocabulary.dat")

        self.model = Model(20)
        self.model.load('../testdata/lda_model')

        self.doc_tokens = ['macbook', 'ipad',  # exist in vocabulary and model
                'mac os x', 'chrome',  # only exist in vocabulary
                'nokia', 'null']  # inexistent

    def test_parse_from_tokens(self):
        # initialize document during lda training.
        self.document.parse_from_tokens(
                self.doc_tokens, random, self.vocabulary)

        self.assertEqual(4, self.document.num_words())
        topic_hist = self.document.doc_topic_hist
        for i in xrange(len(topic_hist.non_zeros) - 1):
            self.assertGreaterEqual(topic_hist.non_zeros[i].count,
                    topic_hist.non_zeros[i + 1].count)
        logging.info(str(self.document))

        # initialize document during lda inference.
        self.document.parse_from_tokens(
                self.doc_tokens, random, self.vocabulary, self.model)
        self.assertEqual(2, self.document.num_words())
        for i in xrange(len(topic_hist.non_zeros) - 1):
            self.assertGreaterEqual(topic_hist.non_zeros[i].count,
                    topic_hist.non_zeros[i + 1].count)
        # print str(self.document)

    def test_serialize_and_parse(self):
        self.document.parse_from_tokens(
                self.doc_tokens, random, self.vocabulary)

        test_doc = Document(20)
        test_doc.parse_from_string(self.document.serialize_to_string())

        self.assertEqual(self.document.num_words(), test_doc.num_words())
        self.assertEqual(str(self.document), str(test_doc))

    def test_increase_decrease_topic(self):
        self.document.parse_from_tokens(
                self.doc_tokens, random, self.vocabulary, self.model)
        self.document.increase_topic(0, 5)
        self.document.increase_topic(4, 5)
        self.document.increase_topic(9, 5)
        topic_hist = self.document.doc_topic_hist
        for i in xrange(len(topic_hist.non_zeros) - 1):
            self.assertGreaterEqual(topic_hist.non_zeros[i].count,
                    topic_hist.non_zeros[i + 1].count)

        self.document.decrease_topic(4, 4)
        self.document.decrease_topic(9, 3)
        for i in xrange(len(topic_hist.non_zeros) - 1):
            self.assertGreaterEqual(topic_hist.non_zeros[i].count,
                    topic_hist.non_zeros[i + 1].count)

if __name__ == '__main__':
    unittest.main()

