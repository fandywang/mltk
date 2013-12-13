#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import random
import unittest
import sys

sys.path.append('..')
from common.document import Document
from common.model import Model
from common.vocabulary import Vocabulary
import model_evaluator

class ModelEvaluatorTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(20)
        self.model.load('../testdata/lda_model')
        self.vocabulary = Vocabulary()
        self.vocabulary.load('../testdata/vocabulary.dat')

    def test_compute_loglikelihood(self):
        doc_tokens = ['macbook', 'ipad',  # exist in vocabulary and model
                'mac os x', 'chrome',  # only exist in vocabulary
                'nokia', 'null']  # inexistent
        document = Document(self.model.num_topics)
        rand = random.Random()
        rand.seed(0)
        document.parse_from_tokens(
                doc_tokens, rand, self.vocabulary, self.model)
        documents = [document, document]
        self.assertEqual(-14.113955684239654,
                model_evaluator.compute_loglikelihood(self.model, self.vocabulary, documents))

if __name__ == '__main__':
    unittest.main()

