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
import sys

sys.path.append('..')
from common.model import Model
from common.vocabulary import Vocabulary
from sparselda_gibbs_sampler import SparseLDAGibbsSampler

class SparseLDAGibbsSamplerTest(unittest.TestCase):

    def setUp(self):
        model = Model(20)
        model.load('../testdata/lda_model')
        vocabulary = Vocabulary()
        vocabulary.load('../testdata/vocabulary.dat')
        self.sparselda_gibbs_sampler = \
                SparseLDAGibbsSampler(model, vocabulary, 10, 5)

    def test_infer_topics(self):
        doc_tokens = []
        doc_topic_dist = self.sparselda_gibbs_sampler.infer_topics(doc_tokens)
        self.assertEqual(0, len(doc_topic_dist))

        doc_tokens = ['apple', 'ipad']
        doc_topic_dist = self.sparselda_gibbs_sampler.infer_topics(doc_tokens)
        self.assertEqual(3, len(doc_topic_dist))
        self.assertTrue(1 in doc_topic_dist)

        doc_tokens = ['apple', 'ipad', 'apple', 'null', 'nokia', 'macbook']
        doc_topic_dist = self.sparselda_gibbs_sampler.infer_topics(doc_tokens)
        self.assertEqual(4, len(doc_topic_dist))
        self.assertTrue(0 in doc_topic_dist)
        self.assertEqual(0.1, doc_topic_dist[0])
        self.assertTrue(2 in doc_topic_dist)
        self.assertEqual(0.3, doc_topic_dist[2])

if __name__ == '__main__':
    unittest.main()

