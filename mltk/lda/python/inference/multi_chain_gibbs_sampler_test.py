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
from multi_chain_gibbs_sampler import MultiChainGibbsSampler

class MultiChainGibbsSamplerTest(unittest.TestCase):

    def setUp(self):
        model = Model(20)
        model.load('../testdata/lda_model')
        vocabulary = Vocabulary()
        vocabulary.load('../testdata/vocabulary.dat')
        self.multi_chain_gibbs_sampler = \
                MultiChainGibbsSampler(model, vocabulary, 10, 10, 5)

    def test_infer_topics(self):
        doc_tokens = []
        doc_topic_dist = self.multi_chain_gibbs_sampler.infer_topics(doc_tokens)
        self.assertEqual(0, len(doc_topic_dist))

        doc_tokens = ['apple', 'ipad']
        doc_topic_dist = self.multi_chain_gibbs_sampler.infer_topics(doc_tokens)
        print doc_topic_dist
        self.assertEqual(5, len(doc_topic_dist))
        self.assertTrue(0 in doc_topic_dist)
        self.assertEqual(0.05, doc_topic_dist[0])
        self.assertTrue(1 in doc_topic_dist)
        self.assertEqual(0.32, doc_topic_dist[1])
        self.assertTrue(3 in doc_topic_dist)
        self.assertEqual(0.14, doc_topic_dist[3])

        doc_tokens = ['apple', 'ipad', 'apple', 'null', 'nokia', 'macbook']
        doc_topic_dist = self.multi_chain_gibbs_sampler.infer_topics(doc_tokens)
        print doc_topic_dist
        self.assertEqual(6, len(doc_topic_dist))

if __name__ == '__main__':
    unittest.main()

