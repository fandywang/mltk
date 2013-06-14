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

import random
import unittest
import sys

sys.path.append('..')
from common.model import Model
from common.vocabulary import Vocabulary
from sparselda_train_gibbs_sampler import SparseLDATrainGibbsSampler

class SparseLDATrainGibbsSamplerTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(20)
        self.vocabulary = Vocabulary()
        self.vocabulary.load('../testdata/vocabulary.dat')
        self.sparselda_train_gibbs_sampler = \
                SparseLDATrainGibbsSampler(self.model, self.vocabulary)

    def test_load_corpus(self):
        self.sparselda_train_gibbs_sampler.load_corpus('../testdata/corpus')
        self.assertEqual(4, len(self.sparselda_train_gibbs_sampler.documents))

    def test_gibbs_sampling(self):
        self.sparselda_train_gibbs_sampler.load_corpus('../testdata/corpus')
        rand = random.Random()
        for i in xrange(100):
            self.sparselda_train_gibbs_sampler.gibbs_sampling(rand)
            if (i + 1) % 10 == 0:
                self.sparselda_train_gibbs_sampler.save_checkpoint(
                        '../testdata/checkpoint', i + 1)
        self.sparselda_train_gibbs_sampler.save_model(
                '../testdata/train_model', 100)

    def test_load_checkpoint(self):
        cur_iteration = self.sparselda_train_gibbs_sampler.load_checkpoint(
                '../testdata/checkpoint')
        rand = random.Random()
        for i in xrange(cur_iteration, 200):
            self.sparselda_train_gibbs_sampler.gibbs_sampling(rand)
            if (i + 1) % 10 == 0:
                self.sparselda_train_gibbs_sampler.save_checkpoint(
                        '../testdata/checkpoint', i + 1)

if __name__ == '__main__':
    unittest.main()

