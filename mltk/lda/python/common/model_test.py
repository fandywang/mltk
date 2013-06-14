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

import os
import unittest
from model import Model
from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram

class ModelTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(20)

        # initialize self.model.global_topic_hist and
        # self.model.word_topic_hist
        for i in xrange(10):
            ordered_sparse_topic_hist = OrderedSparseTopicHistogram(20)
            for j in xrange(10 + i):
                ordered_sparse_topic_hist.increase_topic(j, j + 1)
                self.model.global_topic_hist[j] += j + 1
            self.model.word_topic_hist[i] = ordered_sparse_topic_hist

    def test_save_and_load(self):
        model_dir = '../testdata/lda_model'
        self.model.save(model_dir)
        self.assertTrue(os.path.exists(model_dir))

        new_model = Model(20)
        new_model.load(model_dir)

        self.assertEqual(new_model.num_topics, self.model.num_topics)
        self.assertEqual(len(new_model.word_topic_hist),
                len(self.model.word_topic_hist))

        for word, new_sparse_topic_hist in new_model.word_topic_hist.iteritems():
            self.assertTrue(word in self.model.word_topic_hist)
            sparse_topic_hist = self.model.word_topic_hist[word]
            self.assertEqual(new_sparse_topic_hist.size(),
                    sparse_topic_hist.size())

            for j in xrange(new_sparse_topic_hist.size()):
                self.assertEqual(new_sparse_topic_hist.non_zeros[j].topic,
                        sparse_topic_hist.non_zeros[j].topic)
                self.assertEqual(new_sparse_topic_hist.non_zeros[j].count,
                        sparse_topic_hist.non_zeros[j].count)

        self.assertEqual(new_model.hyper_params.topic_prior,
                self.model.hyper_params.topic_prior)
        self.assertEqual(new_model.hyper_params.word_prior,
                self.model.hyper_params.word_prior)

        # print self.model

    def test_has_word(self):
        self.assertTrue(self.model.has_word(0))
        self.assertTrue(self.model.has_word(2))
        self.assertTrue(self.model.has_word(4))
        self.assertTrue(self.model.has_word(6))
        self.assertTrue(self.model.has_word(8))
        self.assertFalse(self.model.has_word(10))
        self.assertFalse(self.model.has_word(12))
        self.assertFalse(self.model.has_word(14))
        self.assertFalse(self.model.has_word(16))
        self.assertFalse(self.model.has_word(18))

    def test_get_word_topic_dist(self):
        word_topic_dist = self.model.get_word_topic_dist(10)
        self.assertTrue(len(word_topic_dist))
        # print word_topic_dist

if __name__ == '__main__':
    unittest.main()

