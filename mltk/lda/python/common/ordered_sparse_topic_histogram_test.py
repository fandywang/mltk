#!/usr/bin/env  python
# coding=utf-8

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

from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram

class OrderedSparseTopicHistogramTest(unittest.TestCase):

    def setUp(self):
        self.num_topics = 20
        self.ordered_sparse_topic_hist = \
                OrderedSparseTopicHistogram(self.num_topics)
        for i in xrange(10):
            self.ordered_sparse_topic_hist.increase_topic(i, i + 1)

    def test_ordered_sparse_topic_hist(self):
        self.assertEqual(10, len(self.ordered_sparse_topic_hist.non_zeros))
        for i in xrange(len(self.ordered_sparse_topic_hist.non_zeros)):
            self.assertEqual(10 - i - 1,
                    self.ordered_sparse_topic_hist.non_zeros[i].topic)
            self.assertEqual(10 - i,
                    self.ordered_sparse_topic_hist.non_zeros[i].count)

    def test_num_topics(self):
        self.assertEqual(self.num_topics,
                self.ordered_sparse_topic_hist.num_topics)

    def test_size(self):
        self.assertEqual(10, self.ordered_sparse_topic_hist.size())

    def test_serialize_and_parse(self):
        blob = self.ordered_sparse_topic_hist.serialize_to_string()

        sparse_topic_hist = OrderedSparseTopicHistogram(self.num_topics)
        sparse_topic_hist.parse_from_string(blob)

        self.assertEqual(sparse_topic_hist.size(),
                self.ordered_sparse_topic_hist.size())
        self.assertEqual(str(sparse_topic_hist),
                str(self.ordered_sparse_topic_hist))

    def test_count(self):
        for i in xrange(10):
            self.assertEqual(i + 1, self.ordered_sparse_topic_hist.count(i))
        for i in xrange(10, 20):
            self.assertEqual(0, self.ordered_sparse_topic_hist.count(i))

    def test_increase_topic(self):
        for i in xrange(20):
            if i < 10:
                self.assertEqual(2 * (i + 1),
                        self.ordered_sparse_topic_hist.increase_topic(i, i + 1))
            else:
                self.assertEqual(i + 1,
                        self.ordered_sparse_topic_hist.increase_topic(i, i + 1))

            for j in xrange(len(self.ordered_sparse_topic_hist.non_zeros) - 1):
                self.assertGreaterEqual(
                        self.ordered_sparse_topic_hist.non_zeros[j].count,
                        self.ordered_sparse_topic_hist.non_zeros[j + 1].count)

        self.assertEqual(2, self.ordered_sparse_topic_hist.count(0))
        self.assertEqual(12, self.ordered_sparse_topic_hist.count(5))
        self.assertEqual(11, self.ordered_sparse_topic_hist.count(10))
        self.assertEqual(16, self.ordered_sparse_topic_hist.count(15))
        self.assertEqual(20, self.ordered_sparse_topic_hist.increase_topic(15, 4))

    def test_decrease_topic(self):
        self.assertEqual(6, self.ordered_sparse_topic_hist.count(5))
        self.assertEqual(7, self.ordered_sparse_topic_hist.count(6))
        self.assertEqual(5, self.ordered_sparse_topic_hist.decrease_topic(5, 1))
        self.assertEqual(3, self.ordered_sparse_topic_hist.decrease_topic(6, 4))
        self.assertEqual(10, self.ordered_sparse_topic_hist.size())
        self.assertEqual(5, self.ordered_sparse_topic_hist.count(5))
        self.assertEqual(3, self.ordered_sparse_topic_hist.count(6))

        for i in xrange(len(self.ordered_sparse_topic_hist.non_zeros) - 1):
            self.assertGreaterEqual(
                    self.ordered_sparse_topic_hist.non_zeros[i].count,
                    self.ordered_sparse_topic_hist.non_zeros[i + 1].count)

        self.assertEqual(0, self.ordered_sparse_topic_hist.decrease_topic(6, 3))
        self.assertEqual(9, self.ordered_sparse_topic_hist.size())
        for i in xrange(len(self.ordered_sparse_topic_hist.non_zeros) - 1):
            self.assertGreaterEqual(
                    self.ordered_sparse_topic_hist.non_zeros[i].count,
                    self.ordered_sparse_topic_hist.non_zeros[i + 1].count)

if __name__ == '__main__':
    unittest.main()

