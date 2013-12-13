#!/usr/bin/env python
# coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import unittest

from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram

class OrderedSparseTopicHistogramTest(unittest.TestCase):

    def setUp(self):
        self.num_topics = 20
        self.ordered_sparse_topic_hist = \
                OrderedSparseTopicHistogram(self.num_topics)
        [self.ordered_sparse_topic_hist.increase_topic(i, i + 1) \
            for i in xrange(10)]

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

    def test_get_non_zeros(self):
      i = 9
      for non_zero in self.ordered_sparse_topic_hist.get_non_zeros():
          self.assertEqual(i, non_zero.topic)
          self.assertEqual(i + 1, non_zero.count)
          i = i - 1

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

