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
from recordio import RecordWriter
from recordio import RecordReader
from lda_pb2 import WordTopicHistogramPB

class RecordIOTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_read_and_write_normal(self):
        fp = open('../testdata/recordio.dat', 'wb')
        record_writer = RecordWriter(fp)
        self.assertFalse(record_writer.write(111))
        self.assertFalse(record_writer.write(111.89))
        self.assertFalse(record_writer.write(True))
        self.assertTrue(record_writer.write('111'))
        self.assertTrue(record_writer.write('89'))
        self.assertTrue(record_writer.write('apple'))
        self.assertTrue(record_writer.write('ipad'))
        fp.close()

        fp = open('../testdata/recordio.dat', 'rb')
        record_reader = RecordReader(fp)
        self.assertEqual('111', record_reader.read())
        self.assertEqual('89', record_reader.read())
        self.assertEqual('apple', record_reader.read())
        self.assertEqual('ipad', record_reader.read())
        self.assertIsNone(record_reader.read())
        fp.close()

    def test_read_and_writer_pb(self):
        fp = open('../testdata/recordio.dat', 'wb')
        record_writer = RecordWriter(fp)
        for i in xrange(20):
            word_topic_hist = WordTopicHistogramPB()
            word_topic_hist.word = i
            for j in xrange(20):
                non_zero = word_topic_hist.sparse_topic_hist.non_zeros.add()
                non_zero.topic = j
                non_zero.count = j + 1
            self.assertTrue(
                    record_writer.write(word_topic_hist.SerializeToString()))
        fp.close()

        fp = open('../testdata/recordio.dat', 'rb')
        record_reader = RecordReader(fp)
        i = 0
        while True:
            blob = record_reader.read()
            if blob == None:
                break
            word_topic_hist = WordTopicHistogramPB()
            word_topic_hist.ParseFromString(blob)
            self.assertEqual(i, word_topic_hist.word)
            sparse_topic_hist = word_topic_hist.sparse_topic_hist
            self.assertEqual(20, len(sparse_topic_hist.non_zeros))
            for j in xrange(len(sparse_topic_hist.non_zeros)):
                self.assertEqual(j, sparse_topic_hist.non_zeros[j].topic)
                self.assertEqual(j + 1, sparse_topic_hist.non_zeros[j].count)
            i += 1
        self.assertEqual(20, i)
        fp.close()

if __name__ == '__main__':
    unittest.main()

