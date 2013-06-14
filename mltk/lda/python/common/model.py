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

import logging
import os

from lda_pb2 import GlobalTopicHistogramPB
from lda_pb2 import HyperParamsPB
from lda_pb2 import WordTopicHistogramPB
from lda_pb2 import SparseTopicHistogramPB
from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram
from recordio import RecordReader
from recordio import RecordWriter
from vocabulary import Vocabulary

class HyperParams(object):

    # TODO(fandywang): optimize the hyper_params.
    # Because we find that an asymmetric Dirichlet prior over the document-
    # topic distributions has substantial advantages over a symmetric prior,
    # while an asymmetric prior over topic-word distributions provides no
    # real benefit.
    #
    # See 'Hanna Wallach, David Mimno, and Andrew McCallum. 2009.
    # Rethinking LDA: Why priors matter. In Proceedings of NIPS-09,
    # Vancouver, BC.' for more details.
    def __init__(self, topic_prior = 0.01, word_prior = 0.1):
        self.topic_prior = topic_prior
        self.word_prior = word_prior

    def serialize_to_string(self):
        hyper_params_pb = HyperParamsPB()
        hyper_params_pb.topic_prior = self.topic_prior
        hyper_params_pb.word_prior = self.word_prior
        return hyper_params_pb.SerializeToString()

    def parse_from_string(self, hyper_params_str):
        hyper_params_pb = HyperParamsPB()
        hyper_params_pb.ParseFromString(hyper_params_str)
        self.topic_prior = hyper_params_pb.topic_prior
        self.word_prior = hyper_params_pb.word_prior

    def __str(self):
        return '<topic_prior: ' + str(self.topic_prior) + \
                ', word_prior: ' + str(self.word_prior) + '>'


class Model(object):
    """Model implements the sparselda model.
    It includes the following parts:
        0. num_topics, represents |K|.
        1. global_topic_hist, represents N(z).
        2. word_topic_hist, represents N(w|z).
        3. hyper_params
           3.1 topic_prior, represents the dirichlet prior of topic \alpha.
           3.2 word_prior, represents the dirichlet prior of word \beta.
    """
    GLOABLE_TOPIC_HIST_FILENAME = "lda.global_topic_hist"
    WORD_TOPIC_HIST_FILENAME = "lda.word_topic_hist"
    HYPER_PARAMS_FILENAME = "lda.hyper_params"

    def __init__(self, num_topics, topic_prior = 0.1, word_prior = 0.01):
        self.num_topics = num_topics

        self.global_topic_hist = [0] * self.num_topics  # item fmt: N(z)
        self.word_topic_hist = {}  # item fmt: w -> N(w|z)

        self.hyper_params = HyperParams()
        self.hyper_params.topic_prior = topic_prior  # alpha, default symmetrical
        self.hyper_params.word_prior = word_prior  # beta, default symmetrical

    def save(self, model_dir):
        logging.info('Save lda model to %s.' % model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self._save_word_topic_hist(model_dir + "/" +
                self.__class__.WORD_TOPIC_HIST_FILENAME)
        self._save_global_topic_hist(model_dir + "/" +
                 self.__class__.GLOABLE_TOPIC_HIST_FILENAME)
        self._save_hyper_params(model_dir + "/" +
                 self.__class__.HYPER_PARAMS_FILENAME)

    def load(self, model_dir):
        logging.info('Load lda model from %s.' % model_dir)
        assert self._load_global_topic_hist(model_dir + "/" +
                self.__class__.GLOABLE_TOPIC_HIST_FILENAME)
        self.num_topics = len(self.global_topic_hist)
        assert self._load_word_topic_hist(model_dir + "/" +
                self.__class__.WORD_TOPIC_HIST_FILENAME)
        assert self._load_hyper_params(model_dir + "/" +
                self.__class__.HYPER_PARAMS_FILENAME)

    def _save_global_topic_hist(self, filename):
        fp = open(filename, 'wb')
        record_writer = RecordWriter(fp)
        global_topic_hist_pb = GlobalTopicHistogramPB()
        for topic_count in self.global_topic_hist:
            global_topic_hist_pb.topic_counts.append(topic_count)
        record_writer.write(global_topic_hist_pb.SerializeToString())
        fp.close()

    def _save_word_topic_hist(self, filename):
        fp = open(filename, 'wb')
        record_writer = RecordWriter(fp)
        for word, ordered_sparse_topic_hist in self.word_topic_hist.iteritems():
            word_topic_hist_pb = WordTopicHistogramPB()
            word_topic_hist_pb.word = word
            word_topic_hist_pb.sparse_topic_hist.ParseFromString(
                    ordered_sparse_topic_hist.serialize_to_string())
            record_writer.write(word_topic_hist_pb.SerializeToString())
        fp.close()

    def _save_hyper_params(self, filename):
        fp = open(filename, 'wb')
        record_writer = RecordWriter(fp)
        record_writer.write(self.hyper_params.serialize_to_string())
        fp.close()

    def _load_global_topic_hist(self, filename):
        logging.info('Loading global_topic_hist vector N(z).')
        self.global_topic_hist = []

        fp = open(filename, "rb")
        record_reader = RecordReader(fp)
        blob = record_reader.read()
        fp.close()
        if blob == None:
            logging.error('GlobalTopicHist is nil, file %s' % filename)
            return False

        global_topic_hist_pb = GlobalTopicHistogramPB()
        global_topic_hist_pb.ParseFromString(blob)
        for topic_count in global_topic_hist_pb.topic_counts:
            self.global_topic_hist.append(topic_count)
        return True

    def _load_word_topic_hist(self, filename):
        logging.info('Loading word_topic_hist matrix N(w|z).')
        self.word_topic_hist.clear()

        fp = open(filename, "rb")
        record_reader = RecordReader(fp)
        while True:
            blob = record_reader.read()
            if blob == None:
                break

            word_topic_hist_pb = WordTopicHistogramPB()
            word_topic_hist_pb.ParseFromString(blob)

            ordered_sparse_topic_hist = \
                    OrderedSparseTopicHistogram(self.num_topics)
            ordered_sparse_topic_hist.parse_from_string(
                    word_topic_hist_pb.sparse_topic_hist.SerializeToString())
            self.word_topic_hist[word_topic_hist_pb.word] = \
                    ordered_sparse_topic_hist
        fp.close()
        return (len(self.word_topic_hist) > 0)

    def _load_hyper_params(self, filename):
        logging.info('Loading hyper_params topic_prior and word_prior.')
        fp = open(filename, "rb")
        record_reader = RecordReader(fp)
        blob = record_reader.read()
        fp.close()
        if blob == None:
            logging.error('HyperParams is nil, file %s' % filename)
            return False

        self.hyper_params.parse_from_string(blob)
        return True

    def has_word(self, word):
        return word in self.word_topic_hist

    def get_word_topic_dist(self, vocab_size):
        """Returns topic-word distributions matrix p(w|z), indexed by word.
        """
        word_topic_dist = {}
        word_prior_sum = self.hyper_params.word_prior * vocab_size

        # TODO(fandywang): only cache sub-matrix p(w|z) of frequency words.
        for word_id, ordered_sparse_topic_hist in self.word_topic_hist.iteritems():
            dense_topic_dist = []
            for topic in xrange(self.num_topics):
                dense_topic_dist.append(self.hyper_params.word_prior /
                        (word_prior_sum + self.global_topic_hist[topic]))
            for non_zero in ordered_sparse_topic_hist.non_zeros:
                dense_topic_dist[non_zero.topic] = \
                        (self.hyper_params.word_prior + non_zero.count) / \
                        (word_prior_sum + self.global_topic_hist[non_zero.topic])
            word_topic_dist[word_id] = dense_topic_dist

        return word_topic_dist

    def __str__(self):
        """Outputs a human-readable representation of the model.
        """
        model_str = []
        model_str.append('NumTopics: %d' % self.num_topics)
        model_str.append('GlobalTopicHist: %s' % str(self.global_topic_hist))
        model_str.append('WordTopicHist: ')
        for word, ordered_sparse_topic_hist in self.word_topic_hist.iteritems():
            model_str.append('word: %d' % word)
            model_str.append('topic_hist: %s' % str(ordered_sparse_topic_hist))
        model_str.append('HyperParams: ')
        model_str.append(str(self.hyper_params))
        return '\n'.join(model_str)

