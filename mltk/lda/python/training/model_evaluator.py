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
import math
import os
import random
import sys

sys.path.append('..')
from common.document import Document
from common.model import Model
from common.vocabulary import Vocabulary

class ModelEvaluator(object):
    """ModelEvaluator implements the evaluation method of lda model's quality.
    """

    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary

        # cache matrix p(w|z), indexed by word.
        self.word_topic_dist = \
                self.model.get_word_topic_dist(self.vocabulary.size())

    def compute_loglikelihood(self, documents):
        """Compute and return the loglikelihood of documents.

        p(D|M) = p(d1)p(d2)...

        p(d) = p(w1)p(w2)...
             = sum_z {p(z|d)p(w1|z)} * sum_z {p(z|d)p(w2|z)} * ...

        log(p(d)) = log(sum_z p(z|d)p(w1|z)) + log(sum_z p(z|d)p(w2|z)) + ...

        p(D|M) -> log(p(D|M)) = log(p(d1)) + log(p(d2)) + ...
        """
        loglikelihood = 0.0
        for document in documents:
            doc_dense_topic_dist = \
                    self._compute_doc_topic_distribution(document)
            doc_loglikelihood = 0.0
            for word in document.words:
                word_topic_dist = self.word_topic_dist.get(word.id)
                if word_topic_dist is None:
                    continue
                word_prob_sum = 0.0
                for topic, prob in enumerate(word_topic_dist):
                    word_prob_sum += prob * doc_dense_topic_dist[topic]
                doc_loglikelihood += math.log(word_prob_sum)
            loglikelihood += doc_loglikelihood
        return loglikelihood

    def _compute_doc_topic_distribution(self, document):
        dense_topic_dist = []
        denominator = (self.model.hyper_params.topic_prior
                * self.model.num_topics + document.num_words())
        for i in xrange(self.model.num_topics):
            dense_topic_dist.append(
                    self.model.hyper_params.topic_prior / denominator)
        for non_zero in document.doc_topic_hist.non_zeros:
            dense_topic_dist[non_zero.topic] = \
                    ((self.model.hyper_params.topic_prior + non_zero.count)
                    / denominator)

        return dense_topic_dist

