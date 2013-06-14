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
import sys

sys.path.append('..')
from common.document import Document
from common.ordered_sparse_topic_histogram import OrderedSparseTopicHistogram
from common.model import Model
from common.vocabulary import Vocabulary
from sparselda_gibbs_sampler import SparseLDAGibbsSampler

class MultiChainGibbsSampler(SparseLDAGibbsSampler):
    """MultiChainGibbsSampler implements multi-markov-chain based SparseLDA
    gibbs sampling inference algorithm.

    See 'X. Wei, W. Bruce Croft. LDA-based document models for ad hoc retrieval.
    In Proc. SIGIR. 2006.' for more details.
    """

    def __init__(self, model, vocabulary, num_markov_chains,
            total_iterations, burn_in_iterations):
        super(MultiChainGibbsSampler, self).__init__(model, vocabulary,
                total_iterations, burn_in_iterations)
        self.num_markov_chains = num_markov_chains

    def infer_topics(self, doc_tokens):
        """Inference topics embedded in the given document, which represents as
        a token sequence named 'doc_tokens'.

        Returns the sparse topics sorted by their probabilities p(z|d),
        such as {'apple' : 0.87, 'iphone' : 0.23, 'ipad': 0.17, 'nokia' : 0.1}
        """
        rand = random.Random()
        rand.seed(hash(str(doc_tokens)))

        accumulated_topic_dist = {}
        for i in range(0, self.num_markov_chains):
            topic_dist = self._inference_one_chain(doc_tokens, rand)

            for topic, prob in topic_dist.iteritems():
                if topic in accumulated_topic_dist:
                    accumulated_topic_dist[topic] += prob
                else:
                    accumulated_topic_dist[topic] = prob

        topic_dist = self._l1normalize_distribution(accumulated_topic_dist)
        sorted(topic_dist.items(), lambda x, y: cmp(x[1], y[1]), reverse = True)
        return topic_dist

