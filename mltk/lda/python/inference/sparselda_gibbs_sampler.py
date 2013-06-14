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
import random
import sys

sys.path.append('..')
from common.document import Document
from common.model import Model
from common.vocabulary import Vocabulary

class SparseLDAGibbsSampler(object):
    """SparseLDAGibbsSampler implements the SparseLDA gibbs sampling inference
    algorithm.  In gibbs sampling formula:

        (0) p(z|w) --> p(z|d) * p(w|z) --> [alpha(z) + N(z|d)] * p(w|z)

        (1) s(z, w) = alpha(z) * p(w|z)
        (2) r(z, w, d) = N(z|d) * p(w|z)
        (3) p(w|z) = [beta + N(w|z)] / [beta * V + N(z)]

    The process divides the full sampling mass p(z|w) into two buckets,
    where s(z, w) is a smoothing-only bucket, and r(z, w, d) is a
    document-topic bucket.

    To achieve time efficiency, topic distributions matrix p(w|z) are
    pre-computed and cached.

    See 'Limin Yao, David Mimno, Andrew McCallum. Efficient methods for topic
    model inference on streaming document collections, In SIGKDD, 2009.' for
    more details.
    """

    def __init__(self, model, vocabulary, total_iterations,
            burn_in_iterations):
        self.model = model
        self.vocabulary = vocabulary
        self.total_iterations = total_iterations
        self.burn_in_iterations = burn_in_iterations

        # cache p(w|z), indexed by word.
        # item fmt: word -> dense topic distribution.
        # TODO(fandywang): only cache the submatrix p(w|z) of frequent words
        # for time and memory efficieny, in other words, high scalability.
        self.word_topic_dist = self.model.get_word_topic_dist(vocabulary.size())
        self.smoothing_only_sum = {}  # cache s(z, w)
        self.__init_smoothing_only_sum()

    def __init_smoothing_only_sum(self):
        """Compute and cahce the smoothing_only_sum s(z, w).
        """
        for word_id in self.model.word_topic_hist.keys():
            cur_sum = 0.0
            topic_dist = self.word_topic_dist[word_id]
            for topic, prob in enumerate(topic_dist):
                cur_sum += self.model.hyper_params.topic_prior * prob
            self.smoothing_only_sum[word_id] = cur_sum

    def infer_topics(self, doc_tokens):
        """Inference topics embedded in the given document, which represents as
        a token sequence named 'doc_tokens'.

        Returns the dict of topics sorted by their probabilities p(z|d), such as
        {1 : 0.87, 6 : 0.23, 4: 0.17, 15 : 0.1}
        """
        rand = random.Random()
        rand.seed(hash(str(doc_tokens)))

        doc_topic_dist = self._inference_one_chain(doc_tokens, rand)
        sorted(doc_topic_dist.items(), lambda x, y: cmp(x[1], y[1]),
                reverse = True)
        return doc_topic_dist

    # TODO(fandywang): infer topic words later.
    def infer_topic_words(self, doc_tokens):
        """Inference topic words embedded in the given document, which
        represents as a token sequence named 'doc_tokens'.

        Returns the dict of topic words sorted by their probabilities p(w|d) =
        p(z|d)*p(w|z),
        such as {'apple' : 0.87, 'iphone' : 0.23, 'ipad': 0.17, 'nokia' : 0.1}
        """
        doc_topic_words_dist = {}
        doc_topic_dist = self.infer_topics(doc_tokens)
        # cahce the p(w|z), indexd by topic.
        for topic, prob in doc_topic_dist:
            pass
        return doc_topic_words_dist

    def _inference_one_chain(self, doc_tokens, rand):
        """Inference topics with one markov chain.

        Returns the sparse topics p(z|d).
        """
        document = Document(self.model.num_topics)
        document.parse_from_tokens(doc_tokens, rand,
                self.vocabulary, self.model)
        if document.num_words() == 0:
            return dict()

        accumulated_topic_hist = {}
        for i in xrange(self.total_iterations):
            # one iteration
            for word in document.words:
                # --
                document.decrease_topic(word.topic, 1)

                new_topic = self._sample_word_topic(document, word.id, rand)
                assert new_topic != None
                word.topic = new_topic
                # ++
                document.increase_topic(new_topic, 1)

            if i >= self.burn_in_iterations:
                for non_zero in document.doc_topic_hist.non_zeros:
                    if non_zero.topic in accumulated_topic_hist:
                        accumulated_topic_hist[non_zero.topic] += non_zero.count
                    else:
                        accumulated_topic_hist[non_zero.topic] = non_zero.count

        topic_dist = self._l1normalize_distribution(accumulated_topic_hist)
        return topic_dist

    def _sample_word_topic(self, doc, word, rand):
        """Sampling a new topic for current word.

        Returns the new topic.
        """
        doc_topic_bucket, doc_topic_sum = \
                self._compute_doc_topic_bucket(doc, word)

        total_mass = self.smoothing_only_sum[word] + doc_topic_sum
        sample = rand.uniform(0.0, total_mass)

        # sample in document topic bucket
        if sample < doc_topic_sum:
            for topic_prob in doc_topic_bucket:
                sample -= topic_prob[1]
                if sample <= 0:
                    return topic_prob[0]
        else:  # sample in smoothing only bucket
            sample -= doc_topic_sum
            topic_dist = self.word_topic_dist[word]
            for topic, prob in enumerate(topic_dist):
                sample -= prob
                if sample <= 0:
                    return topic
        logging.error('sample word topic error, sample: %f, dist_sum: %f.'
                % (sample, dist_sum))
        return None

    def _compute_doc_topic_bucket(self, doc, word):
        """Compute the document topic bucket r(z, w, d).

        Returns document-topic distributions and their sum.
        """
        doc_topic_bucket = []
        doc_topic_sum = 0.0

        dense_topic_dist = self.word_topic_dist[word]
        for non_zero in doc.doc_topic_hist.non_zeros:
            doc_topic_bucket.append([non_zero.topic,
                    non_zero.count * dense_topic_dist[non_zero.topic]])
            doc_topic_sum += doc_topic_bucket[-1][1]
        return doc_topic_bucket, doc_topic_sum

    def _l1normalize_distribution(self, topic_dict):
        """Returns the l1-normalized topic distributions.
        """
        topic_dist = {}
        weight_sum = 0
        for topic, weight in topic_dict.iteritems():
            weight_sum += weight
        if weight_sum == 0:
            logging.warning('The sum of topic weight is zero.')
            return topic_dist
        for topic, weight in topic_dict.iteritems():
            topic_dist[topic] = float(weight) / weight_sum
        return topic_dist

