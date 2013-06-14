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

import sys

sys.path.append('..')
from common.model import Model
from common.vocabulary import Vocabulary

class TopicWordsStat(object):
    """TopicWords implements topic words tools.
    """

    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary

    def save(self, topic_words_file, accumulated_prob_threshold):
        """Save the topic words to file.
        """
        fp = open(topic_words_file, 'w')
        fp.write(self.get_topic_top_words(accumulated_prob_threshold))
        fp.close()

    def get_topic_top_words(self, accumulated_prob_threshold):
        """Returns topics' top words.
        """
        topic_top_words = []
        sparse_topic_word_dist = self.compute_topic_word_distribution()

        for topic, word_probs in enumerate(sparse_topic_word_dist):
            top_words = []
            top_words.append(str(topic))
            top_words.append(str(self.model.global_topic_hist[topic]))
            accumulated_prob = 0.0
            for word_prob in word_probs:
                top_words.append(
                        self.vocabulary.word(word_prob[0]).encode('gbk', 'ignore'))
                top_words.append(str(word_prob[1]))
                accumulated_prob += word_prob[1]
                if accumulated_prob > accumulated_prob_threshold:
                    break
            topic_top_words.append('\t'.join(top_words))

        return '\n'.join(topic_top_words)

    def compute_topic_word_distribution(self):
        """Compute the topic word distribution p(w|z), indexed by topic z.
        """
        # item fmt: z -> <w, p(w|z)>
        sparse_topic_word_dist = []
        for topic in xrange(self.model.num_topics):
            sparse_topic_word_dist.append([])

        for word_id, ordered_sparse_topic_hist in \
                self.model.word_topic_hist.iteritems():
            for non_zero in ordered_sparse_topic_hist.non_zeros:
                sparse_topic_word_dist[non_zero.topic].append(
                        [word_id,
                        (non_zero.count + self.model.hyper_params.word_prior) /
                        (self.model.hyper_params.word_prior * self.vocabulary.size() +
                        self.model.global_topic_hist[non_zero.topic])])

        for topic, word_probs in enumerate(sparse_topic_word_dist):
            word_probs.sort(cmp=lambda x,y:cmp(x[1], y[1]), reverse=True)

        return sparse_topic_word_dist
