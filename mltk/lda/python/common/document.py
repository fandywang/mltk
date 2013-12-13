#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import random

from lda_pb2 import DocumentPB
from model import Model
from ordered_sparse_topic_histogram import OrderedSparseTopicHistogram
from vocabulary import Vocabulary

class Word(object):

    def __init__(self, id, topic):
        self.id = id
        self.topic = topic

    def __str__(self):
        return '<word_id: ' + str(self.id) + ', topic: ' + str(self.topic) + '>'


class Document(object):

    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.words = None  # word occurances of the document,
                           # item fmt: Word<id, topic>
        self.doc_topic_hist = None  # N(z|d)

    def parse_from_tokens(self, doc_tokens, rand, vocabulary, model = None):
        """Parse the text document from tokens. Only tokens in vocabulary
        and model will be considered.
        """
        self.words = []
        self.doc_topic_hist = OrderedSparseTopicHistogram(self.num_topics)

        for token in doc_tokens:
            word_index = vocabulary.word_index(token)
            if (word_index != -1 and
                    (model == None or model.has_word(word_index))):
                # initialize a random topic for current word
                topic = rand.randint(0, self.num_topics - 1)
                self.words.append(Word(word_index, topic))
                self.doc_topic_hist.increase_topic(topic, 1)

    def serialize_to_string(self):
        """Serialize document to DocumentPB string.
        """
        document_pb = DocumentPB()
        for word in self.words:
            word_pb = document_pb.words.add()
            word_pb.id = word.id
            word_pb.topic = word.topic
        return document_pb.SerializeToString()

    def parse_from_string(self, document_str):
        """Parse document from DocumentPB serialized string.
        """
        self.words = []
        self.doc_topic_hist = OrderedSparseTopicHistogram(self.num_topics)

        self.document_pb = DocumentPB()
        self.document_pb.ParseFromString(document_str)
        for word_pb in self.document_pb.words:
            self.words.append(Word(word_pb.id, word_pb.topic))
            self.increase_topic(word_pb.topic, 1)

    def num_words(self):
        return len(self.words)

    def get_words(self):
        for word in self.words:
            yield word

    def get_topic_count(self, topic):
        """Returns N(z|d).
        """
        return self.doc_topic_hist.count(topic)

    def increase_topic(self, topic, count = 1):
        """Adds count to current topic, and returns the updated count.
        """
        return self.doc_topic_hist.increase_topic(topic, count)

    def decrease_topic(self, topic, count = 1):
        """Subtracts count from current topic, and returns the updated count.
        """
        return self.doc_topic_hist.decrease_topic(topic, count)

    def __str__(self):
        """Outputs a human-readable representation of the model.
        """
        document_str = []
        for word in self.words:
            document_str.append(str(word))
        document_str.append(str(self.doc_topic_hist))
        return '\n'.join(document_str)

