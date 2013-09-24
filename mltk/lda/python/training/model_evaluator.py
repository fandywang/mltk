#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import logging
import math
import multiprocessing
import numpy as np
import os
import random
import sys

sys.path.append('..')
from common.document import Document
from common.model import Model
from common.vocabulary import Vocabulary

"""ModelEvaluator implements the evaluation method of lda model's quality.
"""
g_model = None
g_vocabulary = None
g_word_topic_dist = None

def compute_doc_topic_distribution(document):
    denominator = (g_model.hyper_params.topic_prior * g_model.num_topics + document.num_words())
    default_topic_prob = g_model.hyper_params.topic_prior / denominator
    dense_topic_dist = np.array([default_topic_prob] * g_model.num_topics, dtype = 'float64')
    for non_zero in document.doc_topic_hist.get_non_zeros():
        dense_topic_dist[non_zero.topic] = \
                (g_model.hyper_params.topic_prior + non_zero.count) / denominator

    return dense_topic_dist

def compute_doc_loglikelihood(document):
    doc_dense_topic_dist = compute_doc_topic_distribution(document)
    doc_loglikelihood = 0.0

    for word in document.get_words():
        word_topic_dist = g_word_topic_dist.get(word.id)
        if word_topic_dist is None:
            continue
        word_prob_sum = sum(word_topic_dist * doc_dense_topic_dist)
        doc_loglikelihood += math.log(word_prob_sum)

    return doc_loglikelihood

def compute_loglikelihood(model, vocabulary, documents):
    """Compute and return the loglikelihood of documents.

    p(D|M) = p(d1)p(d2)...

    p(d) = p(w1)p(w2)...
         = sum_z {p(z|d)p(w1|z)} * sum_z {p(z|d)p(w2|z)} * ...

    log(p(d)) = log(sum_z p(z|d)p(w1|z)) + log(sum_z p(z|d)p(w2|z)) + ...

    p(D|M) -> log(p(D|M)) = log(p(d1)) + log(p(d2)) + ...
    """
    # cache matrix p(w|z), indexed by word.
    global g_model, g_vocabulary, g_word_topic_dist
    g_model = model
    g_vocabulary = vocabulary
    g_word_topic_dist = model.get_word_topic_dist(g_vocabulary.size())

    loglikelihood = 0.0
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count() * 2)
    it = pool.imap(compute_doc_loglikelihood, documents)
    for document in documents:
        loglikelihood += it.next()
    return loglikelihood

