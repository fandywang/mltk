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
import optparse
import os
import random

from common.model import Model
from common.vocabulary import Vocabulary
from training.sparselda_train_gibbs_sampler import SparseLDATrainGibbsSampler
from training.model_evaluator import ModelEvaluator
from training.topic_words_stat import TopicWordsStat

def main(args):
    model = Model(args.num_topics, args.topic_prior, args.word_prior)
    vocabulary = Vocabulary()
    vocabulary.load(args.vocabulary_file)
    sparselda_train_gibbs_sampler = SparseLDATrainGibbsSampler(
            model, vocabulary)
    sparselda_train_gibbs_sampler.load_corpus(args.corpus_dir)

    rand = random.Random()

    for i in xrange(args.total_iterations):
        logging.info('sparselda trainer, gibbs sampling iteration %d.'
                % (i + 1))
        sparselda_train_gibbs_sampler.gibbs_sampling(rand)

        # dump lda model
        if i == 0 or (i + 1) % args.save_model_interval == 0:
            logging.info('iteration %d start saving lda model.' % (i + 1))
            sparselda_train_gibbs_sampler.save_model(args.model_dir, i + 1)
            topic_words_stat = TopicWordsStat(model, vocabulary)
            topic_words_stat.save(
                    '%s/topic_top_words.%d' % (args.model_dir, i + 1),
                    args.topic_word_accumulated_prob_threshold)
            logging.info('iteration %d save lda model ok.' % (i + 1))

        # dump checkpoint
        if i == 0 or (i + 1) % args.save_checkpoint_interval == 0:
            logging.info('iteration %d start saving checkpoint.' % (i + 1))
            sparselda_train_gibbs_sampler.save_checkpoint(
                    args.checkpoint_dir, i + 1)
            logging.info('iteration %d save checkpoint ok.' % (i + 1))

        # compute the loglikelihood
        if i == 0 or (i + 1) % args.compute_loglikelihood_interval == 0:
            logging.info('iteration %d start computing loglikelihood.' % (i + 1))
            model_evaluator = ModelEvaluator(model, vocabulary)
            ll = model_evaluator.compute_loglikelihood(
                    sparselda_train_gibbs_sampler.documents)
            logging.info('iteration %d loglikelihood is %f.' % (i + 1, ll))

if __name__ == '__main__':
    parser = optparse.OptionParser('usage: python lda_trainer.py -h.')
    parser.add_option('--corpus_dir',
            help = 'the corpus directory, line fmt: w1 \t w2 \t w3 ... .')
    parser.add_option('--vocabulary_file',
            help = 'the vocabulary file, line fmt: w [\tfreq].')
    parser.add_option('--num_topics', type = int, help = 'the num of topics.')
    parser.add_option('--topic_prior', type = float, default = 0.1,
            help = 'the topic prior alpha.')
    parser.add_option('--word_prior', type = float, default = 0.01,
            help = 'the word prior beta.')
    parser.add_option('--total_iterations', type = int, default = 10000,
            help = 'the total iteration.')
    parser.add_option('--model_dir', help = 'the model directory.')
    parser.add_option('--save_model_interval', type = int, default = 100,
            help = 'the interval to save lda model.')
    parser.add_option('--topic_word_accumulated_prob_threshold',
            type = float, default = 0.5,
            help = 'the accumulated_prob_threshold of topic words.')
    parser.add_option('--save_checkpoint_interval', type = int, default = 10,
            help = 'the interval to save checkpoint.')
    parser.add_option('--checkpoint_dir', help = 'the checkpoint directory.')
    parser.add_option('--compute_loglikelihood_interval', type = int,
            default = 10, help = 'the interval to compute loglikelihood.')

    (options, args) = parser.parse_args()
    logging.basicConfig(filename = os.path.join(os.getcwd(), 'log.txt'),
            level = logging.DEBUG, filemode = 'a',
            format = '%(asctime)s - %(levelname)s: %(message)s')
    logging.info('Parameters : %s' % str(options))

    main(options)
