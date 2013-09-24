#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

import logging
import optparse

from common.model import Model
from common.vocabulary import Vocabulary
from inference.sparselda_gibbs_sampler import SparseLDAGibbsSampler

def main(args):
    model = Model(0)
    model.load(args.model_dir)
    vocabulary = Vocabulary()
    vocabulary.load(args.vocabulary)
    multi_chain_gibbs_sampler = MultiChainGibbsSampler(model, vocabulary,
            args.num_markov_chains, args.total_iterations,
            args.burn_in_iterations)

    fp = open(args.documents, 'r')
    for doc_str in fp.readlines():
        doc_str = doc_str.decode('gbk')
        doc_tokens = doc_str.strip().split('\t')
        topic_dist = multi_chain_gibbs_sampler.infer_topics(doc_tokens)
        print doc_str
        print topic_dist
    fp.close()

if __name__ == '__main__':
    parser = optparse.OptionParser('usage: python lda_inference.py -h.')
    parser.add_option('--model_dir', help = 'the lda model directory.')
    parser.add_option('--vocabulary_file', help = 'the vocabulary file.')
    parser.add_option('--document_file_file',
            help = 'the document file in gbk, line fmt: w1 \t w2 \t w3 \t... .')
    parser.add_option('--num_markov_chains', type = int, default = 5,
            help = 'the num of markov chains.')
    parser.add_option('--total_iterations', type = int, default = 50,
            help = 'the num of total_iterations.')
    parser.add_option('--burn_in_iterations', type = int, default = 20,
            help = 'the num of burn_in iteration.')

    (options, args) = parser.parse_args()
    logging.info('Parameters : %s' % str(options))

    main(options)

