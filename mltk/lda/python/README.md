## python-sparselda
================
python-sparselda is a Latent Dirichlet Allocation(LDA) topic modeling package based on SparseLDA Gibbs Sampling inference algorithm, and written in Python 2.6 or newer, Python 3.0 or newer excluded.

Frankly, python-sparselda is just a mini-project, we hope it can help you better understand the standard LDA and SparseLDA algorithms. RTFSC for more details. Have fun.

Please use the github issue tracker for python-sparselda at:
https://github.com/fandywang/mltk/tree/master/mltk/lda/python.

## Members
* [wangkuiyi](https://github.com/wangkuiyi)
* [xueminzhao](https://github.com/xmzhao)
* [richardsun](https://github.com/richardsun)
* [yhcharles](https://github.com/yhcharles)
* [fandywang](https://github.com/fandywang)
* [zhihuijin](https://github.com/zhihuijin)
* [ubiwang](https://github.com/ubiwang)

## Usage
================
### 1. Install Google Protocol Buffers
python-sparselda serialize and persistent store the lda model and checkpoint based on protobuf, so you should install it first.

    wget https://protobuf.googlecode.com/files/protobuf-2.5.0.tar.bz2
    tar -zxvf protobuf-2.5.0.tar.bz2
    cd protobuf-2.5.0
    ./configure
    make
    sudo make install
    cd python
    python ./setup.py build
    sudo python ./setup.py install

    cd python-sparselda/common
    protoc -I=. --python_out=. lda.proto

### 2. Training
#### 2.1 Command line
    Usage: python lda_trainer.py [options].

    Options:
    -h, --help   show this help message and exit
    --corpus_dir=CORPUS_DIR
            the corpus directory.
    --vocabulary_file=VOCABULARY_FILE
            the vocabulary file.
    --num_topics=NUM_TOPICS
            the num of topics.
    --topic_prior=TOPIC_PRIOR
            the topic prior alpha (50 / num_topics).
    --word_prior=WORD_PRIOR
            the word prior beta (0.01).
    --total_iterations=TOTAL_ITERATIONS
            the total iteration.
    --model_dir=MODEL_DIR
            the model directory.
    --save_model_interval=SAVE_MODEL_INTERVAL
            the interval iterations to save lda model.
    --topic_word_accumulated_prob_threshold=TOPIC_WORD_ACCUMULATED_PROB_THRESHOLD
            the accumulated_prob_threshold of topic top words.
    --save_checkpoint_interval=SAVE_CHECKPOINT_INTERVAL
            the interval to save checkpoint.
    --checkpoint_dir=CHECKPOINT_DIR
            the checkpoint directory.
    --compute_loglikelihood_interval=COMPUTE_LOGLIKELIHOOD_INTERVAL
            the interval to compute loglikelihood.

#### 2.2 Input corpus format
The corpus for training/estimating the model have the line format as follows:

    [document1]
    [document2]
    ...
    [documentM]

in which each line is one document. [documenti] is the ith document of the dataset that consists of a list of Ni words/terms.

    [documenti] = [wordi1]\t[wordi2]\t...\t[wordiNi]

in which all [wordij] &lt;i=1...M, j=1...Ni&gt; are text strings and they are separated by the tab character.

**Note that** the terms document and word here are abstract and should not only be understood as normal text documents.
This's because LDA can be used to discover the underlying topic structures of any kind of discrete data. Therefore,
python-sparselda is not limited to text and natural language processing but can also be applied to other kinds of data
like images.

Also, keep in mind that for text/Web data collections, you should first preprocess the data (e.g., word segment,
removing stopwords and rare words, stemming, etc.) before estimating with python-sparselda.

#### 2.3 Input vocabulary format
The vocabulary for training/estimating the model have the line format as follows:

    [word1]
    [word2]
    ...
    [wordV]

in which each line is a unique word. Words only appear in vocabulary will be considered for parameter estimation.

#### 2.4 Outputs
##### 1) LDA Model
It includs three files.
* lda.topic_word_hist: This file contains the word-topic histograms, i.e., N(word|topic).
* lda.global_topic_hist: This file contains the global topic histogram, i.e., N(topic).
* lda.hyper_params: This file contails the hyperparams, i.e., alpha and beta.

##### 2) Checkpoint
Every `--save_checkpoint_interval` iterations, the lda_trainer will dump current checkpoint for fault tolerance.
The checkpoint mainly includes two types files.
* LDA Model: See above.
* Corpus: This directory contains serialized documents.

##### 3) Topic words
* lda.topic_words: This file contains most likely words of each topic. The number of topic top words is depend on `--topic_word_accumulated_prob_threshold`.

### 3. Inference
Please refer the example: lda_inferencer.py.

**Note that** we strongly recommend you to use `MultiChainGibbsSampler` class for trade off between efficiency and effectiveness.

### 4. Evaluation
Instead of manual evaluation, we want to evaluate topics quality automatically, and filter out a few meaningless topics to enchance the inference effect.

## TODO
================
1. Hyperparameters optimization.
2. Memory optimization.
3. Performance optimization, such as using NumPy.
4. Data and model parallelization.

## Credit
1. python-sparselda is mainly inspired by Yi Wang's [PLDA](http://plda.googlecode.com/files/aaim.pdf) and Limin Yao's [SparseLDA](https://people.cs.umass.edu/~mimno/papers/fast-topic-model.pdf).
2. The code design comes from team work.

## References
================
1. Blei, A. Ng, and M. Jordan. [Latent Dirichlet allocation](http://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf). Journal of Machine Learning Research, 2003.
2. Gregor Heinrich. [Parameter estimation for text analysis](http://www.arbylon.net/publications/text-est.pdf). Technical Note, 2004.
3. Griﬃths, T. L., & Steyvers, M. [Finding scientiﬁc topics](http://www.pnas.org/content/101/suppl.1/5228.full.pdf). Proceedings of the National Academy of Sciences(PNAS), 2004.
4. I. Porteous, D. Newman, A. Ihler, A. Asuncion, P. Smyth, and M. Welling. [Fast collapsed Gibbs sampling for latent Dirichlet allocation](http://www.ics.uci.edu/~asuncion/pubs/KDD_08.pdf). In SIGKDD, 2008.
5. Limin Yao, David Mimno, Andrew McCallum. [Efficient methods for topic model inference on streaming document collections](https://people.cs.umass.edu/~mimno/papers/fast-topic-model.pdf), In SIGKDD, 2009.
6. Newman et al. [Distributed Inference for Latent Dirichlet Allocation](http://www.csee.ogi.edu/~zak/cs506-pslc/dist_lda.pdf), NIPS 2007.
7. Rickjin, [LDA 数学八卦](http://vdisk.weibo.com/s/q0sGh/1360334108?utm_source=weibolife). Technical Note, 2013.
8. X. Wei, W. Bruce Croft. [LDA-based document models for ad hoc retrieval](http://www.bradblock.com/LDA_Based_Document_Models_for_Ad_hoc_Retrieval.pdf). In Proc. SIGIR. 2006.
9. Yi Wang, Hongjie Bai, Matt Stanton, Wen-Yen Chen, and Edward Y. Chang. [PLDA: Parallel Latent Dirichlet Allocation for Large-scale Applications](http://plda.googlecode.com/files/aaim.pdf). AAIM 2009.

## Links
===============
Here are some pointers to other implementations of LDA.

1. [LDA-C](http://www.cs.princeton.edu/~blei/lda-c/index.html): A C implementation of variational EM for latent Dirichlet allocation (LDA), a topic model for text or other discrete data.
2. [GibbsLDA++](http://gibbslda.sourceforge.net/): A C/C++ implementation of Latent Dirichlet Allocation (LDA) using Gibbs Sampling technique for parameter estimation and inference.
3. [plda/plda+](https://code.google.com/p/plda/): A parallel C++ implementation of Latent Dirichlet Allocation (LDA).
4. [Mr. LDA](https://github.com/lintool/Mr.LDA): A Latent Dirichlet Allocation topic modeling package based on Variational Bayesian learning approach using MapReduce and Hadoop, developed by a Cloud Computing Research Team in University of Maryland, College Park.
5. [Yahoo_LDA](https://github.com/sudar/Yahoo_LDA): Y!LDA Topic Modelling Framework, it provides a fast C++ implementation of the inferencing algorithm which can use both multi-core parallelism and multi-machine parallelism using a hadoop cluster. It can infer about a thousand topics on a million document corpus while running for a thousand iterations on an eight core machine in one day.
6. [Mahout](https://cwiki.apache.org/confluence/display/MAHOUT/Latent+Dirichlet+Allocation): Mahout's goal is to build scalable machine learning libraries.
7. [MALLET ](http://mallet.cs.umass.edu/): A Java-based package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text.
8. [ompi-lda](https://code.google.com/p/ompi-lda/): OpenMP and MPI Based Paralllel Implementation of LDA.
9. [lda-go](https://code.google.com/p/lda-go/): Gibbs sampling training and inference of the Latent Dirichlet Allocation model written in Google's Go programming language.
10. [Matlab Topic Modeling Toolbox](http://psiexp.ss.uci.edu/research/programs_data/toolbox.htm)
11. [lda-j](http://www.arbylon.net/projects/): Java version of LDA-C and a short Java version of Gibbs Sampling for LDA.

## Copyright and license
==============================
Copyright(c) 2013 python-sparselda project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this work except in compliance with the License.
You may obtain a copy of the License in the LICENSE file, or at:

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
