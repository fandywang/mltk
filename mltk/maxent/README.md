MLTK.MaxEnt
==========================
Maximum Entropy Modelling.

### Features
1. supporting real-valued features
2. supporting three effective parameter estimation methods, including SGD, LBFGF and OWLQN
3. supporting a simple feature selection (feature cutoff)

Usage
---------------------
### 1. Data format
The corpus for training/test the model have the line format as follows:

    [instance1]
    [instance2]
    ...
    [instanceM]

in which each line is one instance. [instancei] is the ith instane of the dataset that consists of a list of Ni features.

    [instancei] = class\t[feature1:value1]\t[feature2:value2]\t...\t[featureNi:valueNi]

in which all [featurej:valuej] &lt; j=1...Ni&gt; are feature-value pairs and they are separated by the tab character.

### 2. Training
#### Command line

        Usage: ./bin/maxent_trainer [options].
        --helpshort  show this help message and exit
        --train_data_file (the filename of training data.) type: string default: "" 
        --model_file (the filename of maxent model.) type: string default: ""
        --optim_method (the optimization method, LBFGS, OWLQN, or SGD.) type: string default: "LBFGS"
        --l1_reg (the L1 regularization.) type: double default: 0 
        --l2_reg (the L2 regularization.) type: double default: 0 
        --num_iterations (the total iterations.) type: int32 default: 100
        --newton_m (the cache size for newton methods, OWLQN and LBFGS.) type: int32  default: 10
        --sgd_learning_rate (the learning rate of SGD.) type: int32 default: 1
        --num_heldout (the number of heldout data.) type: int32 default: 0
        --feature_cutoff (the minmum frequency of feature.) type: int32 default: 1 

References
---------------------
1. 李航. [《统计学习方法》](http://book.douban.com/subject/10590856/). 2012.
2. Adam L. Berger, Stephen A. Della Pietra, and Vincent J. Della Pietra. 1996.
[A maximum entropy approach to natural language processing](http://acl.ldc.upenn.edu/J/J96/J96-1002.pdf).
Computational Linguistics.
3. Yoshimasa Tsuruoka, Jun'ichi Tsujii, and Sophia Ananiadou. 2009. [Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty](http://www.aclweb.org/anthology-new/P/P09/P09-1054.pdf), In Proceedings of ACL-IJCNLP.
4. Galen Andrew, Jianfeng Gao. 2007. [Scalable Training of L1-Regularized Log-Linear Models](http://www.machinelearning.org/proceedings/icml2007/papers/449.pdf). ICML.
5. [Michael Collins](http://www.cs.columbia.edu/~mcollins/). [Log-linear Models](http://www.cs.columbia.edu/~mcollins/loglinear.pdf). Tech notes.
6. [Michael Collins](http://www.cs.columbia.edu/~mcollins/). [Log-linear Models, MEMMs, and CRFs](http://www.cs.columbia.edu/~mcollins/crf.pdf). Tech notes.


Copyright and license
---------------------
Copyright (C) 2013 MLTK Project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this work except in compliance with the License.
You may obtain a copy of the License in the LICENSE file, or at:

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
