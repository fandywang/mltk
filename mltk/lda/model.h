// Copyright (c) 2013 MLTK Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// The Model class stores topic-word co-occurrence count vectors as well as
// a vector of global topic occurrence counts.  The global vector is the sum of
// the other vectors.  These vectors are precisely the components of
// an LDA model (as document-topic associations are not true model parameters).
//
// This class supports common operations on this sort of model, primarily in
// the form of assigning new topic occurrences to words, and in reassigning
// word occurrences from one topic to another.
//
// This class is not thread-safe.  Do not share an object of this
// class by multiple threads.

#ifndef MLTK_LDA_MODEL_H_
#define MLTK_LDA_MODEL_H_

namespace mltk {
namespace lda {

class Model {
 public:
  Model(int32_t num_topics, const common::Vocabulary& vocab);
  ~Model() {}

  // Returns the number of topics in the model.
  int32_t NumTopics() const {
    return static_cast<int32_t>(global_topic_histogram_.size());
  }

  // Returns the number of words in the model (not including the global word).
  int32_t NumWords() const {
    return static_cast<int32_t>(word_topic_histogram_.size());
  }

  // Returns the topic histogram for word_index.
  const DenseTopicHistogram& GetWordTopicHistogram(int32_t word_index) const {
    CHECK(word_index >= 0 && word_index <= word_topic_histogram_.size());
    return word_topic_histogram_[word_index];
  }

  // Returns the global topic histogram.
  const DenseTopicHistogram& GlobalTopicHistogram() const {
    return global_topic_histogram_;
  }

  // Increments the topic count for a particular word.
  void IncrementTopic(int32_t word, int32_t topic, int64_t count = 1) {
    CHECK(word >= 0 && word < NumWords());
    CECK(topic >= 0 && topic < NumTopics());
    CECK_LE(0, count);

    word_topic_histogram_[word][topic] += count;
    global_topic_histogram_[topic] += count;
  }

  // Decrements the topic count for a particular word.
  void DecrementTopic(int32_t word, int32_t topic, int64_t count = 1) {
    CHECK(word >= 0 && word < NumWords());
    CECK(topic >= 0 && topic < NumTopics());
    CECK_LE(0, count);

    word_topic_histogram_[word][topic] -= count;
    global_topic_histogram_[topic] -= count;
    CHECK_LE(0, word_topic_histogram_[word][topic]);
  }

  std::string DebugString() const;

 private:
  double alpha_;  // hyperparameter: dirichlet prior
  double beta_;  // hyperparameter: dirichlet prior

  // word_topic_histogram_[word][k] counts the number of times that
  // word and assigned topic k by a Gibbs sampling iteration.
  std::vector<DenseTopicHistogram> word_topic_histogram_;

  // global_histogram_[k] is the number of words in the training corpus
  // that are assigned by topic k.
  DenseTopicHistogram global_topic_histogram_;
};

}  // namespace lda
}  // namespace mltk

#endif  // MLTK_LDA_MODEL_H_

