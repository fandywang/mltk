// Copyright (c) 2013 MaxEnt Project.
// Author: Lifeng Wang (ofandywang@gmail.com)
//
// STL DoubleVector Warapper and its utils.

#ifndef MLTK_MAXENT_MATH_VECTOR_H_
#define MLTK_MAXENT_MATH_VECTOR_H_

#include <assert.h>
#include <iostream>
#include <vector>

namespace mltk {
namespace maxent {

class DoubleVector {
 public:
  DoubleVector() {}
  explicit DoubleVector(const size_t n) { vec_.resize(n, 0); }
  DoubleVector(const size_t n, const double val) { vec_.resize(n, val); }
  explicit DoubleVector(const std::vector<double>& vec) : vec_(vec) {}
  ~DoubleVector() {}

  const std::vector<double>& STLVector() const { return vec_; }
  std::vector<double>& STLVector() { return vec_; }

  size_t Size() const { return vec_.size(); }

  double& operator[](int32_t i) { return vec_[i]; }
  const double& operator[](int32_t i) const { return vec_[i]; }

  DoubleVector& operator+=(const DoubleVector& b) {
    assert(b.Size() == vec_.size());
    for (size_t i = 0; i < vec_.size(); ++i) {
      vec_[i] += b[i];
    }
    return *this;
  }

  DoubleVector& operator*=(const double c) {
    for (size_t i = 0; i < vec_.size(); ++i) {
      vec_[i] *= c;
    }
    return *this;
  }

  void Project(const DoubleVector& y) {
    for (size_t i = 0; i < vec_.size(); ++i) {
      // if (sign(vec_[i]) != sign(y[i])) vec_[i] = 0;
      if (vec_[i] * y[i] <= 0) vec_[i] = 0;
    }
  }

 private:
  std::vector<double> vec_;
};

inline double DotProduct(const DoubleVector& a, const DoubleVector& b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.Size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline std::ostream& operator<<(std::ostream& stream, const DoubleVector& a) {
  stream << "(";
  for (size_t i = 0; i < a.Size(); ++i) {
    if (i != 0) stream << ", ";
    stream << a[i];
  }
  stream << ")";
  return stream;
}

inline const DoubleVector operator+(const DoubleVector& a,
                                    const DoubleVector& b) {
  DoubleVector v(a.Size());
  assert(a.Size() == b.Size());
  for (size_t i = 0; i < a.Size(); ++i) {
    v[i] = a[i] + b[i];
  }
  return v;
}

inline const DoubleVector operator-(const DoubleVector& a,
                                    const DoubleVector& b) {
  DoubleVector v(a.Size());
  assert(a.Size() == b.Size());
  for (size_t i = 0; i < a.Size(); ++i) {
    v[i] = a[i] - b[i];
  }
  return v;
}

inline const DoubleVector operator*(const DoubleVector& a, const double c) {
  DoubleVector v(a.Size());
  for (size_t i = 0; i < a.Size(); ++i) {
    v[i] = a[i] * c;
  }
  return v;
}

inline const DoubleVector operator*(const double c, const DoubleVector& a) {
  return a * c;
}

}  // namespace maxent
}  // namespace mltk

#endif  // MLTK_MAXENT_MATH_VECTOR_H_
