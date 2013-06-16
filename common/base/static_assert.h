// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

#ifndef COMMON_BASE_STATIC_ASSERT_H
#define COMMON_BASE_STATIC_ASSERT_H

#include "common/base/cxx11.h"

#ifdef COMMON_CXX11_ENABLED

#define COMMON_STATIC_ASSERT(e, ...) static_assert(e, "" __VA_ARGS__)

#else

#include "common/base/preprocess/join.h"

namespace common {

template <bool x> struct static_assertion_failure;

template <> struct static_assertion_failure<true> {
    enum { value = 1 };
};

template<int x> struct static_assert_test {};

// Static assertions during compilation, Usage:
// COMMON_STATIC_ASSERT(sizeof(Foo) == 48, "Size of Foo must equal to 48");
#define COMMON_STATIC_ASSERT(e, ...) \
    typedef ::common::static_assert_test < \
            sizeof(::common::static_assertion_failure<static_cast<bool>(e)>)> \
            COMMON_PP_JOIN(static_assert_failed, __LINE__)

} // namespace common

#endif // COMMON_CXX11_ENABLED

#endif // COMMON_BASE_STATIC_ASSERT_H

