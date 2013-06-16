// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 12/14/11
// Description: test for COMMON_STATIC_ASSERT

#include "common/base/static_assert.h"

#include <gtest/gtest.h>

namespace common {

TEST(StaticAssert, Test)
{
    COMMON_STATIC_ASSERT(1 == 1);
    COMMON_STATIC_ASSERT(1 == 1, "1 should be equal to 1");
}

TEST(StaticAssert, NoCompileTest)
{
#if 0 // uncomment to test
    COMMON_STATIC_ASSERT(false);
    COMMON_STATIC_ASSERT(1 == 2);
    COMMON_STATIC_ASSERT(1 == 2, "1 == 2");
#endif
}

} // namespace common
