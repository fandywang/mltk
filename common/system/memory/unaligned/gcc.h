// Copyright (c) 2010, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

#ifndef COMMON_SYSTEM_MEMORY_UNALIGNED_GCC_H
#define COMMON_SYSTEM_MEMORY_UNALIGNED_GCC_H

// suppress mistake warning: ‘packed’ attribute ignored for field of type ‘T’
#pragma GCC system_header

#include <stddef.h>

#include "common/base/static_assert.h"
#include "common/base/type_cast.h"
#include "common/system/memory/unaligned/check_direct_include.h"

namespace common {

template <typename T, size_t size>
struct UnalignedWrapper
{
    STATIC_ASSERT(size == 2 || size == 4 || size == 8);
    T value __attribute__((packed));
};

template <typename T>
struct UnalignedWrapper<T, 1>
{
    T value; // one byte needn't packed
};

template <typename T>
T GetUnaligned(const void* p)
{
    return static_cast<const UnalignedWrapper<T, sizeof(T)>*>(p)->value;
}

template <typename T, typename U>
void PutUnaligned(void* p, const U& value)
{
    T t = implicit_cast<T>(value);
    static_cast<UnalignedWrapper<T, sizeof(T)>*>(p)->value = t;
}

} // namespace common

#endif // COMMON_SYSTEM_MEMORY_UNALIGNED_GCC_H
