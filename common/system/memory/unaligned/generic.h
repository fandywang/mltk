// Copyright (c) 2010, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

#ifndef COMMON_SYSTEM_MEMORY_UNALIGNED_GENERIC_H
#define COMMON_SYSTEM_MEMORY_UNALIGNED_GENERIC_H

// generic solution, using memcpy

#include <string.h>

#include "common/base/type_cast.h"
#include "common/system/memory/unaligned/check_direct_include.h"

namespace common {

template <typename T>
T GetUnaligned(const void* p)
{
    T t;
    memcpy(&t, p, sizeof(t));
    return t;
}

template <typename T, typename U>
void PutUnaligned(void* p, const U& value)
{
    T t = implicit_cast<T>(value);
    memcpy(p, &t, sizeof(t));
}

} // namespace common

#endif // COMMON_SYSTEM_MEMORY_UNALIGNED_GENERIC_H
