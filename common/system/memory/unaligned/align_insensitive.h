// Copyright (c) 2010, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

#ifndef COMMON_SYSTEM_MEMORY_UNALIGNED_ALIGN_INSENSITIVE_H
#define COMMON_SYSTEM_MEMORY_UNALIGNED_ALIGN_INSENSITIVE_H

// internal header, no inclusion guard needed

#include "common/base/type_cast.h"
#include "common/system/memory/unaligned/check_direct_include.h"

namespace common {

// align insensitive archs

template <typename T>
T GetUnaligned(const void* p)
{
    return *static_cast<const T*>(p);
}

// introduce U make T must be given explicitly
template <typename T, typename U>
void PutUnaligned(void* p, const U& value)
{
    *static_cast<T*>(p) = implicit_cast<T>(value);
}

} // namespace common

#endif // COMMON_SYSTEM_MEMORY_UNALIGNED_ALIGN_INSENSITIVE_H
