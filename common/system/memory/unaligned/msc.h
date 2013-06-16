// Copyright (c) 2010, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

#ifndef COMMON_SYSTEM_MEMORY_UNALIGNED_MSC_H
#define COMMON_SYSTEM_MEMORY_UNALIGNED_MSC_H

#include "common/system/memory/unaligned/check_direct_include.h"

#include "common/base/type_cast.h"

#if defined(_M_MRX000) || defined(_M_ALPHA) || defined(_M_PPC) || defined(_M_IA64) || defined(_M_AMD64)

namespace common {

// microsoft c support __unaligned keyword under align sensitive archs
template <typename T>
T GetUnaligned(const void* p)
{
    return *static_cast<const __unaligned T*>(p);
}

template <typename T, typename U>
void PutUnaligned(void* p, const U& value)
{
    *static_cast<__unaligned T*>(p) = implicit_cast<T>(value);
}

} // namespace common
#else
// fallback to generic implementation
#include "common/system/memory/unaligned/generic.h"
#endif

#endif // COMMON_SYSTEM_MEMORY_UNALIGNED_MSC_H
