//-----------------------------------------------------------------------------
// MurmurHash2 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - This code makes a few assumptions about how your machine behaves -

// 1. We can read a 4-byte value from any address without crashing
// 2. sizeof(int) == 4

// And it has a few limitations -

// 1. It will not work incrementally.
// 2. It will not produce the same results on little-endian and big-endian
//    machines.
#ifndef CORE_UTILS_MURMURHASH_H_
#define CORE_UTILS_MURMURHASH_H_

#include <cstdint>

#include <algorithm>
namespace clink {

uint64_t MurmurHash64A(const void* key, int len, uint64_t seed);
}

#endif  // CORE_UTILS_MURMURHASH_H_
