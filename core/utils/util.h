/* Copyright (c) 2021, Qihoo, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

#ifndef CLINK_UTILS_UTIL_H
#define CLINK_UTILS_UTIL_H

#include <memory>  // std::shared_ptr, std::unique_ptr

// C++11 Check
#if __cplusplus < 201103L
#error This library needs at least a C++11 compliant compiler.
#endif  // >= C++11

// Define std::make_unique for c++11
namespace std {

#if (__cplusplus >= 201103L && __cplusplus < 201402L)
template <typename T, typename... Args>
inline unique_ptr<T> make_unique(Args&&... args) {
  return unique_ptr<T>(new T(forward<Args>(args)...));
}
#endif  // C++11 <= version < C++14

}  // namespace std

// Branch prediction
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif  // CLINK_UTILS_UTIL_H
