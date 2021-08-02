
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

#ifndef CORE_UTILS_MD5_UTIL_H_
#define CORE_UTILS_MD5_UTIL_H_
#include <string>

namespace clink {
class MD5Util {
 public:
  // 计算字符串的md5
  static void MD5(const std::string& str, std::string* md5);

  static void MD5(const void* key, size_t len, std::string* md5);

  // 计算文件中内容的md5
  static int MD5File(const std::string& filename, std::string* md5);

};  // MD5Util
}  // namespace clink

#endif  // CORE_UTILS_MD5_UTIL_H_
