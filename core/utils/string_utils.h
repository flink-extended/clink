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

#ifndef CORE_UTILS_STRING_UTILS_H_
#define CORE_UTILS_STRING_UTILS_H_
#include <string>
#include <vector>
namespace clink {
class StringUtils {
 public:
  static void Split(const std::string& str, const std::string& delim,
                    std::vector<std::string>* output);

  static void ReplaceAll(std::string& str, const std::string& from,
                         const std::string& to);

  static void ToLower(std::string* str);

  static bool StartsWith(const std::string& fullstr, const std::string& prefix);

  static bool EndsWith(const std::string& fullstr, const std::string& ending);

  static void Trim(std::string& str);

  static bool IsNumber(const std::string& str);

  static std::string RandomString(int len);

  static int CompareIgnoreCase(const std::string& lhs, const std::string& rhs);

  static bool SplitExpression(const std::string& input,
                              const std::string& regex,
                              std::vector<std::string>* result);

  static bool IsBracketValid(const std::string& str);

  static int64_t StringHash(const std::string& str);
};
}  // namespace clink

#endif  // CORE_UTILS_STRING_UTILS_H_
