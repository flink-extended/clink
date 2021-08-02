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

#include "core/utils/string_utils.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cwctype>
#include <iostream>
#include <queue>
#include <set>
#include <stack>
#include <utility>

#include "core/common/common.h"
#include "core/utils/murmurhash.h"

namespace clink {

void StringUtils::Split(const std::string& str, const std::string& delim,
                        std::vector<std::string>* output) {
  output->clear();
  auto start = 0U;
  auto end = str.find(delim, start);
  int size = delim.size();
  while (end != std::string::npos) {
    output->emplace_back(std::move(str.substr(start, end - start)));
    start = end + size;
    end = str.find(delim, start);
  }

  if (start < str.size()) {
    output->emplace_back(std::move(str.substr(start)));
  }
}

void StringUtils::ReplaceAll(std::string& str, const std::string& from,
                             const std::string& to) {
  if (str.empty()) return;
  if (from.empty()) return;

  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

void StringUtils::ToLower(std::string* str) {
  std::transform(str->begin(), str->end(), str->begin(), ::tolower);
}

bool StringUtils::EndsWith(const std::string& fullstr,
                           const std::string& ending) {
  if (ending.empty()) {
    return true;
  }
  if (fullstr.length() < ending.length()) {
    return false;
  }
  return (fullstr.substr(fullstr.length() - ending.length(), ending.length()) ==
          ending);
}

bool StringUtils::StartsWith(const std::string& fullstr,
                             const std::string& prefix) {
  if (prefix.empty()) {
    return true;
  }
  if (fullstr.length() < prefix.length()) {
    return false;
  }
  return (fullstr.substr(0, prefix.length()) == prefix);
}

void StringUtils::Trim(std::string& str) {
  while (str.begin() != str.end() && std::iswspace(*str.begin())) {
    str.erase(str.begin());
  }

  auto it = str.end();
  while (it != str.begin() && std::iswspace(*--it)) {
    str.erase(it);
  }
}
////去除字符串中所有空格
// void StringUtils::RemoveAllSpace(std::string& str) {
//  while (str.begin() != str.end() && std::iswspace(*str.begin())) {
//    str.erase(str.begin());
//  }
//
//  auto it = str.end();
//  while (it != str.begin() && std::iswspace(*--it)) {
//    str.erase(it);
//  }
//}

bool StringUtils::SplitExpression(const std::string& input,
                                  const std::string& regex,
                                  std::vector<std::string>* result) {
  size_t pos = 0, lastPos = 0;
  while ((pos = input.find_first_of(regex, lastPos)) != std::string::npos) {
    if (pos > lastPos)
      result->emplace_back(input.substr(lastPos, pos - lastPos));
    lastPos = pos + 1;
  }
  if (lastPos < input.length()) {
    result->emplace_back(input.substr(lastPos));
  }
  return true;
}

int StringUtils::CompareIgnoreCase(const std::string& lhs,
                                   const std::string& rhs) {
  if (lhs.length() != rhs.length()) {
    return -1;
  }
  return strncasecmp(lhs.c_str(), rhs.c_str(), lhs.length());
}

bool StringUtils::IsBracketValid(const std::string& str) {
  std::stack<unsigned char> stack;
  for (auto& ch : str) {
    if (stack.empty() && ch == ')') {
      return false;
    } else {
      if (!stack.empty() && stack.top() != '(') {
        return false;
      }
      if (ch == '(') {
        stack.push(ch);
      }
      if (ch == ')' && stack.top() == '(') {
        stack.pop();
      }
    }
  }
  return stack.empty();
}

bool StringUtils::IsNumber(const std::string& str) {
  if (str.empty()) {
    return false;
  }
  if (str.find_first_not_of(real_chars) != std::string::npos) {
    return false;
  }
  return true;
}

int64_t StringUtils::StringHash(const std::string& str) {
  return MurmurHash64A(str.c_str(), str.size(), 0);
}

}  // namespace clink
