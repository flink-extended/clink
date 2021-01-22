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

#ifndef CORE_UTILS_CONVERT_UTIL_H_
#define CORE_UTILS_CONVERT_UTIL_H_
#include <climits>
#include <string>

#include "core/common/common.h"
namespace perception_feature {
class ConvertUtil {
 public:
  static const std::string& ToString(const Feature& any, std::string& result);
  static inline void ToDouble(const Feature& any, double& value) {
    value = 0;
    if (any.has_double_list()) {
      value = any.double_list().value(0);
    } else if (any.has_float_list()) {
      value = any.float_list().value(0);
    } else if (any.has_int64_list()) {
      value = any.int64_list().value(0);
    } else if (any.has_int_list()) {
      value = any.int_list().value(0);
    }
  }
  static void ToInt(const Feature&, int&);
  static void ToFloat(const Feature& any, float& value);
  static const std::string* ToString(const Feature& any);
  //  static void ToIntArray(const Feature& feature, std::vector<int> value);
};

}  // namespace perception_feature

#endif  // CORE_UTILS_CONVERT_UTIL_H_
