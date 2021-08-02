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
#include "core/utils/feature_internal.h"
#include "core/utils/feature_util.h"
namespace clink {
class ConvertUtil {
 public:
  static const std::string& ToString(const proto::Record& any,
                                     std::string& result);

  // static inline const std::string& ToString(const Feature& feature,
  //                                           std::string& result) {
  //   result = "";
  //   switch (feature.data_type()) {
  //     case proto::DT_STRING:
  //       result = GetFeatureValues<std::string>(feature).Get(0);
  //       break;
  //     case proto::DT_INT32:
  //       result = std::to_string(GetFeatureValues<int32_t>(feature).Get(0));
  //       break;
  //     case proto::DT_INT64:
  //       result = std::to_string(GetFeatureValues<int64_t>(feature).Get(0));
  //       break;
  //     case proto::DT_FLOAT:
  //       result = std::to_string(GetFeatureValues<float>(feature).Get(0));
  //       break;
  //     case proto::DT_DOUBLE:
  //       result = std::to_string(GetFeatureValues<double>(feature).Get(0));
  //       break;
  //     default:
  //       break;
  //   }
  //   return result;
  // }

  // static inline void ToDouble(const Feature& feature, double& value) {
  //   value = 0;
  //   switch (FeatureUtil::GetType(feature)) {
  //     case proto::DT_DOUBLE:
  //       value = GetFeatureValues<double>(feature).Get(0);
  //       break;
  //     case proto::DT_FLOAT:
  //       value = GetFeatureValues<float>(feature).Get(0);
  //       break;
  //     case proto::DT_INT32:
  //       value = GetFeatureValues<int>(feature).Get(0);
  //       break;
  //     case proto::DT_INT64:
  //       value = GetFeatureValues<int64_t>(feature).Get(0);
  //       break;
  //     default:
  //       break;
  //   }
  // }

  static inline void ToDouble(const proto::Record& any, double& value) {
    value = 0;
    switch (any.kind_case()) {
      case proto::Record::kDoubleList:
        value = GetFeatureValues<double>(any).Get(0);
        break;
      case proto::Record::kFloatList:
        value = GetFeatureValues<float>(any).Get(0);
        break;
      case proto::Record::kIntList:
        value = GetFeatureValues<int>(any).Get(0);
        break;
      case proto::Record::kInt64List:
        value = GetFeatureValues<int64_t>(any).Get(0);
        break;
      default:
        break;
    }
  }

  static void ToInt(const proto::Record&, int&);

  static void ToFloat(const proto::Record& any, float& value);

  // static inline void ToFloat(const Feature& feature, float& value) {
  //   value = 0;
  //   switch (feature.data_type()) {
  //     case proto::DT_INT32:
  //       value = GetFeatureValues<int32_t>(feature).Get(0);
  //       break;
  //     case proto::DT_INT64:
  //       value = GetFeatureValues<int64_t>(feature).Get(0);
  //       break;
  //     case proto::DT_FLOAT:
  //       value = GetFeatureValues<float>(feature).Get(0);
  //       break;
  //     case proto::DT_DOUBLE:
  //       value = GetFeatureValues<double>(feature).Get(0);
  //       break;
  //     default:
  //       break;
  //   }
  // }
  static const std::string* ToString(const proto::Record& any);
};

}  // namespace clink

#endif  // CORE_UTILS_CONVERT_UTIL_H_
