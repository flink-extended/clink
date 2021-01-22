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

#include "core/utils/convert_util.h"

#include <iostream>

#include "core/config/operation_meta.h"
#include "core/utils/feature_util.h"
namespace perception_feature {
const std::string& ConvertUtil::ToString(const Feature& any,
                                         std::string& result) {
  result = "";
  switch (FeatureUtil::GetType(any)) {
    case RECORD_TYPE_BYTE:
      for (int i = 0; i < any.bytes_list().value_size(); ++i) {
        result += any.bytes_list().value(i);
      }
      // result = any.bytes_list().value(0);
      break;
    case RECORD_TYPE_FLOAT:
      for (int i = 0; i < any.float_list().value_size(); ++i) {
        result += std::to_string(any.float_list().value(i));
      }
      break;
    case RECORD_TYPE_DOUBLE:
      for (int i = 0; i < any.double_list().value_size(); ++i) {
        // double s = any.double_list().value(i);
        result += std::to_string(any.double_list().value(i));
      }
      break;
    case RECORD_TYPE_INT64:
      for (int i = 0; i < any.int64_list().value_size(); ++i) {
        result += std::to_string(any.int64_list().value(i));
      }
      break;
    case RECORD_TYPE_INT:
      for (int i = 0; i < any.int_list().value_size(); ++i) {
        result += std::to_string(any.int_list().value(i));
      }
      break;
    case RECORD_TYPE_BOOL:
      for (int i = 0; i < any.bool_list().value_size(); ++i) {
        result += std::to_string(any.bool_list().value(i));
      }
    case RECORD_TYPE_IV:
      for (int i = 0; i < any.iv_list().iv_record_size(); ++i) {
        int relative_index = any.iv_list().iv_record(i).index();
        int value;
        ConvertUtil::ToInt(any.iv_list().iv_record(i).value(), value);
        result +=
            std::to_string(relative_index) + ":" + std::to_string(value) + " ";
      }

    default:
      break;
  }
  return result;
}
// void ConvertUtil::ToDouble(const Feature& any, double& value) {
//
//  switch (FeatureUtil::GetType(any)) {
//    case RECORD_TYPE_FLOAT:
//      value = any.float_list().value(0);
//      break;
//    case RECORD_TYPE_DOUBLE:
//      value = any.double_list().value(0);
//      break;
//    case RECORD_TYPE_INT64:
//      value = any.int64_list().value(0);
//      break;
//    case RECORD_TYPE_INT:
//      value = any.int_list().value(0);
//      break;
//    default:
//      break;
//  }
//}
// void ConvertUtil::ToDouble(const Feature& any, double& value) {
//  value = 0;
//  if (any.has_double_list()){
//    value = any.double_list().value(0);
//  }else if (any.has_float_list()){
//    value = any.float_list().value(0);
//  }else if (any.has_int64_list()){
//    value = any.double_list().value(0);
//  }else if (any.has_int_list()){
//    value = any.int_list().value(0);
//  }
//}
void ConvertUtil::ToFloat(const Feature& any, float& value) {
  switch (FeatureUtil::GetType(any)) {
    case RECORD_TYPE_FLOAT:
      value = any.float_list().value(0);
      break;
    case RECORD_TYPE_DOUBLE:
      value = static_cast<float>(any.double_list().value(0));
      break;
    case RECORD_TYPE_INT64:
      value = static_cast<float>(any.int64_list().value(0));
      break;
    case RECORD_TYPE_INT:
      value = any.int_list().value(0);
      break;
    default:
      break;
  }
}

void ConvertUtil::ToInt(const Feature& feature, int& value) {
  value = 0;
  switch (FeatureUtil::GetType(feature)) {
    case RECORD_TYPE_FLOAT:
      value = static_cast<int>(feature.float_list().value(0));
      break;
    case RECORD_TYPE_DOUBLE:
      value = static_cast<int>(feature.double_list().value(0));
      break;
    case RECORD_TYPE_INT64:
      value = static_cast<int>(feature.int64_list().value(0));
      break;
    case RECORD_TYPE_INT:
      value = feature.int_list().value(0);
      break;
    default:
      break;
  }
}

const std::string* ConvertUtil::ToString(const Feature& any) {
  if (any.has_bytes_list()) {
    return &any.bytes_list().value(0);
  }
  return nullptr;
}

}  // namespace perception_feature
