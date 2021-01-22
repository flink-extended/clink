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

#ifndef CORE_UTILS_FEATURE_UTIL_H_
#define CORE_UTILS_FEATURE_UTIL_H_
#include <memory>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/common/variable_table.h"
#include "core/config/operation_meta.h"
namespace perception_feature {
typedef enum {
  RECORD_TYPE_UNKNOWN = 0,
  RECORD_TYPE_BYTE,
  RECORD_TYPE_BOOL,
  RECORD_TYPE_INT,
  RECORD_TYPE_INT64,
  RECORD_TYPE_FLOAT,
  RECORD_TYPE_DOUBLE,
  RECORD_TYPE_IV,
} RecordType;
class FeatureUtil {
 public:
  static bool GetParam(const OpParamMap& param_map, const std::string& key,
                       OpParam& param);
  static int GetSize(const Feature&);
  static inline RecordType GetType(const Feature& data) {
    if (data.has_float_list()) {
      return RECORD_TYPE_FLOAT;
    } else if (data.has_double_list()) {
      return RECORD_TYPE_DOUBLE;
    } else if (data.has_int64_list()) {
      return RECORD_TYPE_INT64;
    } else if (data.has_int_list()) {
      return RECORD_TYPE_INT;
    } else if (data.has_bytes_list()) {
      return RECORD_TYPE_BYTE;
    } else if (data.has_bool_list()) {
      return RECORD_TYPE_BOOL;
    } else if (data.has_iv_list()) {
      return RECORD_TYPE_IV;
    }
    return RECORD_TYPE_UNKNOWN;
  }
  static void ToIndexValue(Feature& feature_result,
                           const OperationMetaItem* operation_config,
                           const int& start_index,
                           IVRecordList& iv_record_list);
  static void ToIndexValue(const Feature& feature_result,
                           const OperationMetaItem* operation_config,
                           const int& start_index, std::vector<float>& index,
                           std::vector<float>& value);
  static void CalcIndexValueVector(const Feature& feature,
                                   const int& start_index,
                                   std::vector<float>& index,
                                   std::vector<float>& value);
  static void CalcIndexValueVector(const Feature& feature, const int& index,
                                   IVRecordList& iv_record_list);
  static void BuildResponse(const OperationMeta& operation_meta,
                            const FeatureVariableTable& var_table,
                            std::shared_ptr<FeatureResponse> response);
  static void BuildResponse(const OperationMeta& operation_meta,
                            const FeatureVariableTable& var_table,
                            std::vector<float>& index,
                            std::vector<float>& value);
};
}  // namespace perception_feature

#endif  // CORE_UTILS_FEATURE_UTIL_H_
