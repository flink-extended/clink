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
#include "core/common/context.h"
#include "core/config/operation_meta.h"

namespace clink {

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

  static int GetSize(const proto::Record&);
  // static int GetSize(const Feature&);
  static inline RecordType GetType(const proto::Record& data) {
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

  // static inline proto::DataType GetType(const Feature& data) {
  //   if (data.data_type() != proto::RESERVED) {
  //     return data.data_type();
  //   }
  //   if (data.float_list_size() > 0) {
  //     return proto::DT_FLOAT;
  //   } else if (data.double_list_size() > 0) {
  //     return proto::DT_DOUBLE;
  //   } else if (data.int64_list_size() > 0) {
  //     return proto::DT_INT64;
  //   } else if (data.int_list_size() > 0) {
  //     return proto::DT_INT32;
  //   } else if (data.bytes_list_size() > 0) {
  //     return proto::DT_STRING;
  //   } else if (data.bool_list_size() > 0) {
  //     return proto::DT_BOOL;
  //   } else if (data.iv_list_size() > 0) {
  //     return proto::DT_IV;
  //   }
  //   return proto::RESERVED;
  // }

  static void ToIndexValue(const Feature&,
                           const OperationMetaItem* operation_config,
                           const int& start_index, std::vector<int>* index,
                           std::vector<float>* value);

  static void CalcIndexValueVector(const Feature& feature,
                                   const int& start_index,
                                   std::vector<int>* index,
                                   std::vector<float>* value);

  static void BuildResponse(Context* context, std::vector<int>* index,
                            std::vector<float>* value);

  static void BuildResponse(const OperationMeta& operation_meta,
                            Context* var_table,
                            DinResultRecord* din_result_record);

  static void ToIndexValue(const Feature& feature,
                           const OperationMetaItem* operation_config,
                           const int& start_index,
                           DinResultRecord* din_result_record);

  static void CalcIndexValueVector(const Feature& feature,
                                   const int& start_index,
                                   DinResultRecord* din_result_record);
};
}  // namespace clink

#endif  // CORE_UTILS_FEATURE_UTIL_H_
