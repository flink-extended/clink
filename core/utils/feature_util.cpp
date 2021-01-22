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

#include "core/utils/feature_util.h"

#include "core/operators/base_operator.h"
#include "core/utils/convert_util.h"
namespace perception_feature {
bool FeatureUtil::GetParam(const OpParamMap& param_map, const std::string& key,
                           OpParam& param) {
  param.Clear();
  auto iter = param_map.find(key);
  if (iter != param_map.end()) {
    param = iter->second;
    return true;
  }
  return false;
}
int FeatureUtil::GetSize(const Feature& data) {
  int res = 0;
  switch (GetType(data)) {
    case RECORD_TYPE_FLOAT:
      res = data.float_list().value_size();
      break;
    case RECORD_TYPE_DOUBLE:
      res = data.double_list().value_size();
      break;
    case RECORD_TYPE_INT64:
      res = data.int64_list().value_size();
      break;
    case RECORD_TYPE_INT:
      res = data.int_list().value_size();
      break;
    case RECORD_TYPE_BYTE:
      res = data.bytes_list().value_size();
      break;
    case RECORD_TYPE_BOOL:
      res = data.bool_list().value_size();
      break;
    case RECORD_TYPE_IV:
      res = data.iv_list().iv_record_size();
      break;
    default:
      break;
  }
  return res;
}

// 计算libsvm
void FeatureUtil::ToIndexValue(Feature& feature_result,
                               const OperationMetaItem* operation_config,
                               const int& start_index,
                               IVRecordList& iv_record_list) {
  iv_record_list.Clear();
  if (!feature_result.IsInitialized() ||
      FeatureUtil::GetSize(feature_result) <= 0 ||
      operation_config == nullptr) {
    return;
  }
  bool index_only = true;
  //  auto var = std::dynamic_pointer_cast<BaseOperator>(
  //      operation_config->GetExpressionTree());
  //  if (var != nullptr) {
  //    index_only = var->GetIndexStatus();
  //  }
  // 连续特征的libsvm key为start_index，value为 计算结果
  // 离散特征key为start_index+偏移量 value为1
  FeatureType feature_type = operation_config->GetOutputFeatureType();
  std::string res;
  if (feature_type == proto::CONTINUOUS) {
    IVRecordEntry iv_record_entry;
    iv_record_entry.set_index(start_index);
    iv_record_entry.mutable_value()->Swap(&feature_result);
    iv_record_list.add_iv_record()->Swap(&iv_record_entry);
    return;
  } else if (feature_type == proto::VECTOR) {
    CalcIndexValueVector(feature_result, start_index, iv_record_list);
    return;
  }
  // 离散特征
  int relative_index = -1;  //相对索引

  //  if (index_only) {
  for (int i = 0; i < feature_result.int_list().value_size(); ++i) {
    relative_index = feature_result.int_list().value(i);
    IVRecordEntry iv_record_entry;
    iv_record_entry.set_index(start_index + relative_index);
    iv_record_entry.mutable_value()->mutable_int_list()->add_value(1);
    iv_record_list.add_iv_record()->Swap(&iv_record_entry);
  }
  //  } else {
  //    for (int i = 0; i < feature_result.int_list().value_size(); ++i) {
  //      if (feature_result.int_list().value(i) == 1) {
  //        relative_index = i;
  //        IVRecordEntry iv_record_entry;
  //        iv_record_entry.set_index(start_index + relative_index);
  //        iv_record_entry.mutable_value()->mutable_int_list()->add_value(1);
  //        iv_record_list.add_iv_record()->Swap(&iv_record_entry);
  //      }
  //    }
  //  }
}

void FeatureUtil::ToIndexValue(const Feature& feature_result,
                               const OperationMetaItem* operation_config,
                               const int& start_index,
                               std::vector<float>& index,
                               std::vector<float>& value) {
  if (!feature_result.IsInitialized() ||
      FeatureUtil::GetSize(feature_result) <= 0 ||
      operation_config == nullptr) {
    return;
  }
  // 连续特征的libsvm key为start_index，value为 计算结果
  // 离散特征key为start_index+偏移量 value为1
  const FeatureType& feature_type = operation_config->GetOutputFeatureType();
  switch (feature_type) {
    case proto::CONTINUOUS: {
      index.emplace_back(start_index);
      float result;
      ConvertUtil::ToFloat(feature_result, result);
      value.emplace_back(result);
      break;
    }
    case proto::VECTOR: {
      CalcIndexValueVector(feature_result, start_index, index, value);
      break;
    }
    case proto::DISCRETE: {
      //离散特征
      int relative_index;  //相对索引
      if (feature_result.has_int_list()) {
        for (int i = 0; i < feature_result.int_list().value_size(); ++i) {
          relative_index = feature_result.int_list().value(i);
          index.emplace_back(start_index + relative_index);
          value.emplace_back(1);
        }
      } else if (feature_result.has_iv_list()) {
        for (int i = 0; i < feature_result.iv_list().iv_record_size(); ++i) {
          relative_index = feature_result.iv_list().iv_record(i).index();
          float iv_value;
          ConvertUtil::ToFloat(feature_result.iv_list().iv_record(i).value(),
                               iv_value);
          index.emplace_back(start_index + relative_index);
          value.emplace_back(iv_value);
        }
      }

      break;
    }
    default:
      break;
  }
}

void FeatureUtil::CalcIndexValueVector(const Feature& feature, const int& index,
                                       IVRecordList& iv_record_list) {
  iv_record_list.Clear();
  int size = FeatureUtil::GetSize(feature);
  if (!feature.IsInitialized() || size <= 0) {
    return;
  }

  for (int i = 0; i < size; ++i) {
    IVRecordEntry iv_record_entry;
    switch (FeatureUtil::GetType(feature)) {
      case RECORD_TYPE_DOUBLE:
        iv_record_entry.set_index(index + i);
        iv_record_entry.mutable_value()->mutable_double_list()->add_value(
            feature.double_list().value(i));
        break;
      case RECORD_TYPE_INT:
        iv_record_entry.set_index(index + i);
        iv_record_entry.mutable_value()->mutable_int64_list()->add_value(
            feature.int64_list().value(i));
        break;
      case RECORD_TYPE_BYTE:
        iv_record_entry.set_index(index + i);
        iv_record_entry.mutable_value()->mutable_bytes_list()->add_value(
            feature.bytes_list().value(i));
        break;
      case RECORD_TYPE_FLOAT:
        iv_record_entry.set_index(index + i);
        iv_record_entry.mutable_value()->mutable_float_list()->add_value(
            feature.float_list().value(i));
        break;
      default:
        break;
    }
    if (iv_record_entry.IsInitialized()) {
      iv_record_list.add_iv_record()->CopyFrom(iv_record_entry);
    }
  }
}

void FeatureUtil::CalcIndexValueVector(const Feature& feature,
                                       const int& start_index,
                                       std::vector<float>& index,
                                       std::vector<float>& value) {
  int size = feature.float_list().value_size();
  for (int i = 0; i < size; ++i) {
    index.emplace_back(start_index + i);
    value.emplace_back(feature.float_list().value(i));
  }

  //  int size = FeatureUtil::GetSize(feature);
  //  if (!feature.IsInitialized() || size <= 0) {
  //    return;
  //  }
  //  //std::cout<<" FeatureUtil::GetSize(feature) "<<
  //  FeatureUtil::GetSize(feature)<<std::endl; switch
  //  (FeatureUtil::GetType(feature)) {
  //      case RECORD_TYPE_DOUBLE:
  //        for (int i = 0; i < size; ++i) {
  //          index.emplace_back(start_index+i);
  //          value.emplace_back((float)feature.double_list().value(i));
  //        }
  //        break;
  //      case RECORD_TYPE_INT:
  //        for (int i = 0; i < size; ++i) {
  //          index.emplace_back(start_index + i);
  //          value.emplace_back((float)feature.int64_list().value(i));
  //        }
  //        break;
  //      case RECORD_TYPE_FLOAT:
  //        for (int i = 0; i < size; ++i) {
  //          index.emplace_back(start_index+i);
  //          value.emplace_back(feature.float_list().value(i));
  //        }
  //        break;
  //      default:
  //        break;
  //    }
}

void FeatureUtil::BuildResponse(const OperationMeta& operation_meta,
                                const FeatureVariableTable& var_table,
                                std::vector<float>& index,
                                std::vector<float>& value) {
  const std::vector<FeatureItem>& output_sequence =
      operation_meta.GetOutPutSequence();
  int start_index = 0;
  const FeatureMap& feature_map = var_table.GetFeatureMap();
  for (auto& item : output_sequence) {
    //获取特征配置信息
    const OperationMetaItem* operation_config =
        operation_meta.GetOperationMetaItem(item);
    if (operation_config == nullptr) {
      LOG(ERROR) << "fail to find config for " << item.Name();
      continue;
    }
    const Feature* iter = var_table.GetValue(item.Id());
    if (iter == nullptr) {
      start_index += operation_config->GetFeatureSize();
      continue;
    }
    FeatureUtil::ToIndexValue(*iter, operation_config, start_index, index,
                              value);
    start_index += operation_config->GetFeatureSize();
  }
}

}  // namespace perception_feature
