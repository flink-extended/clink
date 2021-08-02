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

#include <butil/logging.h>

#include <algorithm>

#include "core/operators/base_operator.h"
#include "core/utils/convert_util.h"
#include "core/utils/feature_internal.h"
namespace clink {
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

int FeatureUtil::GetSize(const proto::Record& data) {
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

void FeatureUtil::ToIndexValue(const Feature& feature_result,
                               const OperationMetaItem* operation_config,
                               const int& start_index, std::vector<int>* index,
                               std::vector<float>* value) {
  if (!feature_result.IsInitialized() ||
      FeatureUtil::GetSize(feature_result) <= 0 ||
      nullptr == operation_config || nullptr == index || nullptr == value) {
    return;
  }

  // 连续特征的libsvm key为start_index，value为 计算结果
  // 离散特征key为start_index+偏移量 value为1
  const FeatureType& feature_type = operation_config->output_feature_type();
  switch (feature_type) {
    case proto::CONTINUOUS: {
      index->emplace_back(start_index);
      float result;
      ConvertUtil::ToFloat(feature_result, result);
      value->emplace_back(result);
      break;
    }
    case proto::VECTOR: {
      CalcIndexValueVector(feature_result, start_index, index, value);
      break;
    }
    //离散特征
    case proto::DISCRETE: {
      //离散特征
      int relative_index;  //相对索引
      // proto::DataType data_type = FeatureUtil::GetType(feature_result);
      switch (feature_result.kind_case()) {
        case proto::Record::kIntList: {
          auto& res_item = GetFeatureValues<int>(feature_result);
          for (auto& item : res_item) {
            index->emplace_back(start_index + item);
            value->emplace_back(1);
          }
          break;
        }
        case proto::Record::kInt64List: {
          auto& res_item = GetFeatureValues<int64_t>(feature_result);
          for (auto& item : res_item) {
            index->emplace_back(start_index + item);
            value->emplace_back(1);
          }
          break;
        }
        case proto::Record::kIvList: {
          auto& res_item = GetFeatureValues<IVRecordEntry>(feature_result);
          for (auto& item : res_item) {
            index->emplace_back(start_index + item.index());
            float iv_value;
            ConvertUtil::ToFloat(item.value(), iv_value);
            value->emplace_back(iv_value);
          }
          break;
        }
        default:
          break;
      }
    }
    default:
      break;
  }
}

void FeatureUtil::CalcIndexValueVector(const Feature& feature,
                                       const int& start_index,
                                       std::vector<int>* index,
                                       std::vector<float>* value) {
  if (nullptr == index || nullptr == value) {
    return;
  }

  std::copy(feature.float_list().value().begin(),
            feature.float_list().value().end(), std::back_inserter(*value));
  int size = feature.float_list().value_size();

  for (int i = 0; i < size; ++i) {
    index->emplace_back(start_index + i);
  }
}

void FeatureUtil::BuildResponse(Context* context, std::vector<int>* index,
                                std::vector<float>* value) {
  if (nullptr == index || nullptr == value) {
    return;
  }
  auto operation_meta = context->config()->operation_meta();
  auto& output_sequence = operation_meta->output_sequence();
  int start_index = 0;
  // const FeatureMap& feature_map = var_table.GetFeatureMap();
  for (auto& item : output_sequence) {
    //获取特征配置信息
    auto operation_config = operation_meta->GetOperationMetaItem(item);
    if (operation_config == nullptr) {
      LOG(ERROR) << "fail to find config for " << item.name();
      continue;
    }
    auto iter = context->Get(item.id());
    if (iter == nullptr) {
      start_index += operation_config->feature_size();
      continue;
    }
    FeatureUtil::ToIndexValue(*iter, operation_config, start_index, index,
                              value);
    start_index += operation_config->feature_size();
  }
}

void FeatureUtil::BuildResponse(const OperationMeta& operation_meta,
                                Context* context,
                                DinResultRecord* din_result_record) {
  if (nullptr == din_result_record) {
    return;
  }
  const std::vector<FeatureItem>& output_sequence =
      operation_meta.output_sequence();
  int start_index = 0;

  for (auto& item : output_sequence) {
    //获取特征配置信息
    const OperationMetaItem* operation_config =
        operation_meta.GetOperationMetaItem(item);
    if (operation_config == nullptr) {
      LOG(ERROR) << "fail to find config for " << item.name();
      continue;
    }
    auto iter = context->Get(item.id());
    if (iter == nullptr) {
      start_index += operation_config->feature_size();
      continue;
    }
    FeatureUtil::ToIndexValue(*iter, operation_config, start_index,
                              din_result_record);
    start_index += operation_config->feature_size();
  }
}

void FeatureUtil::ToIndexValue(const Feature& feature,
                               const OperationMetaItem* operation_config,
                               const int& start_index,
                               DinResultRecord* din_result_record) {
  if (!feature.IsInitialized() || FeatureUtil::GetSize(feature) <= 0 ||
      nullptr == operation_config || nullptr == din_result_record) {
    return;
  }
  // 连续特征的libsvm key为start_index，value为 计算结果
  // 离散特征key为start_index+偏移量 value为1
  const FeatureType& feature_type = operation_config->output_feature_type();
  switch (feature_type) {
    case proto::CONTINUOUS: {
      din_result_record->add_index((float)start_index);
      float result;
      ConvertUtil::ToFloat(feature, result);
      din_result_record->add_value(result);
      // value->emplace_back(result);
      break;
    }
    case proto::VECTOR: {
      CalcIndexValueVector(feature, start_index, din_result_record);
      break;
    }
      //离散特征
    case proto::DISCRETE: {
      //离散特征
      int relative_index;  //相对索引
      if (feature.kind_case() == proto::Record::kIntList) {
        auto& res_item = GetFeatureValues<int>(feature);
        for (auto& item : res_item) {
          din_result_record->add_index((float)(start_index + item));
          din_result_record->add_value(1);
        }
      } else if (feature.kind_case() == proto::Record::kIvList) {
        auto& res_item = GetFeatureValues<IVRecordEntry>(feature);
        for (auto& item : res_item) {
          //          index->emplace_back(start_index + item.index());
          float iv_value;
          ConvertUtil::ToFloat(item.value(), iv_value);
          //          value->emplace_back(iv_value);
          din_result_record->add_index((float)(start_index + item.index()));
          din_result_record->add_value(iv_value);
        }
      }

      break;
    }
    default:
      break;
  }
}

void FeatureUtil::CalcIndexValueVector(const Feature& feature,
                                       const int& start_index,
                                       DinResultRecord* din_result_record) {
  if (nullptr == din_result_record) {
    return;
  }
  std::copy(feature.float_list().value().begin(),
            feature.float_list().value().end(),
            google::protobuf::RepeatedFieldBackInserter(
                din_result_record->mutable_value()));
  int size = feature.float_list().value_size();
  for (int i = 0; i < size; ++i) {
    din_result_record->add_index((float)(start_index + i));
  }
}

}  // namespace clink
