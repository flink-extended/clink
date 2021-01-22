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

#include "core/operators/multi_hot.h"

#include <iostream>
#include <set>
#include <utility>

#include "core/utils/convert_util.h"
#include "core/utils/feature_internal.h"
#include "core/utils/feature_util.h"
namespace perception_feature {
MultiHot::MultiHot() : BaseOperator(1) {}
MultiHot::MultiHot(const std::string& feature_name, const OpParamMap& param_map)
    : BaseOperator(1) {
  truncate_num_ = 0;
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParam(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}

std::shared_ptr<BaseOperator> MultiHot::Clone() const {
  return std::make_shared<MultiHot>();
}
int MultiHot::Evaluate(const FeatureMap& feature_map,
                       std::shared_ptr<Feature>& output) {
  output = std::make_shared<Feature>();
  if (!init_status_ || opa_num_ != 1 || children_.empty() ||
      children_[0] == nullptr) {
    return ERR_OP_NOT_INIT;
  }
  std::shared_ptr<Feature> child;
  if (children_[0]->Evaluate(feature_map, child) != STATUS_OK ||
      child == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  auto child_ptr = GetFeatureValues<std::string>(child.get());
  std::string child_value;
  if (child_ptr != nullptr && child_ptr->size() > 0) {
    child_value = child_ptr->Get(0);
  }
  auto var = children_[0]->GetOperationName();
  if (var == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  std::vector<std::string> line_vec;
  StringUtils::Split(child_value, deli_param_.c_str(), line_vec);
  std::string key;

  std::set<int> index_vec;
  if (line_vec.empty()) {
    index_vec.insert(bin_size_);
  } else {
    for (auto& it : line_vec) {
      int index = bin_size_;
      key = *var + "_bins_" + std::move(it);
      auto item = encode_map_.find(key);
      if (item != encode_map_.end()) {
        index = item->second;
      }
      index_vec.insert(index);
    }
  }

  // 如果配置了truncate_num_参数
  // 则需要对multihot结果进行截断和补零操作，使输出的值等长
  if (truncate_num_ > 0 && truncate_num_ <= bin_size_ + 1) {
    // std::set<int> addition_vec;
    // 计算需要补零的个数
    int addition_size = truncate_num_ - index_vec.size();
    int act_size = 0;      // 实际队列长度
    int act_addition = 0;  // 实际补0位
    for (int i = 0; i <= bin_size_; ++i) {
      if (act_size >= truncate_num_) {
        break;
      }
      // 插入值为1的索引
      IVRecordEntry iv_record_entry;
      if (index_vec.find(i) != index_vec.end()) {
        iv_record_entry.set_index(i);
        iv_record_entry.mutable_value()->mutable_int_list()->add_value(1);
        output->mutable_iv_list()->add_iv_record()->Swap(&iv_record_entry);
        // output->mutable_int_list()->add_value(i);
        ++act_size;
      } else if (act_addition < addition_size) {
        iv_record_entry.set_index(i);
        iv_record_entry.mutable_value()->mutable_int_list()->add_value(0);
        output->mutable_iv_list()->add_iv_record()->Swap(&iv_record_entry);
        //  output->mutable_int_list()->add_value(-i);//值为0
        //  的索引用负号标记，计算时碰到索引值为负，则该索引位置置0
        ++act_size;
        ++act_addition;
      }
    }
  } else {  // 不定长输出 multi—hot
    // AppendFeatureValues(index_vec,output.get());
    for (auto& item : index_vec) {
      output->mutable_int_list()->add_value(item);
    }
  }
  return STATUS_OK;
}
bool MultiHot::ParseParamMap(const std::string& feature_name,
                             const OpParamMap& param_map) {
  return ParseParam(feature_name, param_map);
}
bool MultiHot::ParseParam(const std::string& feature_name,
                          const OpParamMap& param_map) {
  std::string deli_key = feature_name + "_deli";
  std::string bins_key = feature_name + "_bins";
  std::string index_key = feature_name + "_index_only";
  std::string truncate_key = feature_name + "_truncate_num";
  OpParam deli_param, bins_param, index_param, truncate_param;
  if (!FeatureUtil::GetParam(param_map, deli_key, deli_param) ||
      !FeatureUtil::GetParam(param_map, bins_key, bins_param) ||
      !FeatureUtil::GetParam(param_map, index_key, index_param)) {
    return false;
  }
  deli_param_ = *ConvertUtil::ToString(deli_param);

  std::string key = feature_name + "_bins_";
  std::string param_key;
  int index = 0;
  switch (FeatureUtil::GetType(bins_param)) {
    case RECORD_TYPE_FLOAT: {
      index = 0;
      for (auto& it : bins_param.float_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_DOUBLE: {
      index = 0;
      for (auto& it : bins_param.double_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_INT64: {
      index = 0;
      for (auto& it : bins_param.int64_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_INT: {
      index = 0;
      for (auto& it : bins_param.int_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_BYTE: {
      index = 0;
      for (auto& it : bins_param.bytes_list().value()) {
        param_key = key + it;
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_BOOL: {
      index = 0;
      for (auto& it : bins_param.bool_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    default:
      break;
  }
  bin_size_ = FeatureUtil::GetSize(bins_param);

  //解析截断长度
  if (FeatureUtil::GetParam(param_map, truncate_key, truncate_param)) {
    ConvertUtil::ToInt(truncate_param, truncate_num_);
  }
  return true;
}
bool MultiHot::ParseParamMap(const std::vector<std::string>& variables,
                             const OpParamMap& param_map) {
  for (auto& item : variables) {
    ParseParam(item, param_map);
  }
  return true;
}

}  // namespace perception_feature
