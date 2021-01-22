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

#include "core/operators/one_hot.h"

#include <iostream>
#include <memory>

#include "core/utils/convert_util.h"
#include "core/utils/feature_internal.h"
#include "core/utils/feature_util.h"
namespace perception_feature {
OneHot::OneHot() : UnaryOperator() { bin_size_ = 0; }
OneHot::OneHot(const std::string& feature_name, const OpParamMap& param_map)
    : UnaryOperator() {
  bin_size_ = 0;
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParam(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}
OneHot::OneHot(const OneHot& node) : UnaryOperator(node) {}

std::shared_ptr<BaseOperator> OneHot::Clone() const {
  return std::make_shared<OneHot>();
}
int OneHot::Evaluate(const FeatureMap& feature_map,
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
  std::string bins_key;  //, child_str;
  int index = bin_size_;

  auto var = children_[0]->GetOperationName();
  if (var == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  bins_key = *var + "_bins_";

  auto child_ptr = GetFeatureValues<std::string>(child.get());
  if (child_ptr != nullptr && child_ptr->size() > 0) {
    bins_key += child_ptr->Get(0);
  }
  auto item = encode_map_.find(bins_key);
  if (item != encode_map_.end()) {
    index = item->second;
  }
  GetFeatureValues<int>(output.get())->Add(index);
  return STATUS_OK;
}
bool OneHot::ParseParamMap(const std::string& feature_name,
                           const OpParamMap& param_map) {
  return ParseParam(feature_name, param_map);
}
bool OneHot::ParseParam(const std::string& name, const OpParamMap& param_map) {
  std::string key = name + "_bins";
  OpParam bin_param;
  if (!FeatureUtil::GetParam(param_map, key, bin_param)) {
    return false;
  }
  key += "_";
  std::string param_key;
  int index = 0;
  switch (FeatureUtil::GetType(bin_param)) {
    case RECORD_TYPE_FLOAT: {
      index = 0;
      for (auto& it : bin_param.float_list().value()) {
        param_key = key + std::to_string(it);
        //        OpParam param;
        //        param.mutable_int_list()->add_value(index++);
        //        InsertParam(param_key, param);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_DOUBLE: {
      index = 0;
      for (auto& it : bin_param.double_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_INT64: {
      index = 0;
      for (auto& it : bin_param.int64_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_INT: {
      index = 0;
      for (auto& it : bin_param.int_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_BYTE: {
      index = 0;
      for (auto& it : bin_param.bytes_list().value()) {
        param_key = key + it;
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    case RECORD_TYPE_BOOL: {
      index = 0;
      for (auto& it : bin_param.bool_list().value()) {
        param_key = key + std::to_string(it);
        encode_map_.emplace(param_key, index++);
      }
      break;
    }
    default:
      break;
  }
  bin_size_ = FeatureUtil::GetSize(bin_param);
  return true;
}
bool OneHot::ParseParamMap(const std::vector<std::string>& variables,
                           const OpParamMap& param_map) {
  for (auto& item : variables) {
    ParseParam(item, param_map);
  }
  return true;
}

}  // namespace perception_feature
