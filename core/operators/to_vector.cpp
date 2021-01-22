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
#include "core/operators/to_vector.h"

#include <utility>
#include <vector>

#include "core/utils/feature_util.h"
#include "core/utils/string_utils.h"

namespace perception_feature {
ToVector::ToVector() : UnaryOperator() { vec_size_ = 0; }

ToVector::ToVector(const std::string& feature_name, const OpParamMap& param_map)
    : UnaryOperator() {
  vec_size_ = 0;
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParam(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}

ToVector::ToVector(const ToVector& node) : UnaryOperator(node) {}

std::shared_ptr<BaseOperator> ToVector::Clone() const {
  return std::make_shared<ToVector>();
}

int ToVector::Evaluate(const FeatureMap& feature_map,
                       std::shared_ptr<Feature>& output) {
  if (!init_status_ || opa_num_ != 1 || children_.empty() ||
      children_[0] == nullptr) {
    return ERR_OP_NOT_INIT;
  }
  std::shared_ptr<Feature> child;
  if (children_[0]->Evaluate(feature_map, child) != STATUS_OK ||
      child == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  RecordType record_type = FeatureUtil::GetType(*child);
  if (record_type == RECORD_TYPE_FLOAT || record_type == RECORD_TYPE_DOUBLE) {
    output = std::move(child);
  } else {
    output = std::make_shared<Feature>();
    std::string child_str;
    ConvertUtil::ToString(*child, child_str);
    if (child_str.empty()) {
      // LOG(WARNING) << "empty vector for feature " << var->GetKey();
      return STATUS_OK;
    }
    std::vector<std::string> res_vec;
    StringUtils::Split(child_str, separator_.c_str(), res_vec);
    if (res_vec.size() != vec_size_) {
      auto var = std::dynamic_pointer_cast<Variable>(children_[0]);
      LOG(WARNING) << "wrong vector number " << var->GetKey();
      return ERR_INVALID_VEC_SIZE;
    }
    for (auto& value : res_vec) {
      if ((vec_data_type_ == proto::INTEGER_TYPE ||
           vec_data_type_ == proto::REAL_TYPE) &&
          !StringUtils::IsNumber(value)) {
        LOG(ERROR) << "value " << value << "is not a number";
        continue;
      }
      if (vec_data_type_ == proto::INTEGER_TYPE) {
        output->mutable_int64_list()->add_value(stoi(value));
      } else if (vec_data_type_ == proto::REAL_TYPE) {
        output->mutable_float_list()->add_value(stof(value));
      } else {
        output->mutable_bytes_list()->add_value(value);
      }
    }
  }

  return STATUS_OK;
}

bool ToVector::ParseParamMap(const std::string& feature_name,
                             const OpParamMap& param_map) {
  return ParseParam(feature_name, param_map);
}

bool ToVector::ParseParam(const std::string& name,
                          const OpParamMap& param_map) {
  std::string key = name + "_size";
  OpParam size_param;
  if (!FeatureUtil::GetParam(param_map, key, size_param)) {
    return false;
  }
  int vec_size = 0;
  ConvertUtil::ToInt(size_param, vec_size);
  vec_size_ = vec_size > 0 ? vec_size : 0;
  OpParam sep_param;
  key = name + "_deli";
  if (!FeatureUtil::GetParam(param_map, key, sep_param)) {
    return false;
  }
  std::string separator;
  ConvertUtil::ToString(sep_param, separator);
  if (separator.empty()) {
    LOG(ERROR) << "invalid vector separator";
    return false;
  }
  separator_ = separator;
  OpParam type_param;
  key = name + "_type";
  if (!FeatureUtil::GetParam(param_map, key, type_param)) {
    return false;
  }
  std::string vec_type;
  ConvertUtil::ToString(type_param, vec_type);
  if (!StringUtils::CompareIgnoreCase(vec_type, "REAL_TYPE")) {
    vec_data_type_ = proto::FeatureDataType::REAL_TYPE;
  } else if (!StringUtils::CompareIgnoreCase(vec_type, "INTEGER_TYPE")) {
    vec_data_type_ = proto::FeatureDataType::INTEGER_TYPE;
  } else if (!StringUtils::CompareIgnoreCase(vec_type, "STRING_TYPE")) {
    vec_data_type_ = proto::FeatureDataType::STRING_TYPE;
  } else {
    LOG(ERROR) << "invalid vector type";
    return false;
  }
  return true;
}

}  // namespace perception_feature
