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

#include <butil/logging.h>
#include <core/utils/feature_internal.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "core/utils/feature_util.h"
#include "core/utils/string_utils.h"
namespace clink {

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

// ToVector::ToVector(const ToVector& node) : UnaryOperator(node) {}

std::shared_ptr<BaseOperator> ToVector::Clone() const {
  return std::make_shared<ToVector>();
}

const Feature* ToVector::Evaluate(Context* context) {
  if (!init_status_ || opa_num_ != 1 || children_.empty() ||
      children_[0] == nullptr) {
    return nullptr;
  }
  auto child = children_[0]->Evaluate(context);
  if (child == nullptr) {
    return nullptr;
  }
  //将string拆分为vector

  switch (child->kind_case()) {
    case proto::Record::kFloatList:
    case proto::Record::kDoubleList:
    case proto::Record::kIntList:
    case proto::Record::kInt64List:
      return child;
      break;
    case proto::Record::kBytesList: {
      Feature* output = context->CreateMessage();
      const std::string& child_str =
          GetFeatureValues<std::string>(*child).Get(0);
      if (child_str.empty()) {
        return output;
      }
      Transform(child_str, output);
      return output;
    }
    default:
      break;
  }
  return nullptr;
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

void ToVector::Transform(const std::string& input, Feature* output) {
  std::vector<std::string> res_vec;
  StringUtils::Split(input, separator_.c_str(), &res_vec);
  if (res_vec.size() != vec_size_) {
    auto var = std::dynamic_pointer_cast<Variable>(children_[0]);
    LOG(WARNING) << "wrong vector number " << var->GetKey();
    return;
  }
  std::vector<int> res_int;
  std::vector<float> res_float;
  switch (vec_data_type_) {
    case proto::STRING_TYPE:
      AppendFeatureValues(res_vec, output);
      break;
    case proto::INTEGER_TYPE:
      res_int.reserve(vec_size_);
      std::transform(res_vec.begin(), res_vec.end(),
                     std::back_inserter(res_int), [](const std::string& item) {
                       return strtoll(item.c_str(), nullptr, 10);
                     });
      AppendFeatureValues(res_int, output);
      break;
    case proto::REAL_TYPE:
      res_float.reserve(vec_size_);
      std::transform(res_vec.begin(), res_vec.end(),
                     std::back_inserter(res_float),
                     [](const std::string& item) {
                       return strtod(item.c_str(), nullptr);
                     });
      AppendFeatureValues(res_float, output);
      break;
    default:
      break;
  }
}

}  // namespace clink
