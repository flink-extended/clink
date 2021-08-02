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

#include "core/operators/standard.h"

#include <butil/logging.h>

#include "core/utils/convert_util.h"
#include "core/utils/feature_util.h"

namespace clink {
Standard::Standard() : UnaryOperator() {}

Standard::Standard(const std::string& feature_name, const OpParamMap& param_map)
    : UnaryOperator(), feature_name_(feature_name) {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParamMap(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}

const Feature* Standard::Evaluate(Context* context) {
  Feature* output = context->CreateMessage();
  double child_value = 0;
  if (!init_status_ || opa_num_ != 1 || children_.empty() ||
      children_[0] == nullptr) {
    LOG(ERROR) << "operator Standard is not init ";
    return nullptr;
  }
  auto child = children_[0]->Evaluate(context);
  if (child == nullptr) {
    LOG(ERROR) << "Standard Evaluate failed";
    return nullptr;
  }
  ConvertUtil::ToDouble(*child, child_value);
  if (std_ == 0) {
    return nullptr;
  }
  GetFeatureValues<double>(output)->Add((child_value - mean_) / std_);
  // feature->mutable_double_list()->add_value((child_value - mean_) / std_);
  return output;
}

std::shared_ptr<BaseOperator> Standard::Clone() const {
  return std::make_shared<Standard>();
}

bool Standard::ParseParamMap(const std::string& feature_name,
                             const OpParamMap& param_map) {
  std::string key = feature_name + "_mean";
  OpParam param;
  if (!FeatureUtil::GetParam(param_map, key, param)) {
    return false;
  }
  ConvertUtil::ToDouble(param, mean_);
  // InsertParam(key, param);
  key = feature_name + "_std";
  if (!FeatureUtil::GetParam(param_map, key, param)) {
    return false;
  }
  ConvertUtil::ToDouble(param, std_);
  // InsertParam(key, param);
  return true;
}

}  // namespace clink
