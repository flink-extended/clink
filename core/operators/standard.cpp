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

#include "core/utils/convert_util.h"
#include "core/utils/feature_util.h"

namespace perception_feature {
Standard::Standard() : UnaryOperator() {}

Standard::Standard(const Standard& node) : UnaryOperator(node) {}
Standard::Standard(const std::string& feature_name, const OpParamMap& param_map)
    : UnaryOperator() {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParamMap(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}
int Standard::Evaluate(const FeatureMap& feature_map,
                       std::shared_ptr<Feature>& feature) {
  feature = std::make_shared<Feature>();
  double child_value = 0;
  if (!init_status_ || opa_num_ != 1 || children_.empty() ||
      children_[0] == nullptr) {
    return ERR_OP_NOT_INIT;
  }
  std::shared_ptr<Feature> child;
  if (children_[0]->Evaluate(feature_map, child) != STATUS_OK ||
      child == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  ConvertUtil::ToDouble(*child, child_value);
  if (std_ == 0) {
    return ERR_OP_STATUS_FAILED;
  }
  feature->mutable_double_list()->add_value((child_value - mean_) / std_);
  return STATUS_OK;
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

}  // namespace perception_feature
