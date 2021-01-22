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

#include "core/operators/feature_hash.h"

#include "core/operators/multi_hot.h"
#include "core/utils/convert_util.h"
#include "core/utils/feature_internal.h"
#include "core/utils/feature_util.h"
#include "core/utils/murmurhash.h"
namespace perception_feature {
FeatureHash::FeatureHash() : BaseOperator(1) { bin_size_ = 0; }
FeatureHash::FeatureHash(const std::string& feature_name,
                         const OpParamMap& param_map)
    : BaseOperator(1) {
  bin_size_ = 0;
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  if (!ParseParam(feature_name, param_map)) {
    init_status_ = false;
  }
  init_status_ = true;
}
FeatureHash::FeatureHash(const FeatureHash& node) : BaseOperator(node) {
  bin_size_ = 0;
}

std::shared_ptr<BaseOperator> FeatureHash::Clone() const {
  return std::make_shared<FeatureHash>();
}
int FeatureHash::Evaluate(const FeatureMap& feature_map,
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
  std::string child_str;
  ConvertUtil::ToString(*child, child_str);
  if (bin_size_ == 0) {
    return ERR_OP_STATUS_FAILED;
  }
  //  auto child_ptr = GetFeatureValues<std::string>(child.get());
  //
  //  if (child_ptr == nullptr || child_ptr->empty()) {
  //    return ERR_OP_STATUS_FAILED;
  //  }

  int64_t hash_code = StringUtils::StringHash(child_str);
  GetFeatureValues<int>(output.get())->Add(abs(hash_code) % bin_size_);
  return STATUS_OK;
}
bool FeatureHash::ParseParamMap(const std::string& feature_name,
                                const OpParamMap& param_map) {
  return ParseParam(feature_name, param_map);
}
bool FeatureHash::ParseParam(const std::string& name,
                             const OpParamMap& param_map) {
  std::string key = name + "_bin_size";
  OpParam bin_param;
  if (!FeatureUtil::GetParam(param_map, key, bin_param)) {
    return false;
  }
  if (bin_param.int_list().value_size() > 0)
    bin_size_ = bin_param.int_list().value(0);
  return true;
}
bool FeatureHash::ParseParamMap(const std::vector<std::string>& variables,
                                const OpParamMap& param_map) {
  for (auto& item : variables) {
    ParseParam(item, param_map);
  }
  return true;
}

}  // namespace perception_feature
