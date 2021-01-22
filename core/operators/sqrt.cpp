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

#include "core/operators/sqrt.h"

#include "core/utils/convert_util.h"
namespace perception_feature {
Sqrt::Sqrt(const Sqrt& node) : UnaryOperator(node) {}

Sqrt::Sqrt() : UnaryOperator() {}

Sqrt::Sqrt(const std::string& feature_name) : UnaryOperator(feature_name) {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  init_status_ = true;
}

int Sqrt::Evaluate(const FeatureMap& feature_map,
                   std::shared_ptr<Feature>& feature) {
  feature = std::make_shared<Feature>();
  if (opa_num_ != 1 || children_[0] == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  std::shared_ptr<Feature> child;
  if (children_[0]->Evaluate(feature_map, child) != STATUS_OK ||
      child == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  double d_res;
  ConvertUtil::ToDouble(*child, d_res);
  feature->mutable_double_list()->add_value(sqrt(d_res));
  return STATUS_OK;
}

std::shared_ptr<BaseOperator> Sqrt::Clone() const {
  return std::make_shared<Sqrt>();
}

}  // namespace perception_feature
