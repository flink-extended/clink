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

#include "core/operators/cosin.h"

#include <cmath>

#include "core/utils/convert_util.h"

namespace perception_feature {
Cosine::Cosine() : UnaryOperator() {}

Cosine::Cosine(const std::string& feature_name) : UnaryOperator() {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  init_status_ = true;
}

std::shared_ptr<BaseOperator> Cosine::Clone() const {
  return std::make_shared<Cosine>();
}

int Cosine::Evaluate(const FeatureMap& feature_map,
                     std::shared_ptr<Feature>& output) {
  output = std::make_shared<Feature>();
  if (!init_status_ || opa_num_ != 1 || children_[0] == nullptr) {
    return ERR_OP_NOT_INIT;
  }
  if (feature_map.empty()) {
    return ERR_EMPTY_INPUT;
  }
  std::shared_ptr<Feature> child;
  if (children_[0]->Evaluate(feature_map, child) != STATUS_OK ||
      child == nullptr) {
    return ERR_OP_STATUS_FAILED;
  }
  double d_res;
  ConvertUtil::ToDouble(*child, d_res);
  output->mutable_double_list()->add_value(std::cos(d_res));
  return STATUS_OK;
}

}  // namespace perception_feature
