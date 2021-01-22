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

#include "core/operators/keep.h"

#include "core/utils/string_utils.h"
namespace perception_feature {
Keep::Keep() : UnaryOperator() {}

Keep::Keep(const std::string& feature_name) : UnaryOperator() {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  init_status_ = true;
}

int Keep::Evaluate(const FeatureMap& feature_map,
                   std::shared_ptr<Feature>& feature) {
  feature = std::make_shared<Feature>();
  if (!init_status_ || opa_num_ != 1 || children_[0] == nullptr) {
    return ERR_OP_NOT_INIT;
  }
  return children_[0]->Evaluate(feature_map, feature);
}
std::shared_ptr<BaseOperator> Keep::Clone() const {
  return std::make_shared<Keep>();
}

}  // namespace perception_feature
