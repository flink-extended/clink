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

#include "core/operands/real.h"

#include "core/common/common.h"
namespace perception_feature {
Real::Real(const double& value) : value_(value) {}

Real::Real(const Real& node) : Operand(node), value_(node.value_) {}

int Real::Evaluate(const FeatureMap&, std::shared_ptr<Feature>& feature) {
  feature = std::make_shared<Feature>();
  feature->mutable_float_list()->add_value(value_);
  return STATUS_OK;
}

}  // namespace perception_feature
