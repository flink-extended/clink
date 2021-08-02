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
#include "core/operators/binary_operator.h"

#include <iomanip>
#include <memory>
#include <string>

#include "core/utils/convert_util.h"
namespace clink {

BinaryOperator::BinaryOperator() : BaseOperator(2) {}

BinaryOperator::BinaryOperator(const std::string& first_feature,
                               const std::string& second_feature)
    : BaseOperator(2) {}

bool BinaryOperator::GetTreeValue(Context* context, double& left,
                                  double& right) const {
  left = 0;
  right = 0;
  if (opa_num_ != 2 || children_.size() != 2 || children_[0] == nullptr ||
      children_[1] == nullptr) {
    return false;
  }
  auto left_child = children_[0]->Evaluate(context);
  auto right_child = children_[1]->Evaluate(context);

  if (left_child == nullptr || right_child == nullptr) {
    return false;
  }
  ConvertUtil::ToDouble(*left_child, left);
  ConvertUtil::ToDouble(*right_child, right);
  return true;
}

}  // namespace clink
