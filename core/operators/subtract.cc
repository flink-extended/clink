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
#include "core/operators/subtract.h"

#include <memory>
#include <string>

#include "core/utils/feature_internal.h"
namespace clink {

Subtract::Subtract() : BinaryOperator() {}

Subtract::Subtract(const std::string& first_feature,
                   const std::string& second_feature)
    : BinaryOperator() {
  children_.emplace_back(std::make_shared<Variable>(first_feature));
  children_.emplace_back(std::make_shared<Variable>(second_feature));
  init_status_ = true;
}

const Feature* Subtract::Evaluate(Context* context) {
  Feature* output = context->CreateMessage();
  double left = 0, right = 0;
  GetTreeValue(context, left, right);
  GetFeatureValues<double>(output)->Add(left - right);
  return output;
}

std::shared_ptr<BaseOperator> Subtract::Clone() const {
  return std::make_shared<Subtract>();
}

}  // namespace clink