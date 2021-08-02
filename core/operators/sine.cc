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

#include "core/operators/sine.h"

#include <butil/logging.h>

#include "core/utils/convert_util.h"
namespace clink {
Sine::Sine() : UnaryOperator() {}

Sine::Sine(const std::string& feature_name) : UnaryOperator() {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  init_status_ = true;
}

const Feature* Sine::Evaluate(Context* context) {
  Feature* output = context->CreateMessage();
  if (opa_num_ != 1 || children_[0] == nullptr) {
    LOG(ERROR) << "Sine Evaluate failed";
    return nullptr;
  }
  auto child = children_[0]->Evaluate(context);
  if (child == nullptr) {
    LOG(ERROR) << "Sine Evaluate failed";
    return nullptr;
  }
  double d_res;
  ConvertUtil::ToDouble(*child, d_res);
  GetFeatureValues<double>(output)->Add(std::sin(d_res));
  // feature->mutable_double_list()->add_value(std::sin(d_res));
  return output;
}

std::shared_ptr<BaseOperator> Sine::Clone() const {
  return std::make_shared<Sine>();
}

}  // namespace clink
