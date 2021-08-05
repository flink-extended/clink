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

#include "core/operators/tanh.h"

#include <glog/logging.h>

#include "core/utils/convert_util.h"
namespace clink {
Tanh::Tanh() : UnaryOperator() {}

Tanh::Tanh(const std::string& feature_name) : UnaryOperator() {
  children_.emplace_back(std::make_shared<Variable>(feature_name));
  init_status_ = true;
}

const Feature* Tanh::Evaluate(Context* context) {
  Feature* output = context->CreateMessage();
  if (opa_num_ != 1 || children_[0] == nullptr) {
    LOG(ERROR) << "Tanh Evaluate failed, not init";
    return nullptr;
  }
  auto child = children_[0]->Evaluate(context);
  if (child == nullptr) {
    LOG(ERROR) << "Tanh Evaluate failed";
    return nullptr;
  }
  double d_res;
  ConvertUtil::ToDouble(*child, d_res);
  double d_exp = exp(d_res);
  double nd_exp = exp(-d_res);
  GetFeatureValues<double>(output)->Add((d_exp - nd_exp) / (d_exp + nd_exp));
  return output;
}

std::shared_ptr<BaseOperator> Tanh::Clone() const {
  return std::make_shared<Tanh>();
}

}  // namespace clink
