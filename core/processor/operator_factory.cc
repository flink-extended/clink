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

#include "core/processor/operator_factory.h"

#include "core/operators/bucket.h"
#include "core/operators/ceil.h"
#include "core/operators/cosin.h"
#include "core/operators/divide.h"
#include "core/operators/feature_hash.h"
#include "core/operators/floor.h"
#include "core/operators/keep.h"
#include "core/operators/logarithm.h"
#include "core/operators/modular.h"
#include "core/operators/multi_hot.h"
#include "core/operators/multiply.h"
#include "core/operators/one_hot.h"
#include "core/operators/plus.h"
#include "core/operators/power.h"
#include "core/operators/round.h"
#include "core/operators/sigmoid.h"
#include "core/operators/sine.h"
#include "core/operators/sqrt.h"
#include "core/operators/standard.h"
#include "core/operators/subtract.h"
#include "core/operators/tanh.h"
#include "core/operators/to_vector.h"
namespace clink {

OperatorFactory& OperatorFactory::GetInstance() {
  static OperatorFactory operator_factory;
  return operator_factory;
}

OperatorFactory::OperatorFactory() { Init(); }

#define REGISTER_OPERATOR(_processor_name_, _processor_class_) \
  RegisterOperator(#_processor_name_, std::make_shared<_processor_class_>())

void OperatorFactory::Init() {
  REGISTER_OPERATOR(ADD, Plus);
  REGISTER_OPERATOR(SUB, Subtract);
  REGISTER_OPERATOR(MULTI, Multiply);
  REGISTER_OPERATOR(DIV, Divide);
  REGISTER_OPERATOR(POW, Power);
  REGISTER_OPERATOR(MOD, Modular);
  REGISTER_OPERATOR(KEEP, Keep);
  REGISTER_OPERATOR(COS, Cosine);
  REGISTER_OPERATOR(SIN, Sine);
  REGISTER_OPERATOR(SIGMOID, Sigmoid);
  REGISTER_OPERATOR(SQRT, Sqrt);
  REGISTER_OPERATOR(STD, Standard);
  REGISTER_OPERATOR(TANH, Tanh);
  REGISTER_OPERATOR(ONE_HOT, OneHot);
  REGISTER_OPERATOR(BUCKET, Bucket);
  REGISTER_OPERATOR(MULTI_HOT, MultiHot);
  REGISTER_OPERATOR(TO_VECTOR, ToVector);
  REGISTER_OPERATOR(LOG, Logarithm);
  REGISTER_OPERATOR(CEIL, Ceil);
  REGISTER_OPERATOR(FLOOR, Floor);
  REGISTER_OPERATOR(ROUND, Round);
  REGISTER_OPERATOR(FEATURE_HASH, FeatureHash);
}

int OperatorFactory::RegisterOperator(const std::string& name,
                                      std::shared_ptr<BaseOperator> ops) {
  if (operator_mapping_.find(name) == operator_mapping_.end()) {
    operator_mapping_[name] = ops;
  }
  return 0;
}

std::shared_ptr<BaseOperator> OperatorFactory::GetOperator(
    const std::string& name) const {
  auto it = operator_mapping_.find(name);
  if (it != operator_mapping_.end()) {
    return it->second;
  }
  return nullptr;
}

}  // namespace clink
