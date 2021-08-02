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

#ifndef CORE_OPERATORS_ROUND_H_
#define CORE_OPERATORS_ROUND_H_
#include <iostream>
#include <memory>
#include <string>

#include "core/operators/unary_operator.h"
namespace clink {
class Round : public UnaryOperator {
 public:
  Round();

  explicit Round(const std::string& feature_name);

  const Feature* Evaluate(Context*) override;

  std::shared_ptr<BaseOperator> Clone() const override;
};
}  // namespace clink

#endif  // CORE_OPERATORS_ROUND_H_
