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

#ifndef CORE_OPERATORS_UNARY_OPERATOR_H_
#define CORE_OPERATORS_UNARY_OPERATOR_H_
#include <memory>
#include <string>

#include "core/operators/base_operator.h"
namespace perception_feature {
class UnaryOperator : public BaseOperator {
 public:
  UnaryOperator();
  virtual ~UnaryOperator();
  std::shared_ptr<BaseOperator> Clone() const override = 0;

 protected:
  UnaryOperator(const UnaryOperator&);
  explicit UnaryOperator(const std::shared_ptr<OperationNode>&);
  explicit UnaryOperator(const std::string&);

 private:
  UnaryOperator& operator=(const UnaryOperator&);
};
}  // namespace perception_feature
#endif  // CORE_OPERATORS_UNARY_OPERATOR_H_
