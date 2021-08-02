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

#include "core/operators/unary_operator.h"
namespace clink {

UnaryOperator::UnaryOperator(const std::shared_ptr<OperationNode>& child)
    : BaseOperator(1) {
  children_.emplace_back(child);
}

UnaryOperator::UnaryOperator(const UnaryOperator& node) : BaseOperator(node) {}

UnaryOperator::~UnaryOperator() {}

UnaryOperator::UnaryOperator() : BaseOperator(1) {}

UnaryOperator::UnaryOperator(const std::string&) : BaseOperator(1) {}
}  // namespace clink
