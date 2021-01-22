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

#ifndef CORE_COMMON_OPERATION_NODE_H_
#define CORE_COMMON_OPERATION_NODE_H_
#include <memory>
#include <stdexcept>
#include <string>

#include "core/common/common.h"
namespace perception_feature {
typedef enum { OP_OPERAND, OP_OPERATOR } OperationType;
class OperationNode {
 public:
  OperationNode();
  virtual ~OperationNode();
  virtual int Evaluate(const FeatureMap&, std::shared_ptr<Feature>&) = 0;
  virtual const std::string* GetOperationName() = 0;
  const OperationType& GetOperationType();
  void SetOperationType(const OperationType&);

 protected:
  OperationNode(const OperationNode&);

 private:
  OperationNode& operator=(const OperationNode&);
  OperationType operation_type_;
};

}  // namespace perception_feature

#endif  // CORE_COMMON_OPERATION_NODE_H_
