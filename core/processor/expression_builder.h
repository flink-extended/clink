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

#ifndef CORE_PROCESSOR_EXPRESSION_BUILDER_H_
#define CORE_PROCESSOR_EXPRESSION_BUILDER_H_
#include <butil/logging.h>

#include <memory>
#include <string>
#include <vector>

#include "core/common/operation_node.h"
#include "core/operands/integer.h"
#include "core/operands/real.h"
#include "core/operands/variable.h"
#include "core/utils/string_utils.h"
namespace clink {
class ExpressionBuilder {
 public:
  static std::shared_ptr<OperationNode> BuildExpressionTree(
      const std::string&, const OpParamMap& param_map);

 private:
  static bool isValid(const std::string&, const std::vector<std::string>&);
  static std::shared_ptr<OperationNode> GenerateExpressionTree(
      std::vector<std::string>& tokens, const OpParamMap& op_param_map);
};
}  // namespace clink

#endif  // CORE_PROCESSOR_EXPRESSION_BUILDER_H_
