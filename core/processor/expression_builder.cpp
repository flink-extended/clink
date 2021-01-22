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

#include "core/processor/expression_builder.h"

#include <iostream>
namespace perception_feature {
std::shared_ptr<OperationNode> ExpressionBuilder::BuildExpressionTree(
    const std::string& formula, const OpParamMap& param_map,
    const OperatorFactory& operator_factory) {
  if (formula.empty() || !StringUtils::IsBracketValid(formula)) {
    LOG(ERROR) << "failed to parse formula " << formula;
    return nullptr;
  }
  std::string exp_string = formula;
  StringUtils::ReplaceAll(exp_string, " ", "");
  std::vector<std::string> tokens;
  // gcc 4.7.2不支持正则表达式
  // StringUtils::SplitExpression(exp_string, R"(,|\)\)*,*|\(\(*,*|\s+)",
  // tokens);
  StringUtils::SplitExpression(exp_string, "(),", tokens);
  if (!isValid(formula, tokens, operator_factory)) {
    LOG(ERROR) << "failed to parse formula " << formula;
    return nullptr;
  }

  std::shared_ptr<OperationNode> root =
      GenerateExpressionTree(tokens, param_map, operator_factory);
  return tokens.empty() ? root : nullptr;
}

bool ExpressionBuilder::isValid(const std::string& formula,
                                const std::vector<std::string>& tokens,
                                const OperatorFactory& operator_factory) {
  if (formula.empty() || tokens.empty() ||
      !StringUtils::IsBracketValid(formula)) {
    return false;
  }
  return !(!StringUtils::StartsWith(formula, tokens.at(0)) ||
           operator_factory.GetOperator(tokens.at(0)) == nullptr);
}

std::shared_ptr<OperationNode> ExpressionBuilder::GenerateExpressionTree(
    std::vector<std::string>& tokens, const OpParamMap& op_param_map,
    const OperatorFactory& operator_factory) {
  if (tokens.empty()) {
    return nullptr;
  }
  std::string token = *tokens.begin();
  if (token.empty()) {
    return nullptr;
  }
  tokens.erase(tokens.begin());
  std::shared_ptr<BaseOperator> ops = operator_factory.GetOperator(token);
  //操作符
  if (ops != nullptr) {
    std::shared_ptr<BaseOperator> base_operator = ops->Clone();

    for (int i = 0; i < base_operator->GetChildrenNum(); i++) {
      std::shared_ptr<OperationNode> child =
          GenerateExpressionTree(tokens, op_param_map, operator_factory);
      if (child == nullptr) {
        LOG(ERROR) << "build child failed " << token;
        return nullptr;
      }
      const OperationType& operation_type = child->GetOperationType();
      if (operation_type == OP_OPERAND) {
        auto var = std::dynamic_pointer_cast<Variable>(child);
        if (var != nullptr) {
          if (!base_operator->ParseParamMap(var->GetKey(), op_param_map)) {
            LOG(ERROR) << "parse param failed " << token;
            return nullptr;
          }
          base_operator->AddVariables(var->GetKey());
        }
      } else if (operation_type == OP_OPERATOR) {
        // child是operator
        auto var = std::dynamic_pointer_cast<BaseOperator>(child);
        if (var != nullptr) {
          const std::vector<std::string>& variables = var->GetVariables();
          if (!base_operator->ParseParamMap(variables, op_param_map)) {
            LOG(ERROR) << "parse param failed " << token;
            return nullptr;
          }
          base_operator->AppendVariables(variables);
        }
      }
      base_operator->AddChild(child);
    }
    base_operator->SetInitStatus(true);
    base_operator->SetOperationType(OP_OPERATOR);
    return base_operator;
  } else {
    if (token.find_first_not_of(integer_chars) == std::string::npos) {
      std::shared_ptr<Integer> root =
          std::make_shared<Integer>(std::stoll(token));
      root->SetOperationType(OP_OPERAND);
      return root;
    } else if (token.find_first_not_of(real_chars) == std::string::npos) {
      std::shared_ptr<Real> root = std::make_shared<Real>(std::stof(token));
      root->SetOperationType(OP_OPERAND);
      return root;
    } else {
      std::shared_ptr<Variable> root = std::make_shared<Variable>(token);
      root->SetOperationType(OP_OPERAND);
      return root;
    }
  }
}

}  // namespace perception_feature
