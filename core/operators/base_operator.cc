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

#include "core/operators/base_operator.h"
namespace clink {
BaseOperator::BaseOperator() {
  opa_num_ = 0;
  children_.clear();
  init_status_ = false;
  variables_.clear();
}

BaseOperator::BaseOperator(const BaseOperator& node) : OperationNode(node) {
  opa_num_ = node.opa_num_;
  children_.clear();
  for (auto& it : node.children_) {
    children_.emplace_back(it);
  }
}

BaseOperator::BaseOperator(const int& opa_num) : opa_num_(opa_num) {
  children_.clear();
}

void BaseOperator::AddChild(const std::shared_ptr<OperationNode>& child) {
  children_.emplace_back(child);
}

int BaseOperator::GetChildrenNum() const { return opa_num_; }
void BaseOperator::SetParams(
    const std::unordered_map<std::string, OpParam>& params) {
  for (auto& it : params) {
    params_.insert(std::make_pair(it.first, it.second));
  }
}

const OpParam* BaseOperator::GetParam(const std::string& key) {
  auto iter = params_.find(key);
  if (iter != params_.end()) {
    return &iter->second;
  } else {
    return nullptr;
  }
}

bool BaseOperator::InsertParam(const std::string& key, const OpParam& param) {
  if (params_.find(key) != params_.end()) {
    return false;
  }
  params_.insert(std::make_pair(key, param));
  return true;
}

const std::string* BaseOperator::GetOperationName() { return nullptr; }

const std::vector<std::string>& BaseOperator::GetVariables() const {
  return variables_;
}

void BaseOperator::AppendVariables(const std::vector<std::string>& variables) {
  variables_.insert(variables_.end(), variables.begin(), variables.end());
}

void BaseOperator::AddVariables(const std::string& var) {
  variables_.emplace_back(var);
}

}  // namespace clink
