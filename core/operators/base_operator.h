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

#ifndef CORE_OPERATORS_BASE_OPERATOR_H_
#define CORE_OPERATORS_BASE_OPERATOR_H_
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/common/operation_node.h"
#include "core/operands/variable.h"
namespace perception_feature {
class BaseOperator : public OperationNode {
 public:
  virtual ~BaseOperator();
  void AddChild(const std::shared_ptr<OperationNode>& child);
  void SetParams(const std::unordered_map<std::string, Feature>&);
  int GetChildrenNum() const;
  virtual std::shared_ptr<BaseOperator> Clone() const = 0;
  // bool GetParam(const std::string& key, OpParam&);
  const OpParam* GetParam(const std::string& key);
  bool InsertParam(const std::string& key, const OpParam&);
  void SetInitStatus(const bool& status) { init_status_ = status; }
  const std::vector<std::string>& GetVariables() const;
  void AppendVariables(const std::vector<std::string>&);
  void AddVariables(const std::string& var);
  //   bool GetIndexStatus() override {
  //    return index_only_;
  //  }
  virtual bool ParseParamMap(const std::string& name,
                             const OpParamMap& param_map) {
    return true;
  }
  virtual bool ParseParamMap(const std::vector<std::string>& name,
                             const OpParamMap& param_map) {
    return true;
  }
  const std::string* GetOperationName() override;

 protected:
  BaseOperator();
  BaseOperator(const BaseOperator&);
  explicit BaseOperator(const int&);
  std::vector<std::shared_ptr<OperationNode>> children_;  //子树列表
  int opa_num_;                                           //操作元个数
  std::unordered_map<std::string, Feature> params_;
  bool init_status_;
  // bool index_only_;//是否对特征进行 one hot 编码
 private:
  BaseOperator& operator=(const BaseOperator&);
  std::vector<std::string> variables_;
};
}  // namespace perception_feature

#endif  // CORE_OPERATORS_BASE_OPERATOR_H_
