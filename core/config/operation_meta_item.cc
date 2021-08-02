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

#include "core/config/operation_meta_item.h"

#include <butil/logging.h>

#include <map>

#include "core/common/common.h"
#include "core/common/feature_item.h"
#include "core/processor/expression_builder.h"
#include "core/utils/string_utils.h"
#include "core/utils/top_sort.h"

namespace clink {

void OperationMetaItem::Reset() {
  input_features_.clear();
  expression_tree_ = nullptr;
  complete_ = false;
}

bool OperationMetaItem::ParseTransform(const Transform& transform) {
  if (!transform.IsInitialized() || transform.formula().empty()) {
    return false;
  }
  //参数转换
  OpParamMap op_param_map;
  op_param_map.clear();
  if (transform.params_size() > 0) {
    for (auto& it : transform.params()) {
      op_param_map.insert(std::make_pair(it.key(), it.value()));
    }
  }
  //构建表达式树
  std::shared_ptr<OperationNode> expression_tree =
      ExpressionBuilder::BuildExpressionTree(transform.formula(), op_param_map);
  if (expression_tree == nullptr) {
    LOG(ERROR) << "fail to build expression tree for " << transform.formula();
    return false;
  }
  expression_tree_ = expression_tree;
  return true;
}

int OperationMetaItem::Init(const Operation& operation) {
  for (auto& input : operation.input_features()) {
    input_features_.emplace_back(FeatureItem(input));
  }
  output_feature_ = FeatureItem(operation.output_feature());
  feature_size_ = operation.feature_size();
  output_feature_type_ = operation.output_feature_type();

  if (operation.transform_size() <= 0) {
    LOG(ERROR) << "parse feature transform error, feature:"
               << operation.output_feature();
    return ERR_PARSE_TRANSFORM;
  }

  if (!ParseTransform(operation.transform(0))) {
    LOG(ERROR) << "fail to parse transform config, feature: "
               << operation.output_feature() << " error no "
               << ERR_PARSE_TRANSFORM;
    return ERR_PARSE_TRANSFORM;
  }
  return STATUS_OK;
}

}  // namespace clink
