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

#ifndef CORE_CONFIG_OPERATION_META_ITEM_H_
#define CORE_CONFIG_OPERATION_META_ITEM_H_
#include <climits>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"
#include "core/common/feature_item.h"
namespace clink {

class OperationNode;

class OperationMetaItem {
 public:
  int Init(const Operation& operation);

  void Reset();

  const std::vector<FeatureItem>& input_features() { return input_features_; }

  const FeatureItem& output_feature() const { return output_feature_; }

  const std::shared_ptr<OperationNode>& expression_tree() const {
    return expression_tree_;
  }

  const bool& feature_complete() { return complete_; }

  void set_complete(const bool& complete) { complete_ = complete; }

  inline const int& feature_size() const { return feature_size_; }

  const FeatureType& output_feature_type() const {
    return output_feature_type_;
  }

 private:
  bool ParseTransform(const Transform& transform);

  std::vector<FeatureItem> input_features_;  //输入特征列表

  FeatureItem output_feature_;  //输出特征名

  std::shared_ptr<OperationNode> expression_tree_;  //表达式树

  bool complete_;  //特征是否完成计算

  int feature_size_;

  FeatureType output_feature_type_;
};
}  // namespace clink

#endif  // CORE_CONFIG_OPERATION_META_ITEM_H_
