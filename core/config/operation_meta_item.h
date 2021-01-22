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
#include "core/common/operation_node.h"
namespace perception_feature {
template <typename T>
using FeatureRelation = std::unordered_map<T, std::unordered_set<T>>;

class OperationMetaItem {
 public:
  OperationMetaItem();
  virtual ~OperationMetaItem();
  void Reset();
  bool CheckComplete();
  void AddInputFeatures(const std::string& input_feature);
  void SetOutputFeature(const std::string& out_feature);
  void SetComplete(const bool& feature_complete);
  inline const std::shared_ptr<OperationNode>& GetExpressionTree() const {
    return expression_tree_;
  }
  const FeatureItem& GetOutputFeature();
  const std::vector<FeatureItem>& GetInputFeatures();
  void SetFeatureSize(const int& feature_size) { feature_size_ = feature_size; }
  inline const int& GetFeatureSize() const { return feature_size_; }
  void SetExpressionTree(
      const std::shared_ptr<OperationNode>& expression_tree) {
    expression_tree_ = expression_tree;
  }
  const FeatureType& GetOutputFeatureType() const {
    return output_feature_type_;
  }
  void SetOutputFeatureType(const FeatureType& feature_type) {
    output_feature_type_ = feature_type;
  }

 private:
  std::vector<FeatureItem> input_features_;         //输入特征列表
  FeatureItem output_feature_;                      //输出特征名
  std::shared_ptr<OperationNode> expression_tree_;  //表达式树
  bool feature_complete_;                           //特征是否完成计算
  int feature_size_;
  FeatureType output_feature_type_;
};
}  // namespace perception_feature

#endif  // CORE_CONFIG_OPERATION_META_ITEM_H_
