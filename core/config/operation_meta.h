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

#ifndef CORE_CONFIG_OPERATION_META_H_
#define CORE_CONFIG_OPERATION_META_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/feature_item.h"
#include "core/config/operation_meta_item.h"
namespace perception_feature {
class OperationMeta {
 public:
  OperationMeta();
  virtual ~OperationMeta();
  void Reset();
  int AddOperation(const FeatureItem& feature,
                   const OperationMetaItem& feature_meta_base);
  void AddFeatureRelation(const std::vector<FeatureItem>&, const FeatureItem&);
  void AddOutputSequence(const std::string& output) {
    output_sequence_.emplace_back(output);
  }
  void SetOutputFromat(const OutputFromat& output_type) {
    output_type_ = output_type;
  }
  bool BfsTraverse();
  inline const std::unordered_map<int64_t, OperationMetaItem>& GetOperationMap()
      const {
    return operation_map_;
  }
  //特征提取顺序
  inline const std::vector<FeatureItem>& GetFeatureItemSequence() const {
    return feature_item_sequence_;
  }
  //特征输出顺序
  inline const std::vector<FeatureItem>& GetOutPutSequence() const {
    return output_sequence_;
  }
  inline const OutputFromat& GetOutputFromat() const { return output_type_; }
  inline const OperationMetaItem* GetOperationMetaItem(
      const FeatureItem& feature) const {
    auto it = operation_map_.find(feature.Id());
    if (it == operation_map_.end()) {
      return nullptr;
    }
    return &it->second;
  }
  inline const int& GetTotalFeatureSize() const { return total_feature_size_; }

 private:
  FeatureRelation<std::string> feature_relation_;
  std::vector<std::string> feature_sequence_;       //特征提取顺序
  std::vector<FeatureItem> feature_item_sequence_;  //特征提取顺序
  std::vector<FeatureItem> output_sequence_;        //特征输出顺序
  int total_feature_size_;
  OutputFromat output_type_;
  std::unordered_map<int64_t, OperationMetaItem>
      operation_map_;  // 配置中所有op
};

}  // namespace perception_feature

#endif  // CORE_CONFIG_OPERATION_META_H_
