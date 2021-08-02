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
#include "core/utils/topsort.h"
#include "core/utils/util.h"
namespace clink {

class OperationMeta {
 public:
  OperationMeta();

  virtual ~OperationMeta();

  void Reset();

  int LoadOperation(const std::string& config_path);

  //特征输出顺序
  inline const std::vector<FeatureItem>& output_sequence() const {
    return output_sequence_;
  }

  //特征提取顺序
  inline const std::vector<std::vector<std::shared_ptr<FeatureItem>>>&
  extract_sequence() const {
    return extract_sequence_;
  }

  inline const OutputFromat& output_type() const { return output_type_; }

  inline const OperationMetaItem* GetOperationMetaItem(
      const FeatureItem& feature) const {
    auto it = operation_map_.find(feature.id());
    if (it == operation_map_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  inline const int& total_feature_size() const { return total_feature_size_; }

 private:
  int AddOperation(std::shared_ptr<OperationMetaItem>&);

  bool TopSort();

 private:
  std::vector<std::vector<std::shared_ptr<FeatureItem>>>
      extract_sequence_;  //特征提取顺序

  std::vector<FeatureItem> output_sequence_;  //特征输出顺序

  int total_feature_size_;

  OutputFromat output_type_;

  std::unordered_map<int64_t, std::shared_ptr<OperationMetaItem>>
      operation_map_;  // 配置中所有op

  std::unique_ptr<utils::TopSort<std::string>> sorter_;
};

}  // namespace clink

#endif  // CORE_CONFIG_OPERATION_META_H_
