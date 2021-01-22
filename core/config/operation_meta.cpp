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

#include "core/config/operation_meta.h"

#include <unordered_set>
#include <utility>

#include "core/common/feature_item.h"
#include "core/utils/top_sort.h"
namespace perception_feature {
OperationMeta::OperationMeta() {}

OperationMeta::~OperationMeta() { operation_map_.clear(); }

void OperationMeta::Reset() { operation_map_.clear(); }

int OperationMeta::AddOperation(
    const FeatureItem& feature,
    const OperationMetaItem& feature_meta_base) {  //添加特征
  if (operation_map_.find(feature.Id()) != operation_map_.end()) {
    return ERR_DUPLICATE_FEATURE;
  }
  total_feature_size_ += feature_meta_base.GetFeatureSize();
  operation_map_.insert(std::make_pair(feature.Id(), feature_meta_base));
  return STATUS_OK;
}

void OperationMeta::AddFeatureRelation(const std::vector<FeatureItem>& input,
                                       const FeatureItem& output) {
  //依赖特征为空直接返回
  if (input.empty()) {
    return;
  }
  for (auto& in : input) {
    auto iter = feature_relation_.find(in.Name());
    if (iter != feature_relation_.end()) {
      if (in.Name().compare(output.Name())) {
        iter->second.emplace(output.Name());
      }
    } else {
      std::unordered_set<std::string> child;
      if (in.Name().compare(output.Name())) {
        child.emplace(output.Name());
      }
      feature_relation_.insert(std::make_pair(in.Name(), child));
    }
  }
}

bool OperationMeta::BfsTraverse() {
  bool result = top_sort::BfsTraverse(feature_relation_, feature_sequence_);
  if (!result) {
    return false;
  }
  for (auto& item : feature_sequence_) {
    FeatureItem feature_item(item);
    feature_item_sequence_.emplace_back(feature_item);
  }
  return true;
}

}  // namespace perception_feature
