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

#include <map>

#include "core/common/feature_item.h"
#include "core/utils/string_utils.h"
#include "core/utils/top_sort.h"
namespace perception_feature {
OperationMetaItem::OperationMetaItem() {}
OperationMetaItem::~OperationMetaItem() {}
void OperationMetaItem::Reset() {
  input_features_.clear();
  expression_tree_ = nullptr;
  feature_complete_ = false;
}
bool OperationMetaItem::CheckComplete() { return feature_complete_; }
void OperationMetaItem::SetComplete(const bool& feature_complete) {
  feature_complete_ = feature_complete;
}
void OperationMetaItem::AddInputFeatures(const std::string& input_feature) {
  FeatureItem feature_item(input_feature);
  input_features_.emplace_back(feature_item);
}
void OperationMetaItem::SetOutputFeature(const std::string& out_feature) {
  output_feature_ = (FeatureItem)out_feature;
}
const FeatureItem& OperationMetaItem::GetOutputFeature() {
  return output_feature_;
}

const std::vector<FeatureItem>& OperationMetaItem::GetInputFeatures() {
  return input_features_;
}

}  // namespace perception_feature
