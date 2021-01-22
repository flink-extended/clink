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

#include "core/processor/feature_list.h"

#include "core/common/common.h"
#include "core/utils/feature_internal.h"
namespace perception_feature {
FeatureList::FeatureList() { feature_list.clear(); }
FeatureList::~FeatureList() { feature_list.clear(); }
void FeatureList::AddValue(const double& value) {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  GetFeatureValues<double>(feature.get())->Add(value);
  feature_list.emplace_back(feature);
}
void FeatureList::AddValue(const float& value) {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  GetFeatureValues<float>(feature.get())->Add(value);
  feature_list.emplace_back(feature);
}
void FeatureList::AddValue(const int& value) {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  GetFeatureValues<int>(feature.get())->Add(value);
  feature_list.emplace_back(feature);
}
void FeatureList::AddValue(const std::string& value) {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  *GetFeatureValues<std::string>(feature.get())->Add() = value;
  // GetFeatureValues<std::string>(feature.get())->Add(value);
  // feature->mutable_bytes_list()->add_value(value);
  feature_list.emplace_back(feature);
}
void FeatureList::AddValue(const std::vector<float>& value) {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  AppendFeatureValues(value, feature.get());
  feature_list.emplace_back(feature);
}
int FeatureList::GetValue(const int& index, std::shared_ptr<Feature>& feature) {
  if (index >= feature_list.size()) {
    feature = nullptr;
    return ERR_INDEX_OUT_BOUNDARY;
  }
  feature = feature_list.at(index);
  return STATUS_OK;
}

void FeatureList::AddValue(const std::vector<double>& value) {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  AppendFeatureValues(value, feature.get());
  feature_list.emplace_back(feature);
}
void FeatureList::AddEmptyValue() {
  std::shared_ptr<Feature> feature = std::make_shared<Feature>();
  feature_list.emplace_back(feature);
}

}  // namespace perception_feature
