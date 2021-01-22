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

#ifndef CORE_COMMON_FEATURE_ITEM_H_
#define CORE_COMMON_FEATURE_ITEM_H_
#include <iostream>
#include <string>

#include "core/common/common.h"
namespace perception_feature {
class FeatureItem {
 public:
  FeatureItem() = default;
  explicit FeatureItem(const std::string& feature_name) {
    feature_id_ = MAKE_HASH(feature_name);
    feature_name_ = feature_name;
  }
  inline const std::string& Name() const { return feature_name_; }
  inline const int64_t& Id() const { return feature_id_; }
  virtual ~FeatureItem() = default;

 private:
  std::string feature_name_;
  int64_t feature_id_;
};

}  // namespace perception_feature

#endif  // CORE_COMMON_FEATURE_ITEM_H_
