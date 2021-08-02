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
namespace clink {
//存储特征名和特征id之间的对应关系
class FeatureItem {
 public:
  FeatureItem() = default;

  explicit FeatureItem(const std::string& feature_name) {
    id_ = MAKE_HASH(feature_name);
    name_ = feature_name;
  }

  FeatureItem(const FeatureItem& item) {
    this->id_ = item.id();
    this->name_ = item.name();
  }

  inline const std::string& name() const { return name_; }

  inline const int64_t& id() const { return id_; }

  virtual ~FeatureItem() = default;

 private:
  std::string name_;
  int64_t id_;
};

}  // namespace clink

#endif  // CORE_COMMON_FEATURE_ITEM_H_
