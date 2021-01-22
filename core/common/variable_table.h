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

#ifndef CORE_COMMON_VARIABLE_TABLE_H_
#define CORE_COMMON_VARIABLE_TABLE_H_
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/common/common.h"
namespace perception_feature {
class FeatureVariableTable {
 private:
  FeatureMap var_table_;  // TODO(lee): 后续并行抽取时考虑并发问题

 public:
  FeatureVariableTable() {}
  virtual ~FeatureVariableTable() { Clear(); }
  inline void Insert(const std::string& token,
                     const std::shared_ptr<Feature>& value) {
    var_table_.insert({MAKE_HASH(token), value});
  }
  inline bool GetValue(const int64_t token,
                       std::shared_ptr<Feature>& feature) const {
    feature = nullptr;
    auto iter = var_table_.find(token);
    if (iter != var_table_.end()) {
      feature = iter->second;
      return true;
    }
    return false;
  }
  inline const Feature* GetValue(const int64_t& token) const {
    auto iter = var_table_.find(token);
    if (iter != var_table_.end()) {
      return iter->second.get();
    }
    return nullptr;
  }
  inline void SetValue(const int64_t& token,
                       const std::shared_ptr<Feature>& value) {
    auto iter = var_table_.find(token);
    if (iter != var_table_.end()) {
      iter->second = std::move(value);
    } else {
      var_table_.insert({token, std::move(value)});
    }
  }
  inline void ReserveSize(const int& size) { var_table_.reserve(size); }
  inline void Clear() {
    // std::cout<<"clear"<<std::endl;
    FeatureMap tmp;
    var_table_.swap(tmp);
  }
  inline bool Empty() { return var_table_.empty(); }
  inline int Size() { return var_table_.size(); }
  inline const FeatureMap& GetFeatureMap() const { return var_table_; }
};
}  // namespace perception_feature

#endif  // CORE_COMMON_VARIABLE_TABLE_H_
