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

#ifndef CORE_COMMON_CONTEXT_H
#define CORE_COMMON_CONTEXT_H
#include <google/protobuf/arena.h>

#include <unordered_map>

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

#include "core/common/common.h"
#include "core/config/feature_config.h"

namespace clink {

class FeatureConfig;
class SourceParserBase;

class Context {
 public:
  Context(FeatureConfig* config);

  virtual ~Context() {}

  inline Feature* CreateMessage() {
    return google::protobuf::Arena::CreateMessage<Feature>(&arena_);
  }

  template <typename T>
  inline T* CreateMessageT() {
    return google::protobuf::Arena::CreateMessage<T>(&arena_);
  }

  inline const Feature* Get(const int64_t& key) {
    auto iter = var_table_.find(key);
    if (iter != var_table_.end()) {
      return iter->second;
    }
    return nullptr;
  }

  inline bool Set(const int64_t& key, const Feature* feature) {
    var_table_[key] = feature;
    return true;
  }

  inline bool Set(const std::string& key, const Feature* feature) {
    var_table_[MAKE_HASH(key)] = feature;
    return true;
  }

  void set_config(FeatureConfig* config) { config_ = config; }

  inline const FeatureConfig* config() { return config_; }

  void set_parser(std::shared_ptr<SourceParserBase>& parser) {
    parser_ = parser;
  }

  std::shared_ptr<SourceParserBase>& parser() { return parser_; }
  const std::vector<std::vector<std::shared_ptr<FeatureItem>>>*
  extract_sequence() {
    return extract_sequence_;
  }

 private:
  std::unordered_map<int64_t, const Feature*> var_table_;

  std::shared_ptr<SourceParserBase> parser_;

  google::protobuf::Arena arena_;

  FeatureConfig* config_;

  const std::vector<std::vector<std::shared_ptr<FeatureItem>>>*
      extract_sequence_;  //特征提取顺序

  // std::vector<FeatureItem>* output_sequence_;  //特征输出顺序
};
}  // namespace clink
#endif