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

#ifndef CORE_CONFIG_FEATURE_CONFIG_H_
#define CORE_CONFIG_FEATURE_CONFIG_H_
#include <memory>
#include <string>
#include <vector>

#include "core/config/operation_meta.h"

namespace clink {
class SourceParserBase;

class FeatureConfig {
 public:
  FeatureConfig();
  
  int LoadConfig(const std::string&);

  std::shared_ptr<SourceParserBase>& source_parser() { return source_parser_; }

  inline const OperationMeta* operation_meta() const {
    return operation_meta_.get();
  }

 private:
  std::shared_ptr<OperationMeta> operation_meta_;

  std::shared_ptr<SourceParserBase> source_parser_;
};
}  // namespace clink

#endif  // CORE_CONFIG_FEATURE_CONFIG_H_
