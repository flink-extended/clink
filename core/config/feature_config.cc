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

#include "core/config/feature_config.h"

#include "core/source/parser_factory.h"
#include "core/source/source_parser_base.h"
namespace clink {

FeatureConfig::FeatureConfig() : source_parser_(nullptr) {
  operation_meta_ = std::make_shared<OperationMeta>();
}

int FeatureConfig::LoadConfig(const std::string& config_path) {
  int res = operation_meta_->LoadOperation(config_path);
  if (res != STATUS_OK) {
    return res;
  }
  source_parser_ = std::move(SourceFactory::CreateSourceParser(config_path));
  return STATUS_OK;
}

}  // namespace clink
