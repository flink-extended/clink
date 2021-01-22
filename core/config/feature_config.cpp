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

#include "core/config/feature_config_loader.h"
namespace perception_feature {
FeatureConfig::~FeatureConfig() {}

FeatureConfig::FeatureConfig() {}

int FeatureConfig::LoadConfig(const std::string& config_path) {
  FeatureConfigLoader config_loader;
  config_loader.SetConfigPath(config_path);
  int res = config_loader.LoadOperationMeta(operation_meta_);
  if (res != STATUS_OK) {
    return res;
  }
  return config_loader.LoadDataSourceConfig(data_source_list_);
}

}  // namespace perception_feature
