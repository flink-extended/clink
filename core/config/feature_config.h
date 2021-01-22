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
#include "core/datasource/data_source_base.h"
namespace perception_feature {
class FeatureConfig {
 public:
  FeatureConfig();
  virtual ~FeatureConfig();
  int LoadConfig(const std::string&);
  const std::vector<std::shared_ptr<DataSourceBase>>& GetDataSourceList()
      const {
    return data_source_list_;
  }
  const OperationMeta& GetOperationMeta() const { return operation_meta_; }

 private:
  std::string config_path_;
  OperationMeta operation_meta_;
  std::vector<std::shared_ptr<DataSourceBase>> data_source_list_;
};
}  // namespace perception_feature

#endif  // CORE_CONFIG_FEATURE_CONFIG_H_
