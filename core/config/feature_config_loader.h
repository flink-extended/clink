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

#ifndef CORE_CONFIG_FEATURE_CONFIG_LOADER_H_
#define CORE_CONFIG_FEATURE_CONFIG_LOADER_H_

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/config/operation_meta.h"
#include "core/datasource/data_source_base.h"
#include "core/operators/operator_factory.h"
#include "core/processor/expression_builder.h"
#include "core/utils/proto_json.h"
namespace perception_feature {
class FeatureConfigLoader {
 public:
  FeatureConfigLoader();
  virtual ~FeatureConfigLoader() = default;
  void SetConfigPath(const std::string& config_path) {
    config_path_ = config_path;
  }
  int LoadDataSourceConfig(std::vector<std::shared_ptr<DataSourceBase>>&
                               data_source_list_);  //数据源配置
  int LoadOperationMeta(OperationMeta& operation_meta);

 private:
  std::string config_path_;
  bool ParseTransform(const Transform&, OperationMetaItem& meta);
  OperatorFactory operator_factory_;
};
}  // namespace perception_feature

#endif  // CORE_CONFIG_FEATURE_CONFIG_LOADER_H_
