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

#ifndef CORE_DATASOURCE_DATA_SOURCE_BASE_H_
#define CORE_DATASOURCE_DATA_SOURCE_BASE_H_
#include <string>
#include <utility>

#include "core/common/common.h"
#include "core/common/variable_table.h"
#include "core/processor/feature_list.h"
namespace perception_feature {
class DataSourceBase {
 public:
  explicit DataSourceBase(const DataSource& data_conf)
      : data_source_(data_conf) {
    Init();
  }
  virtual ~DataSourceBase() = default;

  virtual bool Init() { return true; }

  virtual int LoadConfig(const std::string& conf_path) = 0;
  virtual int ParseInputData(const FeatureList& feature_list,
                             FeatureVariableTable&) = 0;
  virtual int ParseInputData(const std::string& input,
                             FeatureVariableTable&) = 0;
  virtual int ParseInputData(const SampleRecord& input,
                             FeatureVariableTable&) = 0;
  //  virtual int FeatureExtract(const OperationMeta& operation_meta) =
  //  0;//离线抽取 virtual int FeatureExtract(const FeatureRequest&, const
  //  OperationMeta& operation_meta, FeatureResponse&) = 0;//在线抽取 virtual
  //  int FeatureExtract(const FeatureList& feature_list,const OperationMeta
  //  &operation_meta,std::vector<float>& index,std::vector<float>& value) = 0;
  //  virtual int FeatureExtract(const std::string& biz_name,const std::string&
  //  input,std::vector<float>& index,std::vector<float>& value) = 0;
  const DataSource& GetDataSourceConfig() { return data_source_; }

 protected:
  DataSource data_source_;
};

}  // namespace perception_feature
#endif  // CORE_DATASOURCE_DATA_SOURCE_BASE_H_
