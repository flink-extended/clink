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

#ifndef CORE_DATASOURCE_CSV_DATA_PARSER_H_
#define CORE_DATASOURCE_CSV_DATA_PARSER_H_
#include <glog/logging.h>
#include <rapidjson/rapidjson.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/variable_table.h"
#include "core/datasource/data_source_base.h"
#include "core/processor/feature_extract.h"
#include "core/utils/file_utils.h"
#include "core/utils/proto_json.h"
#include "core/utils/simplp_thread_pool.h"
#include "core/utils/string_utils.h"
namespace perception_feature {
class CsvDataParser : public DataSourceBase {
 public:
  explicit CsvDataParser(const DataSource& conf) : DataSourceBase(conf) {}
  virtual ~CsvDataParser() = default;
  bool Init() override { return true; }
  int LoadConfig(const std::string&) override;
  int ParseInputData(const FeatureList& feature_list,
                     FeatureVariableTable&) override;
  int ParseInputData(const std::string& input, FeatureVariableTable&) override;
  int ParseInputData(const SampleRecord& input, FeatureVariableTable&) override;
 private:
  CsvDataConfigList data_config_list_;
};
}  // namespace perception_feature

#endif  // CORE_DATASOURCE_CSV_DATA_PARSER_H_
