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

#include "core/datasource/csv_data_parser.h"

#include <fstream>
#include <memory>
#include <vector>

#include "core/operators/base_operator.h"
#include "core/utils/convert_util.h"
#include "core/utils/feature_util.h"
#include "core/utils/proto_json.h"
#include "core/utils/string_utils.h"
namespace perception_feature {
int CsvDataParser::LoadConfig(const std::string &conf_path) {
  std::string data_source_config = conf_path + "/" + data_source_.data_conf();
  std::ifstream ifs;
  std::string json_str;
  ifs.open(data_source_config, std::ios::in | std::ios::binary);
  json_str.assign(std::istreambuf_iterator<char>(ifs),
                  std::istreambuf_iterator<char>());
  ifs.close();
  CsvDataConfigList csv_data_list;
  int res = ProtoJson::json2pb(json_str, csv_data_list, true);
  if (res != STATUS_OK) {
    LOG(ERROR) << "Parse config file  error, conf:" << data_source_config;
    return res;
  }
  if (csv_data_list.data_path().empty() || csv_data_list.separator().empty() ||
      csv_data_list.config_list_size() == 0) {
    LOG(ERROR) << "Parse config file  error, conf:" << data_source_config;
    return ERR_DATASOURCE_CONFIG;
  }
  data_config_list_ = csv_data_list;
  return STATUS_OK;
}

int CsvDataParser::ParseInputData(const std::string &input,
                                  FeatureVariableTable &var_table) {
  std::vector<std::string> input_vec;
  StringUtils::Split(input, data_config_list_.separator(), input_vec);
  int col_size = data_config_list_.config_list_size();
  if (col_size <= 0) {
    LOG(INFO) << "data config list is empty";
    return ERR_PARSE_INPUT_DATA;
  }
  if (input_vec.size() != col_size) {
    LOG(INFO) << "input size error,actual size: " << input_vec.size()
              << " expected size :" << col_size;
    return ERR_PARSE_INPUT_DATA;
  }
  var_table.ReserveSize(col_size * 2);
  for (auto &it : data_config_list_.config_list()) {
    int index = it.column();
    if (index < 0 || index >= input_vec.size()) {
      continue;
    }
    std::string value = input_vec[index];
    std::shared_ptr<Feature> feature = std::make_shared<Feature>();
    switch (it.feature_data_type()) {
      case proto::REAL_TYPE:
        if (!StringUtils::IsNumber(value)) {
          LOG(INFO) << "value " << value << "is not a number";
          continue;
        }
        feature->mutable_double_list()->add_value(stod(value));
        var_table.Insert(it.feature_name(), feature);
        break;
      case proto::INTEGER_TYPE:
        if (!StringUtils::IsNumber(value)) {
          LOG(INFO) << "value " << value << "is not a number";
          continue;
        }
        feature->mutable_int64_list()->add_value(stoll(value));
        var_table.Insert(it.feature_name(), feature);
        break;
      case proto::STRING_TYPE:
        feature->mutable_bytes_list()->add_value(value);
        var_table.Insert(it.feature_name(), feature);
        break;
      default:
        break;
    }
  }
  return STATUS_OK;
}

int CsvDataParser::ParseInputData(const FeatureList &feature_list,
                                  FeatureVariableTable &var_table) {
  int col_size = data_config_list_.config_list_size();
  if (feature_list.Size() != col_size) {
    LOG(INFO) << "input size error,actual size: " << feature_list.Size()
              << " expected size :" << col_size;
    return ERR_PARSE_INPUT_DATA;
  }
  var_table.ReserveSize(col_size * 2);
  for (auto &it : data_config_list_.config_list()) {
    int index = it.column();
    if (index >= col_size) {
      continue;
    }
    const std::shared_ptr<Feature> &feature = feature_list.GetValue(index);
    if (feature == nullptr) {
      continue;
    }
    // if (!feature->IsInitialized()) continue;
    var_table.Insert(it.feature_name(), feature);
  }
  return STATUS_OK;
}
int CsvDataParser::ParseInputData(const SampleRecord &input,
                                  FeatureVariableTable &var_table) {
  int col_size = data_config_list_.config_list_size();
  if (input.feature_list_size() != col_size) {
    LOG(INFO) << "input size error,actual size: " << input.feature_list_size()
              << " expected size :" << col_size;
    return ERR_PARSE_INPUT_DATA;
  }
  var_table.ReserveSize(col_size * 2);
  for (auto &it : data_config_list_.config_list()) {
    int index = it.column();
    if (index >= col_size) {
      continue;
    }
    const Feature &feature = input.feature_list().Get(index);

    std::shared_ptr<Feature> feature_ptr = std::make_shared<Feature>();
    feature_ptr->CopyFrom(feature);
    // if (!feature->IsInitialized()) continue;
    var_table.Insert(it.feature_name(), feature_ptr);
  }
  return STATUS_OK;
}

}  // namespace perception_feature
