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

#include "core/source/csv_source_parser.h"

#include <fstream>
#include <memory>
#include <vector>

#include "core/common/context.h"
#include "core/common/sample_list.h"
#include "core/operators/base_operator.h"
#include "core/utils/convert_util.h"
#include "core/utils/feature_util.h"
#include "core/utils/proto_json.h"
#include "core/utils/string_utils.h"

namespace clink {
int CsvSourceParser::LoadConfig(const std::string &conf_path) {
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
  data_config_list_.Swap(&csv_data_list);
  return STATUS_OK;
}

int CsvSourceParser::ParseInputData(const std::string &input,
                                    Context *context) {
  std::vector<std::string> input_vec;
  StringUtils::Split(input, data_config_list_.separator(), &input_vec);
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

  for (auto &it : data_config_list_.config_list()) {
    int index = it.column();
    if (index < 0 || index >= input_vec.size()) {
      continue;
    }
    std::string value = input_vec[index];
    Feature *feature = context->CreateMessage();
    switch (it.feature_data_type()) {
      case proto::REAL_TYPE:
        GetFeatureValues<double>(feature)->Add(strtod(value.c_str(), nullptr));
        context->Set(it.feature_name(), feature);
        break;
      case proto::INTEGER_TYPE:
        GetFeatureValues<int64_t>(feature)->Add(
            strtoll(value.c_str(), nullptr, 10));
        context->Set(it.feature_name(), feature);
        break;
      case proto::STRING_TYPE:
        (*GetFeatureValues<std::string>(feature)->Add()) = value;
        context->Set(it.feature_name(), feature);
        break;
      default:
        break;
    }
  }
  return STATUS_OK;
}

int CsvSourceParser::ParseInputData(const Sample &input, Context *context) {
  int col_size = data_config_list_.config_list_size();
  if (input.Size() != col_size) {
    LOG(INFO) << "input size error,actual size: " << input.Size()
              << " expected size :" << col_size;
    return ERR_PARSE_INPUT_DATA;
  }

  // var_table.ReserveSize(col_size * 2);
  for (auto &it : data_config_list_.config_list()) {
    int index = it.column();
    if (index >= col_size) {
      continue;
    }
    auto feature = input.Get(index);
    if (feature == nullptr) {
      continue;
    }
    context->Set(it.feature_name(), feature);
  }
  return STATUS_OK;
}

int CsvSourceParser::ParseInputData(const SampleRecord &input,
                                    Context *context) {
  int col_size = data_config_list_.config_list_size();
  if (input.features_size() != col_size) {
    LOG(INFO) << "input size error,actual size: " << input.features_size()
              << " expected size :" << col_size;
    return ERR_PARSE_INPUT_DATA;
  }

  for (auto &it : data_config_list_.config_list()) {
    int index = it.column();
    if (input.features().contains(index)) {
      auto &iter = input.features().at(index);
      context->Set(it.feature_name(), &iter);
    }
  }
  return STATUS_OK;
}

}  // namespace clink
