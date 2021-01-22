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

#include "core/config/feature_config_loader.h"

#include <utility>

#include "core/datasource/csv_data_parser.h"
#include "core/utils/proto_json.h"
namespace perception_feature {
static const char OPERATION_ENTRY[] = "operation.conf";

static const char DATA_SOURCE_CONFIG[] = "datasource.conf";

FeatureConfigLoader::FeatureConfigLoader() { operator_factory_.Init(); }

int FeatureConfigLoader::LoadDataSourceConfig(
    std::vector<std::shared_ptr<DataSourceBase>>& data_source_list_) {
  std::string data_source_config = config_path_ + "/" + DATA_SOURCE_CONFIG;
  std::ifstream ifs;
  std::string json_str;
  ifs.open(data_source_config, std::ios::in | std::ios::binary);
  json_str.assign(std::istreambuf_iterator<char>(ifs),
                  std::istreambuf_iterator<char>());
  ifs.close();
  proto::DataSourceList data_source_list;
  //  json2pb::Json2PbOptions json_2_pb_options;
  //  json_2_pb_options.base64_to_bytes = true;
  //  std::string err_msg;
  int parse_status = ProtoJson::json2pb(json_str, data_source_list, true);
  if (parse_status != STATUS_OK) {
    LOG(ERROR) << "Parse config file  error, conf:" << data_source_config;
    return parse_status;
  }
  if (!data_source_list.IsInitialized() ||
      data_source_list.data_source_size() <= 0) {
    return ERR_DATASOURCE_EMPTY;
  }
  for (auto& it : data_source_list.data_source()) {
    if (it.data_type() == proto::CSV_DATA && !it.data_conf().empty()) {
      std::shared_ptr<CsvDataParser> file_data_source =
          std::make_shared<CsvDataParser>(it);
      parse_status = file_data_source->LoadConfig(config_path_);
      if (parse_status != STATUS_OK) {
        LOG(ERROR) << "failed to load data data_source, parse error";
        return parse_status;
      }
      data_source_list_.emplace_back(file_data_source);
    }
  }
  return STATUS_OK;
}

int FeatureConfigLoader::LoadOperationMeta(OperationMeta& operation_meta) {
  std::string feature_config_file = config_path_ + "/" + OPERATION_ENTRY;
  std::ifstream ifs;
  std::string json_str;
  ifs.open(feature_config_file, std::ios::in | std::ios::binary);
  json_str.assign(std::istreambuf_iterator<char>(ifs),
                  std::istreambuf_iterator<char>());
  ifs.close();
  OperationList operation_list;
  if (json_str.empty()) {
    return ERR_CONFIG_EMPTY;
  }
  int res = ProtoJson::json2pb(json_str, operation_list, true);
  if (res != STATUS_OK) {
    return res;
  }
  if (!operation_list.IsInitialized() || operation_list.operation_size() == 0) {
    return ERR_OPERATION_EMPTY;
  }
  auto it = operation_list.operation().begin();
  OperationMetaItem operation_meta_item;
  while (it != operation_list.operation().end()) {
    operation_meta_item.Reset();
    for (int i = 0; i < it->input_features_size(); ++i) {
      operation_meta_item.AddInputFeatures(it->input_features(i));
    }
    operation_meta_item.SetOutputFeature(it->output_feature());
    operation_meta_item.SetFeatureSize(it->feature_size());
    operation_meta_item.SetOutputFeatureType(it->output_feature_type());
    if (it->transform_size() <= 0) {
      LOG(ERROR) << "parse feature transform error, feature:"
                 << it->output_feature();
      return ERR_PARSE_TRANSFORM;
    }
    Transform transform = it->transform(0);
    if (!ParseTransform(it->transform(0), operation_meta_item)) {
      LOG(ERROR) << "fail to parse transform config, feature: "
                 << it->output_feature() << " error no " << ERR_PARSE_TRANSFORM;
      ++it;
      continue;
    }
    res = operation_meta.AddOperation(operation_meta_item.GetOutputFeature(),
                                      operation_meta_item);
    if (res != STATUS_OK) {
      LOG(ERROR) << "fail to AddOperation, feature: " << it->output_feature()
                 << "error no" << res;
      ++it;
      continue;
    }
    operation_meta.AddOutputSequence(it->output_feature());
    operation_meta.AddFeatureRelation(operation_meta_item.GetInputFeatures(),
                                      operation_meta_item.GetOutputFeature());
    ++it;
  }
  operation_meta.SetOutputFromat(operation_list.output_format());
  //生成拓扑图
  if (!operation_meta.BfsTraverse()) {
    return ERR_TOP_SORT;
  }
  return STATUS_OK;
}

bool FeatureConfigLoader::ParseTransform(const Transform& transform,
                                         OperationMetaItem& meta) {
  if (!transform.IsInitialized() || transform.formula().empty()) {
    return false;
  }
  //参数转换
  OpParamMap op_param_map;
  op_param_map.clear();
  if (transform.params_size() > 0) {
    for (auto& it : transform.params()) {
      op_param_map.insert(std::make_pair(it.key(), it.value()));
    }
  }

  //构建表达式树
  std::shared_ptr<OperationNode> expression_tree =
      ExpressionBuilder::BuildExpressionTree(transform.formula(), op_param_map,
                                             operator_factory_);
  if (expression_tree == nullptr) {
    LOG(ERROR) << "fail to build expression tree for " << transform.formula();
    return false;
  }
  meta.SetExpressionTree(expression_tree);
  return true;
}

}  // namespace perception_feature
