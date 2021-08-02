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

#include "core/config/operation_meta.h"

#include <butil/logging.h>

#include <fstream>
#include <unordered_set>
#include <utility>

#include "core/utils/proto_json.h"

namespace clink {
static const char OPERATION_ENTRY[] = "operation.conf";

OperationMeta::OperationMeta() {
  sorter_ = std::make_unique<utils::TopSort<std::string>>();
}

OperationMeta::~OperationMeta() { Reset(); }

void OperationMeta::Reset() { operation_map_.clear(); }

int OperationMeta::LoadOperation(const std::string& config_path) {
  std::string config_file = config_path + "/" + OPERATION_ENTRY;
  std::ifstream ifs;
  std::string json_str;
  ifs.open(config_file, std::ios::in | std::ios::binary);
  json_str.assign(std::istreambuf_iterator<char>(ifs),
                  std::istreambuf_iterator<char>());
  ifs.close();
  if (json_str.empty()) {
    return ERR_CONFIG_EMPTY;
  }
  OperationList operation_list;
  int res = ProtoJson::json2pb(json_str, operation_list, true);
  if (res != STATUS_OK) {
    return res;
  }
  if (!operation_list.IsInitialized() || operation_list.operation_size() == 0) {
    return ERR_OPERATION_EMPTY;
  }

  // auto it = operation_list.operation().begin();

  for (auto& it : operation_list.operation()) {
    auto operation_meta_item = std::make_shared<OperationMetaItem>();
    res = operation_meta_item->Init(it);
    if (res != STATUS_OK) {
      continue;
    }
    res = AddOperation(operation_meta_item);
    if (res != STATUS_OK) {
      LOG(ERROR) << "fail to AddOperation, feature: " << it.output_feature()
                 << "error no" << res;
      continue;
    }
    output_sequence_.emplace_back(it.output_feature());
  }
  output_type_ = operation_list.output_format();
  //生成拓扑图
  if (!TopSort()) {
    return ERR_TOP_SORT;
  }
  return STATUS_OK;
}

int OperationMeta::AddOperation(
    std::shared_ptr<OperationMetaItem>& meta_item) {  //添加特征
  if (operation_map_.find(meta_item->output_feature().id()) !=
      operation_map_.end()) {
    return ERR_DUPLICATE_FEATURE;
  }
  total_feature_size_ += meta_item->feature_size();
  operation_map_.insert(
      std::make_pair(meta_item->output_feature().id(), meta_item));

  //添加拓扑关系
  for (auto& input : meta_item->input_features()) {
    sorter_->AddRelation(input.name(), meta_item->output_feature().name());
  }
  return STATUS_OK;
}

bool OperationMeta::TopSort() {
  if (!sorter_->BfsTraversal()) {
    LOG(ERROR) << "fail to complete topsort";
    return false;
  }
  auto result = sorter_->GetParallelSequence();
  for (auto& iter : result) {
    std::vector<std::shared_ptr<FeatureItem>> level_sequence;
    for (auto& item : iter) {
      level_sequence.emplace_back(
          std::move(std::make_shared<FeatureItem>(item)));
    }
    extract_sequence_.emplace_back(level_sequence);
  }
  return true;
}

}  // namespace clink
