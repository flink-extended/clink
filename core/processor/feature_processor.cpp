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

#include "core/processor/feature_processor.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "core/processor/feature_extract.h"
#include "core/processor/feature_plugin.h"
#include "core/utils/config_manager.h"
#include "core/utils/convert_util.h"
#include "core/utils/feature_util.h"
namespace perception_feature {
FeatureProcessor::FeatureProcessor() : current_config_index_(0) {
  configs_[0] = configs_[1] = nullptr;
}
FeatureProcessor::FeatureProcessor(const std::string &remote_url,
                                   const std::string &local_path) {
  int res = ConfigManager::FetchAndExtractConfig(remote_url, local_path);
  if (res == STATUS_OK) {
    init_status_ = true;
  } else {
    init_status_ = false;
  }
  ReloadConfig(local_path, true);
}
FeatureProcessor::~FeatureProcessor() noexcept {
  if (configs_[0] != nullptr) {
    delete configs_[0];
    configs_[0] = nullptr;
  }
  if (configs_[1] != nullptr) {
    delete configs_[1];
    configs_[1] = nullptr;
  }
}
int FeatureProcessor::LoadConfig(const std::string &config_path) {
  int res = ReloadConfig(config_path, true);
  if (res == STATUS_OK) {
    init_status_ = true;
  } else {
    init_status_ = false;
  }
  return res;
}
int FeatureProcessor::ReloadConfig(const std::string &config_path, bool first) {
  std::string conf_path = config_path + "/conf";
  auto *new_config = new FeatureConfig();
  int res = new_config->LoadConfig(conf_path);
  if (res != STATUS_OK) {
    LOG(WARNING) << "Load feature config error:" << res;
    delete new_config;
    return res;
  }

  FeatureConfig *old_config = configs_[current_config_index_];
  configs_[1 - current_config_index_] = new_config;
  int last_index = current_config_index_;
  current_config_index_ = 1 - current_config_index_;
  if (!first) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    delete old_config;
  }
  LOG(INFO) << "reload config success,current_config_index:"
            << current_config_index_ << "last_index " << last_index;

  configs_[last_index] = nullptr;
  return STATUS_OK;
}

int FeatureProcessor::FeatureExtract(const FeatureList &feature_list,
                                     std::vector<float> &index,
                                     std::vector<float> &value) {
  index.clear();
  value.clear();
  if (!init_status_) {
    return ERR_NOT_INIT;
  }
  if (feature_list.Empty()) {
    return ERR_EMPTY_INPUT;
  }
  const FeatureConfig *config = GetFeatureConfig();
  if (config == nullptr) {
    LOG(ERROR) << "No feature config found ";
    return ERR_INVALID_CONFIG;
  }
  if (config->GetDataSourceList().empty()) {
    LOG(ERROR) << "No data source config found ";
    return ERR_INVALID_CONFIG;
  }
  auto &iter = config->GetDataSourceList().at(0);
  //  const int& size = config->GetOperationMeta().GetTotalFeatureSize();
  //  index.reserve(size);
  //  value.reserve(size);
  FeatureVariableTable var_table;
  iter->ParseInputData(feature_list, var_table);
  int extract_status =
      FeatureExtract::Extract(config->GetOperationMeta(), var_table);
  if (extract_status != STATUS_OK) {
    return extract_status;
  }
  FeatureUtil::BuildResponse(config->GetOperationMeta(), var_table, index,
                             value);
  return STATUS_OK;
}
int FeatureProcessor::FeatureExtract(const std::string &input,
                                     std::vector<float> &index,
                                     std::vector<float> &value) {
  index.clear();
  value.clear();
  if (!init_status_) {
    return ERR_NOT_INIT;
  }
  if (input.empty()) {
    return ERR_EMPTY_INPUT;
  }
  const FeatureConfig *config = GetFeatureConfig();
  if (config == nullptr) {
    LOG(ERROR) << "No feature config found ";
    return ERR_INVALID_CONFIG;
  }
  if (config->GetDataSourceList().empty()) {
    LOG(ERROR) << "No data source config found ";
    return ERR_INVALID_CONFIG;
  }
  auto &iter = config->GetDataSourceList().at(0);
  FeatureVariableTable var_table;
  iter->ParseInputData(input, var_table);
  int extract_status =
      FeatureExtract::Extract(config->GetOperationMeta(), var_table);
  if (extract_status != STATUS_OK) {
    return extract_status;
  }
  FeatureUtil::BuildResponse(config->GetOperationMeta(), var_table, index,
                             value);
  return STATUS_OK;
}
int FeatureProcessor::FeatureExtract(const SampleRecord &sample_record,
                                     std::vector<float> &index,
                                     std::vector<float> &value) {
  index.clear();
  value.clear();
  if (!init_status_) {
    return ERR_NOT_INIT;
  }
  if (sample_record.feature_list_size() == 0) {
    return ERR_EMPTY_INPUT;
  }
  const FeatureConfig *config = GetFeatureConfig();
  if (config == nullptr) {
    LOG(ERROR) << "No feature config found ";
    return ERR_INVALID_CONFIG;
  }
  if (config->GetDataSourceList().empty()) {
    LOG(ERROR) << "No data source config found ";
    return ERR_INVALID_CONFIG;
  }
  auto &iter = config->GetDataSourceList().at(0);
  FeatureVariableTable var_table;
  iter->ParseInputData(sample_record, var_table);
  int extract_status =
      FeatureExtract::Extract(config->GetOperationMeta(), var_table);
  if (extract_status != STATUS_OK) {
    return extract_status;
  }
  FeatureUtil::BuildResponse(config->GetOperationMeta(), var_table, index,
                             value);
  return STATUS_OK;
}

extern "C" FEATURE_DLL_DECL FeaturePlugin *load_plugin(void) {
  return new FeatureProcessor;
}
extern "C" FEATURE_DLL_DECL void destroy_plugin(FeaturePlugin *p) { delete p; }

}  // namespace perception_feature

extern "C" FEATURE_DLL_DECL int FeatureExtractOffline(const char *remote_url,
                                                      const char *local_path,
                                                      const char *input,
                                                      char **output) {
  static perception_feature::FeatureProcessor processor(remote_url, local_path);
  std::vector<float> index, value;
  int result = processor.FeatureExtract(input, index, value);
  if (result != perception_feature::STATUS_OK) return result;
  std::ostringstream out;
  if (index.size() != value.size()) {
    return perception_feature::ERR_INDEX_VALUE_UNEQUAL;
  }
  for (int i = 0; i < index.size(); ++i) {
    out << index.at(i) << ":" << value.at(i) << " ";
  }
  *output = reinterpret_cast<char *>(malloc(out.str().size() + 1));
  strcpy(*output, out.str().c_str());
  return result;
}

extern "C" FEATURE_DLL_DECL int FeatureOfflineCleanUp(char *output) {
  if (output) {
    free(output);
  }
  return 0;
}
