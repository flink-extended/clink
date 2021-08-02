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

#include "core/processor/clink_impl.h"

#include <butil/logging.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include "core/common/context.h"
#include "core/config/feature_config.h"
#include "core/processor/clink.h"
//#include "core/clinkimpl/feature_extract.h"
#include <gperftools/profiler.h>

#include "core/common/operation_node.h"
#include "core/source/source_parser_base.h"
#include "core/utils/config_manager.h"
#include "core/utils/convert_util.h"
#include "core/utils/feature_util.h"
// #include "omp.h"
namespace clink {

struct TaskResult {
  const Feature *output;
  const FeatureItem *item;
  TaskResult(const FeatureItem *item, const Feature *output) {
    this->item = item;
    this->output = output;
  }
};

ClinkImpl::ClinkImpl() : current_config_index_(0) {
  configs_.emplace_back(std::make_shared<FeatureConfig>());
  configs_.emplace_back(std::make_shared<FeatureConfig>());
  thread_pool_ = std::make_unique<SimpleThreadPool>(5);
}

std::unique_ptr<Context> ClinkImpl::BuildContext() {
  auto config = configs_[current_config_index_];

  if (UNLIKELY(config == nullptr)) {
    LOG(ERROR) << "No feature config found ";
    return nullptr;
  }

  if (UNLIKELY(config->source_parser() == nullptr)) {
    LOG(ERROR) << "No data source config found ";
    return nullptr;
  }
  return std::make_unique<Context>(config.get());
}

int ClinkImpl::LoadConfig(const std::string &config_path) {
  int res = ReloadConfig(config_path, true);
  if (res == STATUS_OK) {
    init_status_ = true;
  } else {
    init_status_ = false;
  }
  return res;
}

int ClinkImpl::LoadConfig(const std::string &remote_url,
                          const std::string &config_path) {
  int res;
  if (!remote_url.empty()) {
    res = ConfigManager::FetchAndExtractConfig(remote_url, config_path);
    if (res != STATUS_OK) {
      LOG(WARNING) << "Fail to fetch feature config from:" << remote_url
                   << " error code :" << res;
      return res;
    } else {
      LOG(INFO) << "Fetch feature config from " << remote_url
                << " successed,local path" << config_path;
    }
  }

  res = ReloadConfig(config_path, true);
  if (res == STATUS_OK) {
    init_status_ = true;
  } else {
    init_status_ = false;
  }
  return res;
}

int ClinkImpl::ReloadConfig(const std::string &config_path, bool first) {
  // std::string conf_path = config_path ;
  auto new_config = std::make_shared<FeatureConfig>();
  int res = new_config->LoadConfig(config_path);
  if (res != STATUS_OK) {
    LOG(WARNING) << "Load feature config error:" << res;
    return res;
  }
  current_config_index_ = 1 - current_config_index_;
  configs_[current_config_index_] = new_config;
  return STATUS_OK;
}

template <typename T>
int ClinkImpl::FeatureExtract(const T &input, std::vector<int> *index,
                              std::vector<float> *value) {
  index->clear();
  value->clear();
  if (!init_status_) {
    return ERR_NOT_INIT;
  }
  auto context = BuildContext();
  const int &size = context->config()->operation_meta()->total_feature_size();
  index->reserve(size);
  value->reserve(size);
  if (context == nullptr) {
    return ERR_INVALID_CONFIG;
  }
  context->parser()->ParseInputData(input, context.get());
  int extract_status = Extract(context.get());
  if (extract_status != STATUS_OK) {
    return extract_status;
  }

  FeatureUtil::BuildResponse(context.get(), index, value);
  // for (int i = 0; i < index->size(); i++) {
  //   std::cout << index->at(i) << " " << value->at(i) << std::endl;
  // }
  return STATUS_OK;
}

int ClinkImpl::Extract(Context *context) {
  auto operation_meta = context->config()->operation_meta();
  auto &extract_sequence = operation_meta->extract_sequence();
  for (auto &sequence : extract_sequence) {
    std::vector<std::shared_ptr<TaskResult>> results(sequence.size(),
                                                     std::move(nullptr));

    // std::vector<std::future<void>> feature_results;
    //#pragma omp parallel for
    for (int i = 0; i < sequence.size(); ++i) {
      auto &item = sequence.at(i);
      auto operation_meta_item = operation_meta->GetOperationMetaItem(*item);
      if (operation_meta_item == nullptr) {
        continue;
      }
      // feature_results.emplace_back(thread_pool_->enqueue(
      //     std::bind(&ClinkImpl::ExtractParallel, this, operation_meta_item,
      //               item.get(), context, &results[i])));

      ExtractParallel(operation_meta_item, item.get(), context, &results[i]);
    }
    // for (auto &&r : feature_results) {
    //   r.get();
    // }
    // bg.Start();
    for (auto &result : results) {
      if (result != nullptr) {
        context->Set(result->item->id(), result->output);
      }
    }
  }
  return STATUS_OK;
}

void ClinkImpl::ExtractParallel(const OperationMetaItem *op_meta_item,
                                const FeatureItem *item, Context *context,
                                std::shared_ptr<TaskResult> *task_result) {
  *task_result = nullptr;
  if (context == nullptr || op_meta_item == nullptr || item == nullptr) {
    return;
  }
  *task_result = std::make_unique<TaskResult>(item, nullptr);
  auto &tree = op_meta_item->expression_tree();
  if (tree == nullptr) {
    LOG(INFO) << "feature " << op_meta_item->output_feature().name()
              << "expression tree is empty";
    return;
  }
  auto result = tree->Evaluate(context);
  // FeatureResult feature_result(result, item.get());
  (*task_result)->output = result;
}

template int ClinkImpl::FeatureExtract(const std::string &input,
                                       std::vector<int> *index,
                                       std::vector<float> *value);

template int ClinkImpl::FeatureExtract(const Sample &input,
                                       std::vector<int> *index,
                                       std::vector<float> *value);

template int ClinkImpl::FeatureExtract(const SampleRecord &input,
                                       std::vector<int> *index,
                                       std::vector<float> *value);
}  // namespace clink
