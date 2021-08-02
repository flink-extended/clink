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

#ifndef CORE_PROCESSOR_FEATURE_PROCESSOR_H_
#define CORE_PROCESSOR_FEATURE_PROCESSOR_H_
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/common/context.h"
#include "core/common/sample_list.h"
#include "core/processor/clink.h"
#include "core/utils/bthread_group.h"
#include "core/utils/simplp_thread_pool.h"
namespace clink {

class FeatureConfig;
struct TaskResult;
// struct FeatureResult;
// using TaskResult = std::vector<FeatureResult>;

// class FEATURE_DLL_DECL Processor : public FeaturePlugin {
class ClinkImpl {
 public:
  ClinkImpl();

  int LoadConfig(const std::string&);

  int LoadConfig(const std::string& remote_url, const std::string& config_path);

  template <typename T>
  int FeatureExtract(const T& input, std::vector<int>* index,
                     std::vector<float>* value);


 private:
  int ReloadConfig(const std::string&, bool first = false);

  std::unique_ptr<Context> BuildContext();

  int Extract(Context* context);

  void ExtractParallel(const OperationMetaItem* op_meta,
                       const FeatureItem* item, Context* context,
                       std::shared_ptr<TaskResult>* task_result);

  int BuildResponse(Context* context);

 private:
  std::vector<std::shared_ptr<FeatureConfig>> configs_;

  int current_config_index_;

  bool init_status_;

  class ExtractTask : public clink::BthreadGroup::BthreadTask {
   public:
    ExtractTask(std::function<void()>&& cb) : callback_(cb) {}
    inline void Run() override { callback_(); }

   private:
    std::function<void()> callback_;
  };  // ExtractTask

  std::unique_ptr<SimpleThreadPool> thread_pool_;
};

}  // namespace clink

#endif  // CORE_PROCESSOR_FEATURE_PROCESSOR_H_
