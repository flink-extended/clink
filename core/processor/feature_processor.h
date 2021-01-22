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
#include <string>
#include <unordered_map>
#include <vector>

#include "core/config/feature_config.h"
#include "core/operators/operator_factory.h"
#include "core/processor/feature_plugin.h"
namespace perception_feature {
class FEATURE_DLL_DECL FeatureProcessor : public FeaturePlugin {
 public:
  FeatureProcessor();
  FeatureProcessor(const std::string& remote_url,
                   const std::string& local_path);
  virtual ~FeatureProcessor() noexcept;
  int LoadConfig(const std::string&) override;
  int ReloadConfig(const std::string&, bool first = false) override;
  int FeatureExtract(const FeatureList& feature_list, std::vector<float>& index,
                     std::vector<float>& value) override;
  int FeatureExtract(const std::string& input, std::vector<float>& index,
                     std::vector<float>& value) override;
  int FeatureExtract(const SampleRecord& sample_record,
                     std::vector<float>& index,
                     std::vector<float>& value) override;

 private:
  FeatureConfig* configs_[2];
  int current_config_index_;
  bool init_status_;
  const FeatureConfig* GetFeatureConfig() {
    return configs_[current_config_index_];
  }
};

}  // namespace perception_feature

#endif  // CORE_PROCESSOR_FEATURE_PROCESSOR_H_
