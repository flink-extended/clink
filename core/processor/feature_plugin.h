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

#ifndef CORE_PROCESSOR_FEATURE_PLUGIN_H_
#define CORE_PROCESSOR_FEATURE_PLUGIN_H_

#ifndef FEATURE_DLL_DECL
#if defined(_MSC_VER)
#define GFLAGS_DLL_DECL __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define FEATURE_DLL_DECL __attribute__((visibility("default")))
#else
#define FEATURE_DLL_DECL
#endif
#endif
#include <string>
#include <vector>

#include "core/processor/feature_list.h"
#include "sample_list.h"
#include "feature_list.h"
// namespace proto{
// class FeatureRequest;
// class FeatureResponse;
//}
// using FeatureRequest = proto::FeatureRequest;
// using FeatureResponse = proto::FeatureResponse;

namespace perception_feature {
class FeaturePlugin {
 public:
  FeaturePlugin() = default;
  virtual ~FeaturePlugin() = default;
  virtual int LoadConfig(const std::string &) = 0;
  virtual int ReloadConfig(const std::string &, bool first = false) = 0;
  virtual int FeatureExtract(const FeatureList &feature_list,
                             std::vector<float> &index,
                             std::vector<float> &value) = 0;
  virtual int FeatureExtract(const std::string &input,
                             std::vector<float> &index,
                             std::vector<float> &value) = 0;

  virtual int FeatureExtract(const SampleRecord& sample_record,
                     std::vector<float>& index,
                     std::vector<float>& value) = 0;
};
extern "C" FEATURE_DLL_DECL FeaturePlugin *load_plugin(void);
extern "C" FEATURE_DLL_DECL void destroy_plugin(FeaturePlugin *p);
}  // namespace perception_feature

extern "C" FEATURE_DLL_DECL int FeatureExtractOffline(const char *remote_url,
                                                      const char *local_path,
                                                      const char *input,
                                                      char **output);
extern "C" FEATURE_DLL_DECL int FeatureOfflineCleanUp(char *output);

#endif  // CORE_PROCESSOR_FEATURE_PLUGIN_H_
