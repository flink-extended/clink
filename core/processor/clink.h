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

#ifndef CORE_PROCESSOR_CLINK_H_
#define CORE_PROCESSOR_CLINK_H_

#ifndef FEATURE_DLL_DECL
#if defined(_MSC_VER)
#define GFLAGS_DLL_DECL __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define FEATURE_DLL_DECL
#else
#define FEATURE_DLL_DECL
#endif
#endif
#include <memory>
#include <string>
#include <vector>

namespace clink {

class Sample;
class ClinkImpl;

namespace proto {
class SampleRecord;
}
using SampleRecord = proto::SampleRecord;

class Clink {
 public:
  Clink();

  int LoadConfig(const std::string &config_path);

  int LoadConfig(const std::string &remote_url, const std::string &config_path);

  int FeatureExtract(const Sample &sample, std::vector<uint32_t> *index,
                     std::vector<float> *value);

  int FeatureExtract(const std::string &input, std::vector<uint32_t> *index,
                     std::vector<float> *value);

  int FeatureExtract(const SampleRecord &input, std::vector<uint32_t> *index,
                     std::vector<float> *value);

 private:
  std::shared_ptr<ClinkImpl> clink_impl_;
};
extern "C" FEATURE_DLL_DECL Clink *load_plugin(void);

extern "C" FEATURE_DLL_DECL void destroy_plugin(Clink *p);
}  // namespace clink

extern "C" FEATURE_DLL_DECL int FeatureOfflineInit(const char *remote_url,
                                                   const char *local_path);

extern "C" FEATURE_DLL_DECL int FeatureExtractOffline(const char *input,
                                                      char **output);

extern "C" FEATURE_DLL_DECL int FeatureOfflineCleanUp(char *output);

#endif  // CORE_PROCESSOR_FEATURE_PLUGIN_H_
