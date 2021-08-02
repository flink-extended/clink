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

#ifndef CORE_DATASOURCE_SOURCE_BASE_H_
#define CORE_DATASOURCE_SOURCE_BASE_H_
#include <string>
#include <utility>

#include "core/common/common.h"
#include "core/common/context.h"
#include "core/common/sample_list.h"

namespace clink {

class Context;

class SourceParserBase {
 public:
  explicit SourceParserBase(const DataSource& data_conf)
      : data_source_(data_conf) {
    Init();
  }

  virtual ~SourceParserBase() = default;

  virtual bool Init() { return true; }

  virtual int LoadConfig(const std::string& conf_path) = 0;

  virtual int ParseInputData(const Sample& sample, Context*) = 0;

  virtual int ParseInputData(const std::string& input, Context*) = 0;

  virtual int ParseInputData(const SampleRecord& input, Context* context) = 0;

 protected:
  DataSource data_source_;
};

}  // namespace clink
#endif  // CORE_DATASOURCE_DATA_SOURCE_BASE_H_
