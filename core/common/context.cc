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
 
#include "core/common/context.h"

#include "core/source/source_parser_base.h"

namespace clink {

Context::Context(FeatureConfig* config) {
  google::protobuf::ArenaOptions options;
  options.initial_block_size = 512;
  options.start_block_size = 512;
  options.max_block_size = 65536;
  // arena_ = std::make_unique<google::protobuf::Arena>(options);
  arena_.Init(options);
  // config_ = config.get();
  parser_ = config->source_parser();
  config_ = config;
}

}  // namespace clink