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

#ifndef CORE_PROCESSOR_FEATURE_EXTRACT_H_
#define CORE_PROCESSOR_FEATURE_EXTRACT_H_

#include <glog/logging.h>

#include "core/common/variable_table.h"
#include "core/config/operation_meta.h"
namespace perception_feature {
class FeatureExtract {
 public:
  static int Extract(const OperationMeta& operation_meta,
                     FeatureVariableTable& var_table);
};
}  // namespace perception_feature

#endif  // CORE_PROCESSOR_FEATURE_EXTRACT_H_
