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

#include "core/processor/feature_extract.h"

#include <iostream>
#include <memory>

#include "core/utils/convert_util.h"
namespace perception_feature {
int FeatureExtract::Extract(const OperationMeta& operation_meta,
                            FeatureVariableTable& var_table) {
  if (var_table.Empty()) {
    return ERR_OPERATION_EMPTY;
  }
  std::shared_ptr<Feature> output;
  for (auto& iter : operation_meta.GetFeatureItemSequence()) {
    auto it = operation_meta.GetOperationMap().find(iter.Id());
    if (it == operation_meta.GetOperationMap().end()) {
      continue;
    }
    const std::shared_ptr<OperationNode>& tree = it->second.GetExpressionTree();
    if (tree == nullptr) {
      LOG(INFO) << "feature " << iter.Name() << "expression tree is empty";
      continue;
    }
    output = nullptr;
    int eva_result = tree->Evaluate(var_table.GetFeatureMap(), output);
    if (eva_result != STATUS_OK || output == nullptr) {
      LOG(INFO) << "feature " << iter.Name() << " evaluate error ,error no "
                << eva_result;
      continue;
      // return eva_result;
    }
    var_table.SetValue(iter.Id(), output);
  }
  return STATUS_OK;
}

}  // namespace perception_feature
