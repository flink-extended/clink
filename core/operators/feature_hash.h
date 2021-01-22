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

#ifndef CORE_OPERATORS_FEATURE_HASH_H_
#define CORE_OPERATORS_FEATURE_HASH_H_
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/operators/base_operator.h"
namespace perception_feature {
class FeatureHash : public BaseOperator {
 public:
  FeatureHash();
  FeatureHash(const std::string& feature_name, const OpParamMap& param_map);
  int Evaluate(const FeatureMap&, std::shared_ptr<Feature>&) override;
  std::shared_ptr<BaseOperator> Clone() const override;
  bool ParseParamMap(const std::string& name,
                     const OpParamMap& param_map) override;

  bool ParseParamMap(const std::vector<std::string>& variables,
                     const OpParamMap& param_map) override;

 protected:
  FeatureHash(const FeatureHash&);

 private:
  int bin_size_;  //哈希分桶大小
  bool ParseParam(const std::string& name, const OpParamMap& param_map);
};
}  // namespace perception_feature

#endif  // CORE_OPERATORS_FEATURE_HASH_H_
