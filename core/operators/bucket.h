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

#ifndef CORE_OPERATORS_BUCKET_H_
#define CORE_OPERATORS_BUCKET_H_
#include <memory>
#include <string>
#include <vector>

#include "core/operands/variable.h"
#include "core/operators/unary_operator.h"
#include "core/utils/convert_util.h"
namespace clink {
class Bucket : public BaseOperator {
 public:
  Bucket();
  Bucket(const std::string& feature_name, const OpParamMap& param_map);

  const Feature* Evaluate(Context*) override;

  std::shared_ptr<BaseOperator> Clone() const override;

  bool ParseParamMap(const std::string& feature_name,
                     const OpParamMap& param_map) override;
  bool ParseParamMap(const std::vector<std::string>& variables,
                     const OpParamMap& param_map) override;

 private:
  int GetBucketIndex(const Feature& feature);
  
  inline void GetBucketIndex(const Feature& feature, int* index) {
    double value;
    ConvertUtil::ToDouble(feature, value);
    int vec_size = boundaries_.size();
    *index = vec_size;
    if (value < boundaries_.at(vec_size - 1)) {
      *index = std::upper_bound(boundaries_.begin(), boundaries_.end(), value) -
               boundaries_.begin();
    }
  }
  bool ParseParam(const std::string& feature_name, const OpParamMap& param_map);
  void ParseBucketBoundaries(const OpParam& param_map);
  std::string boundary_postfix = "_bucket_boundaries";
  std::string index_postfix = "_index_only";
  std::vector<double> boundaries_;
};
}  // namespace clink

#endif  // CORE_OPERATORS_BUCKET_H_
