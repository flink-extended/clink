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

#ifndef CORE_OPERATORS_TO_VECTOR_H_
#define CORE_OPERATORS_TO_VECTOR_H_
#include <iostream>
#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/operators/base_operator.h"
#include "core/operators/unary_operator.h"
#include "core/utils/convert_util.h"
namespace clink {
class ToVector : public UnaryOperator {
 public:
  ToVector();

  ToVector(const std::string& feature_name, const OpParamMap& param_map);

  const Feature* Evaluate(Context*) override;

  std::shared_ptr<BaseOperator> Clone() const override;

  bool ParseParamMap(const std::string& name,
                     const OpParamMap& param_map) override;

  //  protected:
  //   ToVector(const ToVector&);

 private:
  int vec_size_;

  std::string separator_;  //

  proto::FeatureDataType vec_data_type_;

  bool ParseParam(const std::string& name, const OpParamMap& param_map);

  void Transform(const std::string& input, Feature* output);
};
}  // namespace clink

#endif  // CORE_OPERATORS_TO_VECTOR_H_
